import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_cost_fixture import choose_rows,float_bits,archived_control
from shaping_completion import sha


class CostFixtureTests(unittest.TestCase):
    def test_fixed_rows_never_enter_nontraining_membership(self):
        populations=[str(i//10) for i in range(50)]
        train=[i for i in range(50) if i%10!=0]
        chosen=choose_rows(list(reversed(train)),populations)
        self.assertEqual(chosen,[i for i in range(50) if 1<=i%10<=8])
        self.assertTrue(set(chosen)<=set(train))
        with self.assertRaises(ValueError):choose_rows(train[:8],populations)
        self.assertNotEqual(float_bits(0.),float_bits(-0.))

    def test_control_legacy_archive_requires_both_content_copies(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            root=Path(tmp);study=root/'study';archive=root/'tree-inner-archive';dest=root/'mirror'
            path=study/'runs/c/attempt-001/evaluation/predictions.npz'
            path.parent.mkdir(parents=True);path.write_bytes(b'preserved native evidence')
            archive.mkdir();other=dest/path.relative_to(study);other.parent.mkdir(parents=True);other.write_bytes(path.read_bytes())
            manifest=f'{sha(path)}  {path.relative_to(study)}\n'
            (archive/'SHA256SUMS').write_text(manifest);(dest/'SHA256SUMS').write_text(manifest)
            (archive/'MIRROR_VERIFIED.json').write_text(json.dumps(dict(destination=str(dest),all_artifact_sha256_verified=True)))
            pins=archived_control(study,'extra_trees',[path]);self.assertEqual(pins[str(path)],sha(other))
            other.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'content differs'):archived_control(study,'extra_trees',[path])


if __name__=='__main__':unittest.main()
