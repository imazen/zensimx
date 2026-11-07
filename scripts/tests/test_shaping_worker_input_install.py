import io,json,sys,tarfile,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_worker_input_install import install,install_shared,sha


class InstallTests(unittest.TestCase):
    def test_exact_install_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study=root/'study';target=study/'comparators/inner-0-0/table'
            data=b'frozen inputs';source=root/'source';source.write_bytes(data)
            manifest=root/'INPUTS.json';manifest.write_text(json.dumps([dict(path=str(target),bytes=len(data),sha256=sha(source))]))
            bundle=root/'inputs.tar'
            with tarfile.open(bundle,'w') as tar:
                info=tarfile.TarInfo(str(target).lstrip('/'));info.size=len(data);info.mode=0o755;tar.addfile(info,io.BytesIO(data))
            args=(study,'inner-0-0',study/'worker_assignments/assignment.json',bundle,manifest,sha(bundle),sha(manifest))
            with self.assertRaisesRegex(ValueError,'outside assigned'):
                install_shared(bundle,manifest,sha(bundle),sha(manifest),root/'shared.json')
            self.assertFalse(target.exists())
            install(*args,root/'receipt.json');self.assertEqual(target.read_bytes(),data)
            self.assertEqual(target.stat().st_mode&0o777,0o755)
            with self.assertRaisesRegex(ValueError,'existing input'):
                install(*args,root/'second.json')
            self.assertEqual(target.read_bytes(),data)
            with self.assertRaisesRegex(ValueError,'checksum'):
                install(*args[:5],'wrong',args[6],root/'wrong.json')


if __name__=='__main__':unittest.main()
