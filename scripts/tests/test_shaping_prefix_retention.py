import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_prefix_retention as retention
from shaping_completion import encoded,sha


class PrefixRetentionTests(unittest.TestCase):
    def fixture(self,directory):
        study=Path(directory);cell=next(iter(retention.CELLS));name='ckpt_epoch000.bin'
        root=study/'runs'/cell/'attempt-002';copies=root/'prefix_comparison_v2';copies.mkdir(parents=True)
        for label in ('original','restart'):(copies/(label+'-'+name)).write_bytes(b'preserved original bytes')
        pin=sha(copies/('original-'+name));original=dict(sha256=pin,bytes=24,archive_path='/durable/'+name)
        proof=study/'proof.json';proof.write_bytes(encoded(dict(schema='shaping-prefix-archive-proof-v1',
            archive_receipt_sha256=retention.RECEIPT,cells={cell:dict(authorization_sha256='approved',files={name:original})})))
        (root/'PREFIX_COMPARISON_V2.json').write_bytes(encoded(dict(expected=1,compared={name:
            dict(equal=True,original_sha256=pin,restart_sha256=pin)})))
        return study,cell,root,proof,dict(checkpoint_sha256={name:pin})

    def test_retirement_keeps_proof_and_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study,cell,root,proof,row=self.fixture(directory)
            with patch.object(retention,'authorize',return_value=(row,'approved')),patch.object(retention,'no_open_users'):
                result=retention.retire_locked(study,cell,proof)
                self.assertEqual(result['retired_files'],2)
                self.assertEqual(result['retired_bytes'],48)
                self.assertEqual(retention.retire_locked(study,cell,proof),result)
            self.assertEqual(list((root/'prefix_comparison_v2').iterdir()),[])
            self.assertEqual(sha(proof),sha(root/'PREFIX_ARCHIVE_PROOF.json'))

    def test_drift_and_open_user_preserve_both_duplicates(self):
        for failure in ('proof','canonical','open'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                study,cell,root,proof,row=self.fixture(directory)
                files=list((root/'prefix_comparison_v2').iterdir())
                if failure=='proof':
                    data=json.loads(proof.read_text());data['archive_receipt_sha256']='wrong';proof.write_bytes(encoded(data))
                if failure=='canonical':files[0].write_bytes(b'drift')
                with patch.object(retention,'authorize',return_value=(row,'approved')),patch.object(retention,'no_open_users',
                    side_effect=ValueError('open') if failure=='open' else None):
                    with self.assertRaises(ValueError):retention.retire_locked(study,cell,proof)
                self.assertTrue(all(p.exists() for p in files))

    def test_unrecorded_missing_duplicate_refuses(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study,cell,root,proof,row=self.fixture(directory)
            next((root/'prefix_comparison_v2').iterdir()).unlink()
            with patch.object(retention,'authorize',return_value=(row,'approved')):
                with self.assertRaisesRegex(ValueError,'unrecorded missing'):retention.retire_locked(study,cell,proof)


if __name__=='__main__':unittest.main()
