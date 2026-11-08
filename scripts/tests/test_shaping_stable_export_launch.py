import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import sha
from shaping_stable_export_launch import launch


class TransferOrderingTests(unittest.TestCase):
    def test_failed_transfer_never_submits_any_queue_and_preserves_setup(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);wave=root/'wave';wave.mkdir();source=root/'source'
            owner=source/'scripts/rev4_featpot/shaping_reexport_supervision.py';owner.parent.mkdir(parents=True);owner.write_text('# frozen owner\n')
            record=dict(schema='fixed-qualified-supplemental-wave-v1',host='r3500',source=str(source),source_owner_sha256=sha(owner))
            (wave/'REGISTRATION.json').write_text(json.dumps(record));(wave/'INPUT_PATHS.txt').write_bytes(b'')
            calls=[]
            def run(command,**kwargs):
                calls.append((command,kwargs))
                if any(v.startswith('--files-from=') for v in command):raise subprocess.CalledProcessError(23,command)
            with patch('shaping_reduced_scope.admit_compute'),patch('shaping_stable_export_launch.subprocess.run',side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):launch(wave)
                self.assertTrue((wave/'launch/transfer.log').exists())
                self.assertFalse((wave/'launch/INPUT_VERIFIED.json').exists())
                self.assertFalse(any('systemd-run' in str(c) for c,_ in calls))
                self.assertTrue(any('ALLOW_MISSING=True' in k.get('input','') for _,k in calls))
                with self.assertRaises(FileExistsError):launch(wave)


if __name__=='__main__':unittest.main()
