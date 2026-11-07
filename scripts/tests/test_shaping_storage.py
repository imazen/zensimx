from pathlib import Path
import sys,tempfile,unittest,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_storage import pending_runs,reservation,required_free,GIB

class StorageTests(unittest.TestCase):
    def test_unresolved_attempts_reserve_even_without_live_pid(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            r=Path(d);a=r/'runs/a/attempt-001';a.mkdir(parents=True);(a/'RUN.json').write_text('{}')
            self.assertEqual(pending_runs(r),{str(a.resolve())})
            (a/'RESULT.json').write_text('{}');self.assertEqual(pending_runs(r),set())
    def test_low_space_waits_without_starting_a_reservation(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            r=Path(d);a=r/'runs/a/attempt-001'
            with patch('shaping_storage.shutil.disk_usage') as disk:
                disk.return_value.free=33*GIB
                def reclaimed(seconds):
                    self.assertEqual(seconds,30)
                    self.assertFalse((r/'storage-admission/RESERVATIONS.json').exists())
                    disk.return_value.free=34*GIB
                with patch('shaping_storage.time.sleep',side_effect=reclaimed) as sleep:
                    with reservation(r,a,30*GIB):self.assertEqual(sleep.call_count,1)

    def test_nested_reservations_count_union_and_release_after_exception(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            r=Path(d);a=r/'runs/a/attempt-001';b=r/'runs/b/attempt-001'
            with patch('shaping_storage.shutil.disk_usage') as disk:
                disk.return_value.free=100*GIB
                with reservation(r,a,30*GIB):
                    a.mkdir(parents=True);(a/'RUN.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError,'test'):
                        with reservation(r,b,30*GIB):
                            record=json.loads((r/'storage-admission/b-attempt-001.json').read_text())
                            self.assertEqual(record['pending_runs'],2)
                            self.assertEqual(record['required_free_bytes'],35*GIB)
                            raise ValueError('test')
                    self.assertEqual(len(json.loads((r/'storage-admission/RESERVATIONS.json').read_text())),1)
                self.assertEqual(json.loads((r/'storage-admission/RESERVATIONS.json').read_text()),{})

if __name__=='__main__':unittest.main()
