"""Exercise actual CLI startup, fresh status admission and shared-slot release."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_reexport_supervision import main


class ReexportCliTests(unittest.TestCase):
    def test_status_startup_and_slot_release_on_success_or_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);status=root/'status/STATUS.json';cell=dict(id='cell',family='relu',stage='inner')
                argv=['reexport','--study',str(root),'--binary-dir',str(root/'bin'),'--inspector',str(root/'inspect'),
                      '--family','relu','--cell','cell','--status',str(status),'--reserve-slot']
                with patch.object(sys,'argv',argv),patch('shaping_reexport_supervision.load_registration',return_value=({},[cell],{})), \
                     patch('shaping_reexport_supervision.sweep',side_effect=RuntimeError('preserved scoring failure') if fail else None,return_value=1), \
                     patch('shaping_rescue_dispatch.reserve_worker',return_value=dict(can_start=True)) as reserve, \
                     patch('shaping_rescue_dispatch.release_worker') as release,contextlib.redirect_stdout(io.StringIO()):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'preserved scoring failure'):main()
                    else:main()
                reserve.assert_called_once();release.assert_called_once()
                value=json.loads(status.read_text());self.assertEqual(value['stage'],'starting' if fail else 'sweep_finished')
                if not fail:self.assertEqual(value['completed_supplemental_cells'],1)

    def test_existing_status_refuses_before_claim_or_scoring(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);status=root/'STATUS.json';status.write_text('preserved prior wave')
            argv=['reexport','--study',str(root),'--binary-dir',str(root/'bin'),'--inspector',str(root/'inspect'),
                  '--family','relu','--status',str(status),'--reserve-slot']
            with patch.object(sys,'argv',argv),patch('shaping_reexport_supervision.sweep') as score, \
                 patch('shaping_rescue_dispatch.reserve_worker') as reserve:
                with self.assertRaises(FileExistsError):main()
            score.assert_not_called();reserve.assert_not_called();self.assertEqual(status.read_text(),'preserved prior wave')


if __name__=='__main__':unittest.main()
