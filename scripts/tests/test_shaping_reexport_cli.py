"""Exercise actual CLI startup, fresh status admission and shared-slot release."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_reexport_supervision import main


class ReexportCliTests(unittest.TestCase):
    def test_registered_extension_reaches_scoring_after_its_existing_resolver(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);status=root/'STATUS.json'
            cell=dict(id='extension',family='product_joint_identity',stage='exposure_extension')
            argv=['reexport','--study',str(root),'--binary-dir',str(root/'bin'),'--inspector',str(root/'inspect'),
                  '--family',cell['family'],'--cell',cell['id'],'--status',str(status),'--reserve-slot']
            with patch.object(sys,'argv',argv),patch('shaping_reexport_supervision.load_registration',return_value=({},[],{})), \
                 patch('shaping_extension.resolve',return_value=dict(cell=cell)) as resolve, \
                 patch('shaping_reexport_supervision.sweep',return_value=1) as score, \
                 patch('shaping_rescue_dispatch.reserve_worker',return_value=dict(can_start=True)), \
                 patch('shaping_rescue_dispatch.release_worker'),contextlib.redirect_stdout(io.StringIO()):
                main();resolve.assert_called_once_with(root,cell['id']);score.assert_called_once()
            self.assertEqual(json.loads(status.read_text())['stage'],'sweep_finished')

    def test_wrong_family_unknown_and_duplicate_extensions_never_claim_or_score(self):
        for failure in ('family','unknown','duplicate'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);cell=dict(id='extension',family='separate_joint_identity',stage='exposure_extension')
                argv=['reexport','--study',str(root),'--binary-dir',str(root/'bin'),'--inspector',str(root/'inspect'),
                      '--family','product_joint_identity','--cell',cell['id'],'--reserve-slot']
                if failure=='duplicate':argv+=['--cell',cell['id']]
                with patch.object(sys,'argv',argv),patch('shaping_reexport_supervision.load_registration',return_value=({},[],{})), \
                     patch('shaping_extension.resolve',side_effect=ValueError('unregistered') if failure=='unknown' else None,return_value=dict(cell=cell)), \
                     patch('shaping_rescue_dispatch.reserve_worker') as reserve,patch('shaping_reexport_supervision.sweep') as score:
                    with self.assertRaises(ValueError):main()
                    reserve.assert_not_called();score.assert_not_called()

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
