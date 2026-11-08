import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_progress import corrected_program


class CorrectedDashboardTests(unittest.TestCase):
    def test_only_bound_approved_registration_changes_the_displayed_budget(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);self.assertIsNone(corrected_program(root))
            approval=root/'THRESHOLD_COORDINATE_APPROVAL_20261108.json';approval.write_bytes(encoded(dict(answer='Approve four settings: 1,461 fits')))
            path=root/'corrected_thresholds/REGISTRATION.json';path.parent.mkdir()
            record=dict(schema='corrected-shaped-threshold-registration-v1',core_fit_target=1461,corrected_condition_count=288,
                omitted_original_conditions=[{}]*108,outer_results_read=False,settings=[0,1,2,3],paired_seeds=[0,1,2],
                execution_activated=False,evidence_sha256={str(approval):sha(approval)})
            path.write_bytes(encoded(record));value=corrected_program(root)
            self.assertEqual(value['core_fit_target'],1461);self.assertFalse(value['execution_activated'])
            approval.write_text('changed')
            with self.assertRaisesRegex(ValueError,'approval differs'):corrected_program(root)


if __name__=='__main__':unittest.main()
