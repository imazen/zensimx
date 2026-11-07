import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_progress import audit,report_marker
from shaping_completion import encoded,sha


class ProgressTests(unittest.TestCase):
    def test_successful_process_without_report_is_not_evaluated(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}')
            cell=dict(id='cell',stage='inner',family='relu')
            run=root/'runs/cell/attempt-001';run.mkdir(parents=True)
            (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),exit_code=0,stage='trained')))
            status=root/'worker.json';status.write_bytes(encoded(dict(checked_utc='2020-01-01T00:00:00+00:00',live=2,unexpectedly_stopped=[],ssd_free_bytes=100)))
            with patch('shaping_progress.load_registration',return_value=({},[cell],{})):
                result=audit(root,{'worker':status})
            c=result['counts']['inner'];self.assertEqual(c['successful_fit_cells'],1)
            self.assertEqual(c['original_final_report_cells'],0);self.assertEqual(c['neural_diagnostic_receipt_cells'],0)
            self.assertFalse(result['overall_experiments_complete']);self.assertFalse(result['workers']['worker']['fresh'])

    def test_convergence_failure_is_counted_separately_and_pins_verified(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}')
            cell=dict(id='cell',stage='inner',family='softplus')
            run=root/'runs/cell/attempt-001';run.mkdir(parents=True)
            result=run/'RESULT.json'
            result.write_bytes(encoded(dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),exit_code=0,stage='trained')))
            failure=dict(cell=cell,outer_results_read=False,scientifically_complete=False,requires_investigation=True,evidence_sha256={str(result):sha(result)})
            path=run/'CONVERGENCE_FAILURE.json';path.write_bytes(encoded(failure))
            with patch('shaping_progress.load_registration',return_value=({},[cell],{})):
                counts=audit(root,{})['counts']['inner']
                self.assertEqual(counts['cells_with_convergence_failure'],1)
                self.assertEqual(counts['neural_diagnostic_receipt_cells'],0)
                failure['evidence_sha256'][str(result)]='bad';path.write_bytes(encoded(failure))
                with self.assertRaisesRegex(ValueError,'failure evidence drift'):audit(root,{})

    def test_admitted_matched_outer_controls_are_required_without_claiming_fits(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}')
            p=root/'selection/outer-0/OUTER_ADMISSION.json';p.parent.mkdir(parents=True);p.write_text('{}')
            cell=dict(id='extra',stage='fresh_control_outer_matched',family='relu')
            with patch('shaping_progress.load_registration',return_value=({},[],{})), \
                 patch('shaping_outer_admission.resolve',return_value=(dict(additional_cells=[cell]),p)):
                result=audit(root,{})
            c=result['counts']['fresh_control_outer_matched']
            self.assertEqual(c['required'],1);self.assertEqual(c['successful_fit_cells'],0)
            self.assertEqual(result['required_cells'],1);self.assertFalse(result['overall_experiments_complete'])

    def test_modified_report_cannot_count_as_passed(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cell=dict(id='cell');report=root/'REPORT.json';marker=root/'COMPLETE.json'
            report.write_bytes(encoded(dict(cell=cell,plan_sha256='plan')))
            marker.write_bytes(encoded(dict(report_sha256=sha(report))))
            self.assertTrue(report_marker(marker,cell,'plan'))
            report.write_text('{}')
            with self.assertRaisesRegex(ValueError,'report drift'):report_marker(marker,cell,'plan')


if __name__=='__main__':unittest.main()
