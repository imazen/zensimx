from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import shaping_bootstrap_pipeline as owner


class BootstrapPipelineTests(unittest.TestCase):
    def test_complete_queue_spreads_without_changing_comparisons(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch)
            rows = [dict(file=f'comparison-{i:03}.json', spec=dict(role=str(i))) for i in range(24)]
            (root / 'MATCHED_REGISTRATION.json').write_text(json.dumps(dict(comparisons=rows)))
            def load(study, path):
                return next(row['spec'] for row in rows if row['file'] == path.name), {}
            with patch('shaping_reduced_scope.coverage', return_value={'approved': True}), \
                 patch('shaping_bootstrap_plan.load_comparison', side_effect=load):
                queue = owner.queue(root / 'study', root)
                self.assertEqual(len(queue), 25)
                self.assertEqual([row['spec'] for row in queue[1:]], [row['spec'] for row in rows])
                self.assertEqual({row['host'] for row in queue}, set(owner.HOST_SLOTS))
                rows.pop()
                (root / 'MATCHED_REGISTRATION.json').write_text(json.dumps(dict(comparisons=rows)))
                with self.assertRaisesRegex(ValueError, 'all 24'):
                    owner.queue(root / 'study', root)

    def test_no_outer_report_access_before_all_admissions(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch, \
             patch('shaping_outer_admission.resolve') as admission, \
             patch('shaping_bootstrap.ready_cells') as reports:
            self.assertFalse(owner.ready(Path(scratch), dict(spec=None)))
            admission.assert_not_called()
            reports.assert_not_called()

    def test_completed_output_without_transport_receipt_cannot_be_archived(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch); request = root / 'REQUEST.json'
            request.write_text(json.dumps(dict(result_output=str(root / 'result'))))
            with patch('shaping_bootstrap.complete_output', return_value=True), \
                 patch('shaping_bootstrap.archive_output') as archive:
                with self.assertRaisesRegex(ValueError, 'worker collection evidence'):
                    owner.collect(SimpleNamespace(), {}, request)
                archive.assert_not_called()


if __name__ == '__main__':
    unittest.main()
