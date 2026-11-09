from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from shaping_approved_progress import approved_progress
from shaping_completion import sha


class ApprovedProgressTests(unittest.TestCase):
    def test_preserved_nonshortlisted_rescues_do_not_advance_approved_progress(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch)
            scope = root / 'REDUCED_SCOPE.json'
            scope.write_text(json.dumps(dict(fixed_cells=['inner'], required_by_stage=dict(inner=1, rescue=144, outer=48))))
            cells = [dict(id=str(i), stage='rescue' if i < 48 else 'outer', family='relu') for i in range(64)]
            path = root / 'reduced_scope/outer-0/SHORTLIST.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(outer=0, outer_results_read=False,
                reduced_scope_sha256=sha(scope), cells=cells)))
            def row(cid, stage):
                return dict(cell=cid, stage=stage, family='relu', attempts=[dict(fit_success=True)], archive_receipt_verified=False)
            rows = [row(c['id'], c['stage']) for c in cells] + [row('old-rescue', 'rescue'), row('inner', 'inner')]
            result = approved_progress(root, rows)
            self.assertEqual(result['counts']['rescue']['successful_fit_cells'], 48)
            self.assertEqual(result['counts']['rescue']['required'], 144)
            self.assertEqual(result['unresolved_fold_cells'], 128)
            self.assertEqual(len(rows), 66)
            cells[0]['id'] = 'not-registered'
            path.write_text(json.dumps(dict(outer=0, outer_results_read=False,
                reduced_scope_sha256=sha(scope), cells=cells)))
            with self.assertRaisesRegex(ValueError, 'unregistered cell'):
                approved_progress(root, rows)


if __name__ == '__main__':
    unittest.main()
