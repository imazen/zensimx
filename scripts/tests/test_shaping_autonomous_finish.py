import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_autonomous_finish import job_complete, snapshot_closure_ready


class CompletionEvidenceTests(unittest.TestCase):
    def test_failed_snapshot_probe_cannot_reuse_earlier_zero_pending_count(self):
        collection = dict(checked_utc='2026-11-13T08:10:00+00:00')
        snapshot = dict(stage='failed_with_preserved_evidence',
                        last_cycle_pending_cells=0, checked_utc='2026-11-13T08:24:00+00:00')
        self.assertFalse(snapshot_closure_ready(snapshot, collection))
        snapshot['stage'] = 'watching_completed_collections'
        self.assertTrue(snapshot_closure_ready(snapshot, collection))
        snapshot['checked_utc'] = '2026-11-13T08:00:00+00:00'
        self.assertFalse(snapshot_closure_ready(snapshot, collection))

    def test_bootstrap_closure_requires_all_completed_and_archived_comparisons(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path = Path(directory)/'STATUS.json'
            job = dict(complete_marker=str(path), completion=dict(stage='complete', required=25),
                       completion_lengths=dict(completed=25, archived=25))
            record = dict(stage='complete', required=25, completed=list(range(25)), archived=list(range(24)))
            path.write_text(json.dumps(record))
            self.assertFalse(job_complete(job))
            record['archived'].append(24)
            path.write_text(json.dumps(record))
            self.assertTrue(job_complete(job))
