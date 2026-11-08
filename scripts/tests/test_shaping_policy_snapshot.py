import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_policy_snapshot as policy


class PolicySnapshotTests(unittest.TestCase):
    def fixture(self,directory):
        root=Path(directory)/'study';root.mkdir();(root/'PLAN.json').write_text('{}')
        repo=Path(directory)/'source';(repo/'docs').mkdir(parents=True)
        for name in policy.FILES:(repo/'docs'/name).write_text('frozen policy bytes\n')
        return root,repo

    def test_snapshot_relocates_only_identical_policy_and_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root,repo=self.fixture(directory)
            with patch.object(policy,'REPO',repo):
                original=policy.freeze(root);self.assertEqual(policy.freeze(root),original)
            # The same public text can come from another immutable source export.
            export=Path(directory)/'export.md';export.write_text('frozen policy bytes\n')
            local=export.with_name(policy.FILES[0]);export.rename(local)
            self.assertEqual(policy.policy_path(root,local),root/'policies'/local.name)
            local.write_text('changed policy bytes\n')
            with self.assertRaisesRegex(ValueError,'snapshot drift'):policy.policy_path(root,local)

    def test_existing_frozen_decisions_prevent_new_path_identity(self):
        for relative in ('reduced_scope/outer-0/SHORTLIST.json','selection/outer-0/OUTER_ADMISSION.json'):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root,repo=self.fixture(directory);path=root/relative;path.parent.mkdir(parents=True);path.write_text('{}')
                with patch.object(policy,'REPO',repo):
                    with self.assertRaisesRegex(ValueError,'before registering'):policy.freeze(root)
                self.assertFalse((root/'policies').exists())

    def test_partial_drift_and_outer_exposure_are_preserved_and_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root,repo=self.fixture(directory)
            with patch.object(policy,'REPO',repo):policy.freeze(root)
            path=root/'policies/POLICIES.json';record=json.loads(path.read_text());record['outer_results_read']=True;path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'identity'):policy.policy_path(root,repo/'docs'/policy.FILES[0])
            record['outer_results_read']=False;path.write_text(json.dumps(record));snapshot=root/'policies'/policy.FILES[0];snapshot.write_text('drift')
            with self.assertRaisesRegex(ValueError,'drift'):policy.policy_path(root,repo/'docs'/policy.FILES[0])
            self.assertEqual(snapshot.read_text(),'drift')
