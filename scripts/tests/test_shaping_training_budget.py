import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded
from shaping_training_budget import register,training_limit,NAME


class TrainingBudgetTests(unittest.TestCase):
    def test_explicit_approval_raises_aggregate_only_and_preserves_headroom(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}')
            self.assertEqual(training_limit(root,6),2)
            approval=root/'approval.json';approval.write_bytes(encoded(dict(approved=True,per_job_thread_caps_unchanged=True,
                ram_and_storage_guards_unchanged=True,limits=dict(six=dict(logical_cpus=6,max_trainers=5),
                    twelve=dict(logical_cpus=12,max_trainers=11),sixteen=dict(logical_cpus=16,max_trainers=14)))))
            register(root,approval)
            for cpus,host,expected in [(6,'six',5),(12,'twelve',11),(16,'sixteen',14)]:
                self.assertEqual(training_limit(root,cpus,host=host),expected)
            with self.assertRaisesRegex(ValueError,'availability differs'):training_limit(root,5,host='six')
            approval.write_text('{}')
            with self.assertRaisesRegex(ValueError,'approval drift'):training_limit(root,6,host='six')

    def test_unapproved_oversubscription_cannot_be_registered(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}');approval=root/'approval.json'
            approval.write_bytes(encoded(dict(approved=True,per_job_thread_caps_unchanged=True,ram_and_storage_guards_unchanged=True,
                limits=dict(six=dict(logical_cpus=6,max_trainers=6)))))
            with self.assertRaisesRegex(ValueError,'outside approved'):register(root,approval)
            self.assertFalse((root/NAME).exists())


if __name__=='__main__':unittest.main()
