import json,sys,tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_host_assignment as module
from shaping_completion import sha


class PlacementRelocationTests(unittest.TestCase):
    def fixture(self,root):
        (root/'PLAN.json').write_text('{}');directory=root/'worker_assignments';directory.mkdir()
        cells=[dict(id=str(i),stage='inner',family='product_joint_identity',partition='inner-0-0',setting=0,seed=i) for i in range(3)]
        assignment=directory/'original.json'
        record=dict(host='dev',cells=['0','1','2'],plan_sha256=sha(root/'PLAN.json'),build_sha256='build',partition='inner-0-0',scientific_inputs_unchanged=True)
        assignment.write_text(json.dumps(record));(root/'DRAIN').touch()
        queue=root/'STATUS.json';status=dict(pid=99,checked_utc=datetime.now(timezone.utc).isoformat(),draining=True,drain_file=str(root/'DRAIN'),active={'0':123},required_cells=['0','1','2'],plan_sha256=record['plan_sha256'],build_sha256='build')
        queue.write_text(json.dumps(status));proc=root/'proc';(proc/'99').mkdir(parents=True);(proc/'99/cmdline').write_text('python shaping_inner_dispatch.py --study '+str(root))
        return cells,assignment,queue,proc,status

    def test_only_unstarted_cells_move_and_original_placement_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells,assignment,queue,proc,status=self.fixture(root);original=assignment.read_bytes()
            (root/'runs/0/attempt-001').mkdir(parents=True)
            with patch('shaping_execution.load_registration',return_value=({},cells,{})),patch.object(module.socket,'gethostname',return_value='dev'):
                record=module.relocate_joint_subset(root,assignment,['1'],'i265',queue,root/'receipt',proc=proc)
            self.assertEqual(record['stage'],'placement_split');self.assertEqual(record['kept']['cells'],['0','2']);self.assertEqual(record['moved']['cells'],['1'])
            self.assertEqual(Path(record['backup']).read_bytes(),original);self.assertEqual(record['active_children_preserved'],{'0':123})
            module.check(root,[cells[0]],'build','dev');module.check(root,[cells[1]],'build','i265')
            self.assertFalse((root/'runs/1').exists());self.assertFalse(record['training_replayed'])

    def test_started_reserved_and_active_cells_cannot_move(self):
        for reason in ['started','reserved','active']:
            with self.subTest(reason=reason),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);cells,assignment,queue,proc,status=self.fixture(root);original=assignment.read_bytes()
                if reason=='started':(root/'runs/1/attempt-001').mkdir(parents=True)
                if reason=='reserved':
                    p=root/'cpu-admission';p.mkdir();(p/'CLAIMS.json').write_text('{"1":{"owner":"other"}}')
                if reason=='active':status['active']['1']=124;queue.write_text(json.dumps(status))
                with patch('shaping_execution.load_registration',return_value=({},cells,{})),patch.object(module.socket,'gethostname',return_value='dev'):
                    with self.assertRaises(ValueError):module.relocate_joint_subset(root,assignment,['1'],'i265',queue,root/'receipt',proc=proc)
                self.assertEqual(assignment.read_bytes(),original);self.assertFalse((root/'receipt').exists())

    def test_undrained_stale_or_mismatched_queue_refuses_before_mutation(self):
        for change in [dict(draining=False),dict(checked_utc='2026-01-01T00:00:00+00:00'),dict(required_cells=['0','1']),dict(build_sha256='other')]:
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);cells,assignment,queue,proc,status=self.fixture(root);original=assignment.read_bytes();status.update(change);queue.write_text(json.dumps(status))
                with patch('shaping_execution.load_registration',return_value=({},cells,{})),patch.object(module.socket,'gethostname',return_value='dev'):
                    with self.assertRaises(ValueError):module.relocate_joint_subset(root,assignment,['1'],'i265',queue,root/'receipt',proc=proc)
                self.assertEqual(assignment.read_bytes(),original);self.assertFalse((root/'receipt').exists())


if __name__=='__main__':unittest.main()
