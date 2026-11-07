import json,sys,tempfile,unittest,signal
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_host_assignment as module
from shaping_completion import sha


class PlacementRelocationTests(unittest.TestCase):
    def test_assignment_lookup_preserves_host_and_scientific_checks(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells,assignment,_,_,_=self.fixture(root)
            row=module.assigned_placement(root,cells[0],'build');self.assertEqual(row['host'],'dev');self.assertEqual(row['assignment_sha256'],sha(assignment))
            record=json.loads(assignment.read_text());record['build_sha256']='other';assignment.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'scientific identity'):module.assigned_placement(root,cells[0],'build')
            (assignment.parent/'duplicate.json').write_text(assignment.read_text())
            with self.assertRaisesRegex(ValueError,'ambiguous'):module.assigned_placement(root,cells[0],'build')
    def test_incoming_placement_checksums_are_verified_before_coordinator_mutation(self):
        for corrupt in [False,True]:
            with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                base=Path(d);source=base/'source';target=base/'target';source.mkdir();target.mkdir()
                cells,assignment,queue,proc,status=self.fixture(source)
                with patch('shaping_execution.load_registration',return_value=({},cells,{})),patch.object(module.socket,'gethostname',return_value='dev'):
                    record=module.relocate_joint_subset(source,assignment,['1'],'i265',queue,source/'receipt',proc=proc)
                _,target_assignment,_,_,_=self.fixture(target);original=target_assignment.read_bytes();files=base/'files';files.mkdir()
                for key in ['backup','kept_assignment','moved_assignment']:
                    src=Path(record[key]);(files/src.name).write_bytes(src.read_bytes());record[key]=str(target_assignment.parent/src.name)
                record['original_assignment']=str(target_assignment)
                if corrupt:(files/Path(record['moved_assignment']).name).write_text('{}')
                with patch('shaping_execution.load_registration',return_value=({},cells,{})):
                    if corrupt:
                        with self.assertRaisesRegex(ValueError,'checksum differs'):module.mirror_placement(target,target_assignment,record,files)
                        self.assertEqual(target_assignment.read_bytes(),original)
                    else:
                        module.mirror_placement(target,target_assignment,record,files)
                        module.check(target,[cells[1]],'build','i265');module.check(target,[cells[0]],'build','dev')

    def test_collection_resumes_before_remote_action_but_stays_quiescent_after_uncertainty(self):
        import subprocess
        for uncertain in [False,True]:
            with self.subTest(uncertain=uncertain),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);cells,assignment,proc,driver,guard,legacy,kill,calls=self.legacy_fixture(root)
                (proc/'99/cmdline').write_text('python shaping_worker_watch.py --host dev')
                children=proc/'99/task/99';children.mkdir(parents=True);(children/'children').write_text('')
                collector=root/'COLLECTOR.json';collector.write_text(json.dumps(dict(pid=99,stage='watching',checked_utc=datetime.now(timezone.utc).isoformat())))
                effects=[subprocess.CompletedProcess([],0),subprocess.TimeoutExpired('ssh',60)] if uncertain else [OSError('connect refused')]
                with patch.object(module.os,'kill',side_effect=kill),patch.object(module.subprocess,'run',side_effect=effects):
                    with self.assertRaises((OSError,subprocess.TimeoutExpired)):
                        module.coordinate_remote_legacy(root,assignment,['1'],'i265','dev',root/'tool',legacy,driver,guard,sha(driver),sha(guard),collector,root/'control',proc=proc)
                self.assertEqual(calls,[signal.SIGSTOP] if uncertain else [signal.SIGSTOP,signal.SIGCONT])
                failure=json.loads((root/'control/FAILURE.json').read_text());self.assertEqual(failure['collector_left_stopped'],uncertain)

    def legacy_fixture(self,root):
        cells,assignment,queue,proc,status=self.fixture(root)
        driver=root/'shaping_execution.py';driver.write_text((Path(__file__).resolve().parents[1]/'rev4_featpot/shaping_execution.py').read_text())
        guard=root/'shaping_host_assignment.py';guard.write_text(Path(module.__file__).read_text())
        (proc/'99/cmdline').write_text('python shaping_execution.py --study '+str(root)+' --workers 1')
        def state(value):(proc/'99/stat').write_text('99 (python3) '+value+' '+'0 '*18+'12345\n')
        state('S');calls=[]
        def kill(pid,sig):
            self.assertEqual(pid,99);calls.append(sig);state('T' if sig==signal.SIGSTOP else 'S')
        run=root/'runs/0/attempt-001';run.mkdir(parents=True)
        record=dict(cell=cells[0],pid=101,driver_sha256=sha(driver),plan_sha256=sha(root/'PLAN.json'),build_sha256='build',epochs=240,pairs_per_epoch=50000)
        (run/'RUN.json').write_text(json.dumps(record));(proc/'101').mkdir();(proc/'101/cmdline').write_text('run-heavy '+str(run))
        (proc/'101/stat').write_text('101 (bash) S 99 '+'0 '*17+'333\n')
        legacy=root/'LEGACY_STATUS.json';legacy.write_text(json.dumps(dict(controller_pid=99,cells=['0','1','2'],stage='running')))
        return cells,assignment,proc,driver,guard,legacy,kill,calls

    def test_legacy_handoff_stops_only_writer_and_always_resumes_it(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells,assignment,proc,driver,guard,legacy,kill,calls=self.legacy_fixture(root)
            with patch('shaping_execution.load_registration',return_value=({},cells,{})),patch.object(module.socket,'gethostname',return_value='dev'),patch.object(module.os,'kill',side_effect=kill):
                record=module.relocate_legacy_joint(root,assignment,['1'],'i265',legacy,driver,guard,sha(driver),sha(guard),root/'receipt',root/'control',proc=proc)
            self.assertEqual(calls,[signal.SIGSTOP,signal.SIGCONT]);self.assertEqual(record['kept']['cells'],['0','2']);self.assertEqual(record['active_children_preserved'],{'0':101})
            self.assertTrue((root/'control/RESUMED.json').exists());self.assertFalse((root/'runs/1').exists())

    def test_legacy_failure_resumes_writer_without_moving_started_cells(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells,assignment,proc,driver,guard,legacy,kill,calls=self.legacy_fixture(root);original=assignment.read_bytes()
            (root/'runs/1/attempt-001').mkdir(parents=True)
            with patch('shaping_execution.load_registration',return_value=({},cells,{})),patch.object(module.socket,'gethostname',return_value='dev'),patch.object(module.os,'kill',side_effect=kill):
                with self.assertRaisesRegex(ValueError,'started or reserved'):
                    module.relocate_legacy_joint(root,assignment,['1'],'i265',legacy,driver,guard,sha(driver),sha(guard),root/'receipt',root/'control',proc=proc)
            self.assertEqual(calls,[signal.SIGSTOP,signal.SIGCONT]);self.assertEqual(assignment.read_bytes(),original);self.assertFalse((root/'receipt').exists())

    def test_legacy_guard_must_precede_attempt_creation_and_match_host_behavior(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells,assignment,proc,driver,guard,legacy,kill,calls=self.legacy_fixture(root)
            driver.write_text(driver.read_text().replace("    check(root,[cell],sha(binary_dir/'BUILD.json'))","    pass",1))
            with self.assertRaisesRegex(ValueError,'guard must precede'):module.verify_legacy_guard(driver,guard,sha(driver),sha(guard))
            self.assertFalse(calls)

    def test_legacy_handoff_requires_every_executor_slot_owned_and_live(self):
        for fault in ['unrecorded_slot','foreign_child']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);cells,assignment,proc,driver,guard,legacy,kill,calls=self.legacy_fixture(root);original=assignment.read_bytes()
                if fault=='unrecorded_slot':(proc/'99/cmdline').write_text('python shaping_execution.py --study '+str(root)+' --workers 2')
                else:(proc/'101/stat').write_text('101 (bash) S 100 '+'0 '*17+'333\n')
                with patch('shaping_execution.load_registration',return_value=({},cells,{})),patch.object(module.socket,'gethostname',return_value='dev'),patch.object(module.os,'kill',side_effect=kill):
                    with self.assertRaises(ValueError):module.relocate_legacy_joint(root,assignment,['1'],'i265',legacy,driver,guard,sha(driver),sha(guard),root/'receipt',root/'control',proc=proc)
                self.assertEqual(calls,[signal.SIGSTOP,signal.SIGCONT]);self.assertEqual(assignment.read_bytes(),original)

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
