import json,sys,tempfile,unittest,os,signal,socket,subprocess,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_pretrained_dispatch as module
from shaping_completion import sha


class PretrainedDispatchTests(unittest.TestCase):
    def handoff_fixture(self,root):
        study=root/'study';study.mkdir();binary=root/'bin';binary.mkdir();proc=root/'proc';proc.mkdir()
        (study/'PLAN.json').write_text('{}');(binary/'BUILD.json').write_text('{}')
        cell=dict(id='child',stage='inner',family='product_pretrained_frozen',partition='inner-0-0',seed=0,setting=0)
        run=study/'runs/child/attempt-001';run.mkdir(parents=True)
        identity=dict(pid=300,cell=cell,plan_sha256=sha(study/'PLAN.json'),build_sha256=sha(binary/'BUILD.json'),epochs=120,pairs_per_epoch=50000)
        (run/'RUN.json').write_text(json.dumps(identity));(run/'unchanged.bin').write_bytes(b'active checkpoint')
        def process(pid,argv,children=''):
            p=proc/str(pid);p.mkdir();(p/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
            (p/'stat').write_text(f'{pid} (python) S '+'0 '*18+str(pid*10))
            (p/'task'/str(pid)).mkdir(parents=True);(p/'task'/str(pid)/'children').write_text(children)
        process(100,['python3','shaping_pretrained_dispatch.py','--study',str(study),'--binary-dir',str(binary),'--partition','inner-0-0','--workers','1'],'200')
        process(200,['python3','shaping_execution.py','--study',str(study),'--cell','child','--partition','inner-0-0'])
        process(300,['bash','run-heavy','--out',str(run/'best.bin')])
        status=root/'STATUS.json';status.write_text(json.dumps(dict(pid=100,stage='supervising',failure=None,
            required_cells=['child'],active={'child':200},externally_supervised={},plan_sha256=identity['plan_sha256'],build_sha256=identity['build_sha256'])))
        qualification=root/'QUALIFICATION.json';qualification.write_text(json.dumps(dict(schema='shaping-controller-scope-survival-v1',host=socket.gethostname(),child_completed_after_parent_signal=True,parent_only_signals=[int(v) for v in (signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT)],run_heavy_sha256=sha(Path.home()/'work/claudehints/scripts/run-heavy'))))
        args=SimpleNamespace(study=study,binary_dir=binary,partition='inner-0-0',workers=1,resume_name='new',handoff_status=status,
            handoff_receipt=root/'HANDOFF.json',handoff_qualification=qualification)
        calls=[]
        def send(pid,sig):
            calls.append((pid,sig));self.assertEqual(pid,100)
            if sig==signal.SIGSTOP:
                p=proc/'100/stat';p.write_text(p.read_text().replace(') S ',') T '))
            if sig==signal.SIGTERM:(proc/'100/stat').unlink()
        return args,proc,cell,calls,send

    def test_handoff_signals_only_queue_and_retains_native_attempt(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            args,proc,cell,calls,send=self.handoff_fixture(Path(d))
            with patch.object(module,'load_registration',return_value=({},[cell],{})),patch.object(module.os,'kill',side_effect=send):
                row=module.handoff_controller(args,proc)
            self.assertEqual(calls,[(100,signal.SIGSTOP),(100,signal.SIGTERM),(100,signal.SIGCONT)])
            self.assertEqual(row['stage'],'controller_exited_fits_preserved')
            self.assertFalse(row['training_replayed']);self.assertEqual(row['retained_active_fits']['child']['wrapper_pid'],300)
            self.assertEqual((args.study/'runs/child/attempt-001/unchanged.bin').read_bytes(),b'active checkpoint')

    def test_handoff_refuses_unrecorded_child_and_resumes_queue(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            args,proc,cell,calls,send=self.handoff_fixture(Path(d));(proc/'100/task/100/children').write_text('200 999')
            with patch.object(module,'load_registration',return_value=({},[cell],{})),patch.object(module.os,'kill',side_effect=send):
                with self.assertRaisesRegex(ValueError,'recorded live fit controllers'):module.handoff_controller(args,proc)
            self.assertEqual(calls,[(100,signal.SIGSTOP),(100,signal.SIGCONT)]);self.assertFalse(args.handoff_receipt.exists())

    def test_handoff_accepts_natural_completion_only_with_verified_final(self):
        for valid in [True,False]:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                args,proc,cell,calls,send=self.handoff_fixture(Path(d))
                def finish(pid,sig):
                    send(pid,sig)
                    if sig==signal.SIGTERM:
                        for n in [200,300]:(proc/str(n)/'stat').unlink()
                        run=args.study/'runs/child/attempt-001';(run/'checkpoints').mkdir()
                        final=run/'checkpoints/ckpt_epoch119.bin';final.write_bytes(b'finished')
                        row=json.loads((run/'RUN.json').read_text());row.update(stage='trained',exit_code=0,bake_sha256=sha(final) if valid else 'wrong')
                        (run/'RESULT.json').write_text(json.dumps(row))
                with patch.object(module,'load_registration',return_value=({},[cell],{})),patch.object(module.os,'kill',side_effect=finish):
                    if valid:
                        row=module.handoff_controller(args,proc);self.assertTrue(row['retained_active_fits']['child']['completed_during_handoff'])
                    else:
                        with self.assertRaisesRegex(ValueError,'checkpoint drift'):module.handoff_controller(args,proc)

    def test_handoff_refuses_wrong_qualification_before_any_signal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            args,proc,cell,calls,send=self.handoff_fixture(Path(d));q=json.loads(args.handoff_qualification.read_text());q['run_heavy_sha256']='other';args.handoff_qualification.write_text(json.dumps(q))
            with patch.object(module.os,'kill',side_effect=send):
                with self.assertRaisesRegex(ValueError,'scope-survival qualification'):module.handoff_controller(args,proc)
            self.assertFalse(calls)

    def test_invalid_replacement_arguments_do_not_signal_original_queue(self):
        for name,value in [('workers',5),('resume_name','../other')]:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                args,proc,cell,calls,send=self.handoff_fixture(Path(d));setattr(args,name,value)
                with patch.object(module.os,'kill',side_effect=send):
                    with self.assertRaisesRegex(ValueError,'new valid resume'):module.handoff_controller(args,proc)
                self.assertFalse(calls)

    def test_handoff_refuses_changed_wrapper_and_native_identity(self):
        for change in ['wrapper','exposure']:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                args,proc,cell,calls,send=self.handoff_fixture(Path(d))
                if change=='wrapper':(proc/'300/cmdline').write_bytes(b'other\0')
                else:
                    p=args.study/'runs/child/attempt-001/RUN.json';row=json.loads(p.read_text());row['pairs_per_epoch']=1000;p.write_text(json.dumps(row))
                with patch.object(module,'load_registration',return_value=({},[cell],{})),patch.object(module.os,'kill',side_effect=send):
                    with self.assertRaises(ValueError):module.handoff_controller(args,proc)
                self.assertEqual(calls,[(100,signal.SIGSTOP),(100,signal.SIGCONT)]);self.assertFalse(args.handoff_receipt.exists())

    def test_adopted_fit_requires_full_pair_budget_and_matching_result_identity(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cell=dict(id='child',stage='inner',family='product_pretrained_frozen');run=root/'runs/child/attempt-001';(run/'checkpoints').mkdir(parents=True)
            final=run/'checkpoints/ckpt_epoch119.bin';final.write_bytes(b'fixture')
            record=dict(cell=cell,plan_sha256='plan',build_sha256='build',epochs=120,pairs_per_epoch=50000,stage='trained',exit_code=0,bake_sha256=sha(final))
            for name in ['RUN.json','RESULT.json']:(run/name).write_text(json.dumps(record))
            self.assertEqual(module.existing_trial(root,cell,'plan','build'),('trained',None))
            for name,change in [('RUN.json',dict(pairs_per_epoch=1000)),('RESULT.json',dict(pairs_per_epoch=1000)),('RESULT.json',dict(cell={**cell,'id':'other'}))]:
                for path in ['RUN.json','RESULT.json']:(run/path).write_text(json.dumps(record))
                (run/name).write_text(json.dumps({**record,**change}))
                with self.assertRaisesRegex(ValueError,'identity differs'):module.existing_trial(root,cell,'plan','build')

    def test_cli_ownership_does_not_precreate_initial_audit_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d)
            def dispatch(args):
                self.assertFalse((root/'dispatch/pretrained/inner-0-0').exists())
                self.assertTrue((root/'dispatch/pretrained/locks/inner-0-0.lock').exists())
            argv=['dispatch','--study',str(root),'--binary-dir',str(root/'bin'),'--partition','inner-0-0']
            with patch.object(sys,'argv',argv),patch.object(module,'dispatch',side_effect=dispatch) as owner:
                module.main()
            owner.assert_called_once()

    def fixture(self,root):
        (root/'PLAN.json').write_text('{}')
        parent=dict(id='parent',stage='inner',family='product_joint_identity',partition='inner-0-0',setting=0,seed=0)
        child={**parent,'id':'child','family':'product_pretrained_frozen'}
        run=root/'runs/parent/attempt-001';run.mkdir(parents=True)
        record=dict(cell=parent,plan_sha256=sha(root/'PLAN.json'),build_sha256='build',epochs=240,pairs_per_epoch=50000)
        (run/'RUN.json').write_text(json.dumps(record))
        result={**record,'stage':'interrupted','exit_code':None,'restart_approval_pending':True,'preserved_archive_receipt_sha256':'a'*64}
        (run/'RESULT.json').write_text(json.dumps(result))
        return parent,child,run,result

    def test_preserved_interruption_waits_without_using_partial_prefix(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);parent,child,run,result=self.fixture(root)
            with patch.object(module,'completed_parent') as prefix:
                ready,dependency=module.parent_state(root,child,[parent,child],'build')
            self.assertFalse(ready);prefix.assert_not_called()
            self.assertEqual(dependency['parent_cell'],'parent');self.assertEqual(len(dependency['evidence_sha256']),2)

    def test_failed_and_drifted_parents_remain_refusals(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);parent,child,run,result=self.fixture(root)
            for change in [dict(stage='failed',exit_code=1),dict(restart_approval_pending=False),dict(preserved_archive_receipt_sha256=''),dict(epochs=120),dict(build_sha256='other')]:
                (run/'RESULT.json').write_text(json.dumps({**result,**change}))
                with self.assertRaises(ValueError):module.parent_state(root,child,[parent,child],'build')

    def test_completed_parent_still_uses_existing_owner_gate(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);parent,child,run,result=self.fixture(root)
            (run/'RESULT.json').write_text(json.dumps({**result,'stage':'trained','exit_code':0}))
            with patch.object(module,'completed_parent',side_effect=ValueError('ambiguous')) as gate:
                with self.assertRaisesRegex(ValueError,'ambiguous'):module.parent_state(root,child,[parent,child],'build')
            gate.assert_called_once()

    def exercise_independent_wait(self,foreign=False,placement_race=False):
        class Observed(Exception):pass
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}');binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            cells=[dict(id=str(i),stage='inner',family='product_pretrained_frozen',partition='inner-0-0',setting=0,seed=i) for i in range(2)]
            launched=[];released=[];ticks=[0]
            class Process:
                pid=123;returncode=0
                def __init__(self,command,**kw):launched.append(command)
                def poll(self):return 0
            def sleep(_):
                ticks[0]+=1
                if ticks[0]==2:raise Observed()
            def readiness(_,cell,*rest):return (False,{'parent_cell':'blocked'}) if cell['id']=='0' and not foreign else (True,None)
            placement_calls={}
            def placement(_,cell,*rest):
                placement_calls[cell['id']]=placement_calls.get(cell['id'],0)+1
                if foreign and cell['id']=='0' and not (placement_race and placement_calls['0']==1):
                    return dict(host='remote-fixture',assignment='fixture',assignment_sha256='pin')
                return None
            args=SimpleNamespace(study=root,binary_dir=binary,partition='inner-0-0',workers=2,resume_name='test')
            with patch.object(module,'load_registration',return_value=({},cells,{})),patch.object(module,'existing_trial',return_value=None),patch.object(module,'assigned_placement',side_effect=placement),patch.object(module,'parent_state',side_effect=readiness),patch.object(module,'reserve_worker',return_value={'can_start':True}),patch.object(module,'bind_worker'),patch.object(module,'release_worker',side_effect=lambda _,c:released.append(c)),patch.object(module,'verify_inner_fit'),patch.object(module.subprocess,'Popen',Process),patch.object(module.time,'sleep',side_effect=sleep):
                with self.assertRaises(Observed):module.dispatch(args)
            status=json.loads((root/'dispatch/pretrained/inner-0-0/test/STATUS.json').read_text())
            self.assertEqual(len(launched),1);self.assertEqual(launched[0][-3:],['1','--workers','1'])
            self.assertEqual(launched[0][0],sys.executable)
            self.assertEqual(status['completed'],['1']);self.assertEqual(released,['0','1'] if placement_race else ['1']);self.assertFalse(status['active'])
            self.assertEqual(len(status['required_cells']),2)
            self.assertEqual(set(status['interrupted_parent_dependencies']),set() if foreign else {'0'})
            self.assertEqual(set(status['externally_assigned_required']),{'0'} if foreign else set())
            self.assertEqual(status['stage'],'supervising');self.assertIsNone(status['failure'])

    def test_waiting_parent_does_not_block_independent_registered_child(self):self.exercise_independent_wait()
    def test_foreign_placement_remains_required_and_never_launches_locally(self):self.exercise_independent_wait(True)
    def test_placement_changed_during_admission_releases_claim_without_training(self):self.exercise_independent_wait(True,True)


class HandoffScopeQualification(unittest.TestCase):
    # Explicit caller-only qualification: ordinary unit discovery does not run it.
    def verify_scope_child_survival(self):
        output=Path(os.environ['SHAPING_HANDOFF_QUALIFICATION'])
        self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);identity=root/'PIDS.json';marker=root/'CHILD.json';log=root/'scope.log'
            child="import sys,time,json;from pathlib import Path;p=Path(sys.argv[1]);p.write_text('ready');time.sleep(3);p.write_text(json.dumps(dict(completed_ns=time.monotonic_ns())))"
            parent="import subprocess,sys,os,time,json;from pathlib import Path;c=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[3]]);Path(sys.argv[2]).write_text(json.dumps(dict(parent=os.getpid(),child=c.pid)));time.sleep(30)"
            wrapper=Path.home()/'work/claudehints/scripts/run-heavy'
            with log.open('w') as stream:
                worker=subprocess.Popen([str(wrapper),'--mem','1G','--jobs','1','--',sys.executable,'-c',parent,child,str(identity),str(marker)],stdout=stream,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+10
                while not identity.exists() or not marker.exists():
                    self.assertIsNone(worker.poll());self.assertLess(time.monotonic(),deadline);time.sleep(.02)
                ids=json.loads(identity.read_text());signal_ns=time.monotonic_ns()
                os.kill(ids['parent'],signal.SIGSTOP)
                os.kill(ids['parent'],signal.SIGTERM)
                try:os.kill(ids['parent'],signal.SIGCONT)
                except ProcessLookupError:pass
                worker.wait(timeout=10)
                while marker.read_text()=='ready':
                    self.assertLess(time.monotonic(),deadline);time.sleep(.02)
            completion=json.loads(marker.read_text());self.assertGreater(completion['completed_ns'],signal_ns)
            self.assertNotEqual(worker.returncode,0)
            report=dict(schema='shaping-controller-scope-survival-v1',host=socket.gethostname(),
                checked_utc=module.datetime.now(module.timezone.utc).isoformat(),run_heavy_sha256=sha(wrapper),
                parent_only_signals=[int(v) for v in [signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT]],parent_signal_ns=signal_ns,child_completed_ns=completion['completed_ns'],
                child_completed_after_parent_signal=True,wrapper_exit_code=worker.returncode,wrapper_log=log.read_text())
            output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':unittest.main()
