from pathlib import Path
import sys,tempfile,unittest,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,sha
from shaping_rescue_dispatch import base_ready,completed_runs,resource_controls,dispatch,worker_slot,reserve_worker,release_worker,bind_worker
from types import SimpleNamespace


class DispatchTests(unittest.TestCase):
    def test_benchmark_idle_admission_counts_non_native_and_pending_work(self):
        from contextlib import nullcontext
        for claims,legacy,native in [({'python-control':{}},set(),0),({}, {'pre-native-fit'},0),({},set(),1),({},set(),0)]:
            with self.subTest(claims=claims,legacy=legacy,native=native):
                observation=dict(can_start=True,native_threads=native,pending_dispatcher_threads=0,
                    available_memory_bytes=64*1024**3,minimum_available_memory_bytes=6*1024**3)
                busy=bool(claims or legacy or native)
                with patch('shaping_rescue_dispatch.worker_claims',return_value=nullcontext(claims)), \
                     patch('shaping_rescue_dispatch.pending_scientific_cells',return_value=legacy), \
                     patch('shaping_rescue_dispatch.worker_slot',return_value=observation):
                    result=reserve_worker(Path('/study'),'benchmark',memory_allowance_bytes=4*1024**3,require_idle=True)
                self.assertEqual(result['can_start'],not busy);self.assertEqual('benchmark' in claims,not busy)

    def test_independent_training_keeps_ordinary_concurrent_admission(self):
        from contextlib import nullcontext
        claims={'existing-fit':{}}
        observation=dict(can_start=True,native_threads=1,pending_dispatcher_threads=0,
            available_memory_bytes=64*1024**3,minimum_available_memory_bytes=6*1024**3)
        with patch('shaping_rescue_dispatch.worker_claims',return_value=nullcontext(claims)), \
             patch('shaping_rescue_dispatch.pending_scientific_cells',return_value=set()), \
             patch('shaping_rescue_dispatch.worker_slot',return_value=observation):
            self.assertTrue(reserve_worker(Path('/study'),'second-fit')['can_start'])
        self.assertIn('second-fit',claims)

    def test_resume_preserves_exposure_and_refuses_failed_or_changed_models(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}')
            cell=dict(id='cell',family='relu');run=root/'runs/cell/attempt-001'
            (run/'fit/checkpoints').mkdir(parents=True)
            checkpoint=run/'fit/checkpoints/ckpt_epoch239.bin';checkpoint.write_bytes(b'model')
            identity=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),build_sha256='build',
                resolution={'setting':1},process_role='rescue_controller',epochs=240,
                pairs_per_epoch=50000,training_exposure_pair_draws=12000000,fit_directory='fit')
            result=dict(identity,stage='trained',exit_code=0,bake_sha256=sha(checkpoint))
            (run/'RUN.json').write_text(json.dumps(identity));(run/'RESULT.json').write_text(json.dumps(result))
            with patch('shaping_rescue_dispatch.resolve',return_value={'setting':1}):
                done,pins=completed_runs(root,[cell],'build');self.assertEqual(done,{'cell'});self.assertEqual(len(pins),3)
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint drift'):completed_runs(root,[cell],'build')
                checkpoint.write_bytes(b'model');result['stage']='failed'
                (run/'RESULT.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'failed rescue'):completed_runs(root,[cell],'build')
                result['stage']='trained';result['epochs']=120
                (run/'RESULT.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'exposure differs'):completed_runs(root,[cell],'build')
                (run/'RESULT.json').unlink()
                with self.assertRaises(FileNotFoundError):completed_runs(root,[cell],'build')

    def test_resource_file_cannot_change_science_or_exceed_existing_limit(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);budget=root/'budget.json';drain=root/'drain'
            args=SimpleNamespace(workers=2,worker_budget_file=budget,drain_file=drain)
            budget.write_text('{"workers":4}')
            self.assertEqual(resource_controls(args),(4,False,sha(budget)))
            drain.touch();self.assertTrue(resource_controls(args)[1])
            for value in [{'workers':5},{'workers':True},{'workers':0},{'workers':2,'epochs':1}]:
                budget.write_text(json.dumps(value))
                with self.assertRaises(ValueError):resource_controls(args)

    def test_drain_finishes_active_children_without_launching_replacements(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study=root/'study';study.mkdir();binary=root/'bin';binary.mkdir()
            (study/'PLAN.json').write_text('{}');(binary/'BUILD.json').write_text('{}')
            cells=[dict(id=f'{i:020x}',family='relu',stage='rescue',partition='inner-0-0',rescue='constant_lr',seed=i) for i in range(240)]
            (study/'RESCUE_TRIGGER_OUTER_0.json').write_text(json.dumps(dict(required_rescue_cells=[c['id'] for c in cells])))
            drain=root/'DRAIN';budget=root/'BUDGET';budget.write_text('{"workers":2}')
            args=SimpleNamespace(study=study,binary_dir=binary,neutral_qualification=root,outer=0,workers=2,resume_completed=False,worker_budget_file=budget,drain_file=drain)
            launched=[]
            class Child:
                def __init__(self,command,**kw):
                    cell=command[command.index('--cell')+1];self.cell=cell;self.pid=1000+len(launched);self.returncode=None;launched.append(cell)
                    p=study/'runs'/cell/'attempt-001';p.mkdir(parents=True)
                def poll(self):
                    if not drain.exists():return None
                    self.returncode=0;(study/'runs'/self.cell/'attempt-001/RESULT.json').write_text('{"stage":"trained","exit_code":0}')
                    return 0
            with patch('shaping_rescue_dispatch.reserve_worker',return_value={'can_start':True}),patch('shaping_rescue_dispatch.bind_worker'),patch('shaping_rescue_dispatch.release_worker'),patch('shaping_rescue_dispatch.NEURAL',['relu']),patch('shaping_rescue_dispatch.load_registration',return_value=({},cells,{})),patch('shaping_rescue_dispatch.base_ready',return_value=True),patch('shaping_rescue_dispatch.select'),patch('shaping_rescue_dispatch.selected_base'),patch('shaping_rescue_dispatch.resolve'),patch('shaping_rescue_dispatch.subprocess.Popen',Child),patch('shaping_rescue_dispatch.time.sleep',side_effect=lambda _:drain.touch()):
                dispatch(args)
            status=json.loads((study/'dispatch/rescue/outer-0/STATUS.json').read_text())
            self.assertEqual(len(launched),2);self.assertEqual(len(status['completed']),2)
            self.assertEqual(status['stage'],'drained');self.assertEqual(len(status['required_cells']),240)
            self.assertFalse(status['active']);self.assertIsNone(status['failure'])


    def test_admission_counts_pending_children_and_respects_headroom(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            proc=Path(d);(proc/'meminfo').write_text('MemAvailable: 8388608 kB\n')
            def native(pid,command):
                p=proc/str(pid);p.mkdir();(p/'comm').write_text('zensim_mlp_trai\n');(p/'cmdline').write_text(command)
            native(1,'trainer --out cell-a');active={'cell-a':None,'cell-b':None}
            a=worker_slot(active,proc,8);self.assertTrue(a['can_start']);self.assertEqual(a['pending_dispatcher_threads'],1)
            native(2,'other');native(3,'other')
            self.assertFalse(worker_slot(active,proc,8)['can_start'])
            (proc/'meminfo').write_text('MemAvailable: 5242880 kB\n')
            self.assertFalse(worker_slot({},proc,32)['can_start'])


    def test_optional_capacity_preserves_default_and_rejects_scientific_fields(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            budget=Path(d)/'budget';budget.write_text('{"workers":8}')
            args=SimpleNamespace(workers=2,worker_budget_file=budget,drain_file=None)
            with self.assertRaises(ValueError):resource_controls(args)
            args.worker_capacity=8;self.assertEqual(resource_controls(args)[0],8)
            budget.write_text('{"workers":8,"epochs":480}')
            with self.assertRaises(ValueError):resource_controls(args)

    def test_shared_claims_bound_concurrent_starts_and_preserve_unknown_owners(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);proc=root/'proc';proc.mkdir();(proc/'meminfo').write_text('MemAvailable: 83886080 kB\n')
            with ThreadPoolExecutor(max_workers=8) as pool:
                results=list(pool.map(lambda i:reserve_worker(root,str(i),proc,8),range(8)))
            self.assertEqual(sum(r['can_start'] for r in results),4)
            path=root/'cpu-admission/CLAIMS.json';claims=json.loads(path.read_text());self.assertEqual(len(claims),4)
            cell=next(iter(claims));bind_worker(root,cell,123);self.assertEqual(json.loads(path.read_text())[cell]['child_pid'],123)
            release_worker(root,cell);self.assertTrue(reserve_worker(root,'replacement',proc,8)['can_start'])
            claims=json.loads(path.read_text());cell=next(iter(claims));claims[cell]['owner']='unknown';path.write_text(json.dumps(claims))
            with self.assertRaises(ValueError):release_worker(root,cell)
            self.assertFalse(reserve_worker(root,'ninth',proc,8)['can_start'])
            self.assertEqual(json.loads(path.read_text())[cell]['owner'],'unknown')


    def test_separate_processes_share_one_pending_start_budget(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);proc=root/'proc';proc.mkdir();(proc/'meminfo').write_text('MemAvailable: 83886080 kB\n')
            code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from shaping_rescue_dispatch import reserve_worker;print(json.dumps(reserve_worker(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[2])/'proc',8)))"
            processes=[subprocess.Popen([sys.executable,'-B','-c',code,str(Path(__file__).resolve().parents[1]/'rev4_featpot'),str(root),str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for i in range(8)]
            results=[]
            for process in processes:
                stdout,stderr=process.communicate(timeout=30);self.assertEqual(process.returncode,0,stderr);results.append(json.loads(stdout))
            self.assertEqual(sum(r['can_start'] for r in results),4)
            claims=json.loads((root/'cpu-admission/CLAIMS.json').read_text());self.assertEqual(len(claims),4)
            self.assertEqual(len({r['owner'] for r in claims.values()}),4)
            self.assertFalse(reserve_worker(root,'after-owners-exit',proc,8)['can_start'])


    def test_legacy_startup_reservations_share_the_cpu_budget(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);proc=root/'proc';proc.mkdir();(proc/'meminfo').write_text('MemAvailable: 83886080 kB\n')
            run=root/'runs'/('a'*20)/'attempt-001';run.mkdir(parents=True);(run/'RUN.json').write_text('{}')
            storage=root/'storage-admission';storage.mkdir();(storage/'RESERVATIONS.json').write_text(json.dumps({str(run):{}}))
            for i in range(3):self.assertTrue(reserve_worker(root,str(i),proc,8)['can_start'])
            denied=reserve_worker(root,'fourth',proc,8);self.assertFalse(denied['can_start']);self.assertEqual(denied['existing_scientific_reservations'],1)
            (run/'RESULT.json').write_text('{}');(storage/'RESERVATIONS.json').write_text('{}')
            self.assertTrue(reserve_worker(root,'fourth',proc,8)['can_start'])


    def test_all_paired_reports_and_diagnostics_are_required(self):
        cells=matrix()
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d)
            selected=[c for c in cells if c['stage']=='inner' and c['family']=='relu' and c['partition'] in ('inner-0-0','inner-0-1')]
            for c in selected:
                p=root/'reexport_max_pairs'/c['id']/'attempt-001';(p/'final').mkdir(parents=True)
                (p/'final/COMPLETE.json').write_text('{}');(p/'CONVERGENCE.json').write_text('{}')
            self.assertTrue(base_ready(root,cells,'relu',0))
            (p/'CONVERGENCE.json').unlink();self.assertFalse(base_ready(root,cells,'relu',0))
            (p/'CONVERGENCE.json').write_text('{}')
            (p/'final/COMPLETE.json').unlink();self.assertFalse(base_ready(root,cells,'relu',0))
            # These are presence checks only. Native selection separately hashes
            # and validates reports; empty fixtures never qualify a selection.


if __name__=='__main__':unittest.main()
