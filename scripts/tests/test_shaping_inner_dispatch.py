import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_inner_dispatch as module
from shaping_completion import sha


class InnerDispatchTests(unittest.TestCase):
    def cells(self):
        return [dict(id=str(i),stage='inner',partition='inner-0-0',family='product_joint_identity',setting=0,seed=i) for i in range(3)]

    def test_explicit_registered_subset_never_replays_existing_attempt(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells=self.cells()
            self.assertEqual(len(module.selected_cells(root,cells,['0','2'],'inner-0-0','joint_shapes')),2)
            for requested in (['0','0'],['unknown']):
                with self.assertRaises(ValueError):module.selected_cells(root,cells,requested,'inner-0-0','joint_shapes')
            (root/'runs/0/attempt-001').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'no automatic replay'):
                module.selected_cells(root,cells,['0'],'inner-0-0','joint_shapes')

    def test_completed_fit_requires_native_final_checkpoint_and_full_identity(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}');binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            cell=self.cells()[0];run=root/'runs/0/attempt-001';(run/'checkpoints').mkdir(parents=True)
            final=run/'checkpoints/ckpt_epoch239.bin';final.write_bytes(b'fixture')
            record=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),build_sha256=sha(binary/'BUILD.json'),epochs=240,pairs_per_epoch=50000,stage='trained',exit_code=0,bake_sha256=sha(final))
            for name in ('RUN.json','RESULT.json'):(run/name).write_text(json.dumps(record))
            self.assertEqual(len(module.verify_fit(root,binary,cell)),3)
            final.write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError,'checkpoint drift'):module.verify_fit(root,binary,cell)
            final.write_bytes(b'fixture');record['epochs']=120;(run/'RESULT.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'identity differs'):module.verify_fit(root,binary,cell)

    def exercise_drain(self,invalid_budget=False):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}');binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            budget=root/'budget';budget.write_text('{"workers":2}');drain=root/'drain';launched=[];released=[];ticks=[0]
            class Process:
                def __init__(self,command,**kwargs):self.pid=len(launched)+10;self.returncode=0;launched.append(command)
                def poll(self):return None if ticks[0]<2 else 0
            def tick(_):
                ticks[0]+=1
                if invalid_budget:budget.write_text('{"workers":2,"epochs":480}')
                else:drain.touch()
            args=SimpleNamespace(study=root,binary_dir=binary,output=root/'dispatch',cell=['0','1','2'],partition='inner-0-0',family='joint_shapes',workers=2,worker_capacity=2,worker_budget_file=budget,drain_file=drain,interval=.01)
            with patch.object(module,'load_registration',return_value=({},self.cells(),{})),patch.object(module,'check'),patch.object(module,'reserve_worker',return_value={'can_start':True}),patch.object(module,'bind_worker'),patch.object(module,'release_worker',side_effect=lambda _,cell:released.append(cell)),patch.object(module,'verify_fit',return_value={}),patch.object(module.subprocess,'Popen',Process),patch.object(module.time,'sleep',tick),patch.object(module.os,'sched_getaffinity',return_value=set(range(8))):
                if invalid_budget:
                    with self.assertRaisesRegex(RuntimeError,'non-resource'):module.dispatch(args)
                else:module.dispatch(args)
            status=json.loads((args.output/'STATUS.json').read_text())
            self.assertEqual(len(launched),2);self.assertEqual(len(released),2)
            self.assertFalse(status['active']);self.assertEqual(len(status['completed']),2)
            self.assertEqual(status['stage'],'failed' if invalid_budget else 'drained')

    def test_drain_finishes_children_without_launching_remaining_cell(self):self.exercise_drain()
    def test_invalid_resource_control_drains_live_children_before_failure(self):self.exercise_drain(True)


if __name__=='__main__':unittest.main()
