import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_control_worker as control
import shaping_rescue_worker as worker
from shaping_completion import encoded,sha,matrix
from shaping_prediction_evidence import save


class ControlWorkerTests(unittest.TestCase):
    def test_combined_control_memory_allowance_is_admitted_before_launch(self):
        from shaping_rescue_dispatch import reserve_worker,release_worker
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            root=Path(temporary)
            def observed(*args,**kwargs):return dict(can_start=True,minimum_available_memory_bytes=6*1024**3,available_memory_bytes=7*1024**3)
            with patch('shaping_rescue_dispatch.pending_scientific_cells',return_value=set()),patch('shaping_rescue_dispatch.worker_slot',side_effect=observed):
                denied=reserve_worker(root,'control',memory_allowance_bytes=4*1024**3)
                self.assertFalse(denied['can_start']);self.assertEqual(denied['minimum_available_memory_bytes'],8*1024**3)
                self.assertEqual(json.loads((root/'cpu-admission/CLAIMS.json').read_text()),{})
                self.assertTrue(reserve_worker(root,'ordinary')['can_start']);release_worker(root,'ordinary')

    def test_native_startup_keeps_its_memory_allowance_until_epoch_zero(self):
        from shaping_rescue_dispatch import reserve_worker,release_worker
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            root=Path(temporary)
            def observed(*args,**kwargs):return dict(can_start=True,minimum_available_memory_bytes=6*1024**3,available_memory_bytes=9*1024**3)
            with patch('shaping_rescue_dispatch.pending_scientific_cells',return_value=set()),patch('shaping_rescue_dispatch.worker_slot',side_effect=observed):
                self.assertTrue(reserve_worker(root,'first',warmup_until_checkpoint=True)['can_start'])
                self.assertFalse(reserve_worker(root,'second',warmup_until_checkpoint=True)['can_start'])
                checkpoint=root/'runs/first/attempt-001/checkpoints/ckpt_epoch000.bin';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'epoch-zero')
                self.assertTrue(reserve_worker(root,'second',warmup_until_checkpoint=True)['can_start'])
                release_worker(root,'first');release_worker(root,'second')

    def test_control_capsule_binds_original_admission_and_package_versions(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            root=Path(temporary);study=root/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
            binary=root/'binary';binary.mkdir();(binary/'BUILD.json').write_bytes(encoded(dict(binary_sha256={})))
            marker=root/'input';marker.write_bytes(b'original');cell=next(c for c in matrix() if c['stage']=='outer' and c['family']=='extra_trees')
            plan=dict(input_pins={str(marker):sha(marker)});parts={cell['partition']:dict(train=[1,3],test=[9])}
            with patch('shaping_execution.load_registration',return_value=(plan,[cell],parts)),patch.object(control.owner,'admit',return_value=(plan,parts,'extra_trees',[cell],dict(admission_sha256='outer',selected_setting=2))),patch.object(worker,'DATA_ROOTS',(root,)):
                record=control.register(study,cell['id'],binary,'r3500',root/'request')
                self.assertEqual(record['execution_kind'],'control');self.assertFalse(record['outer_results_read'])
                self.assertIn(str(control.REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json'),record['files'])
                self.assertIn(str(marker),record['files']);self.assertEqual(record['resolution']['expected_indices'],[1,3,9])
                self.assertEqual(record['resolution']['package_versions'],control.packages())
                with patch.object(worker.socket,'gethostname',return_value='r3500'):
                    self.assertEqual(worker.verify(record),[])
                    resolution,command=control.reproduce(dict(record,_request_path=str(root/'request/REQUEST.json')))
                    self.assertEqual(resolution,record['resolution']);self.assertEqual(command[command.index('--mem')+1],'2G')
                    marker.write_bytes(b'drift')
                    with self.assertRaisesRegex(ValueError,'drift'):worker.verify(record)

    def test_noncontrol_and_existing_attempt_refuse(self):
        for family in ('relu','extra_trees'):
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
                root=Path(temporary);cell=next(c for c in matrix() if c['stage']=='outer' and c['family']==family)
                if family=='extra_trees':(root/'runs'/cell['id']).mkdir(parents=True)
                with patch('shaping_execution.load_registration',return_value=({},[cell],{})):
                    with self.assertRaises(ValueError):control.context(root,cell['id'],root/'bin')

    def test_native_artifact_and_replay_input_corruption_refuse(self):
        for family in ('extra_trees','shaped_additive'):
            with self.subTest(family=family),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
                root=Path(temporary);evaluation=root/'evaluation';evaluation.mkdir()
                cell=dict(id='cell',family=family);outer=dict(admission_sha256='outer',selected_setting=2)
                scope=dict(cell=cell,plan_sha256='plan',build_sha256='build',outer=outer,expected_indices=[1,3,9],package_versions={})
                record=dict(cell=cell,host='r3500',resolution=scope)
                identity=dict(cell=cell,plan_sha256='plan',build_sha256='build',**outer)
                (root/'RUN.json').write_bytes(encoded(dict(identity,stage='running')))
                (root/'RESULT.json').write_bytes(encoded(dict(identity,stage='complete',exit_code=0)))
                copies={}
                if family=='extra_trees':
                    gate=dict(raw_ulp=0)
                    for name,key in [('model.opaque-vector.zcth','model_sha256'),('model.parity.npz','parity_sha256'),('model.parity.log','log_sha256')]:
                        (root/name).write_bytes(name.encode());gate[key]=sha(root/name)
                    qualification=dict(native_tree_gate=gate);corrupt=root/'model.parity.npz'
                else:
                    (root/'native').mkdir();(root/'native/model.bin').write_bytes(b'model')
                    qualification=dict(model_sha256=sha(root/'native/model.bin'));(root/'native-inputs').mkdir()
                    for name in ('inputs.npz','gram.npz','SPEC.json','INPUTS.json'):
                        (root/'native-inputs'/name).write_bytes(name.encode());copies[name]=dict(sha256=sha(root/'native-inputs'/name))
                    corrupt=root/'native-inputs/gram.npz'
                report=dict(cell=cell,plan_sha256='plan',admission_sha256='outer',outer_results_read=True,outer_test={},qualification=qualification)
                save(evaluation,report,[1,3,9],np.array([100.,95.,94.]))
                (evaluation/'REPORT.json').write_bytes(encoded(report));(evaluation/'COMPLETE.json').write_bytes(encoded(dict(cell='cell',report_sha256=sha(evaluation/'REPORT.json'))))
                (root/'WORKER_EXECUTION.json').write_bytes(encoded(dict(execution_host='r3500',package_versions={},native_replay_inputs=copies)))
                control.validate_result(record,root);corrupt.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'drift'):control.validate_result(record,root)


if __name__=='__main__':unittest.main()
