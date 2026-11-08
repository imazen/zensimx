import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,encoded,sha
from shaping_evaluation import diagnostic_epochs
from shaping_evaluation_worker import request,_work,publish,verify_request,transfer_inputs


class WorkerEvaluationTests(unittest.TestCase):
    def test_full_host_is_probed_once_per_sweep_without_losing_waiting_cells(self):
        import shutil
        from shaping_evaluation_worker import watch
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run=self.fixture(root)
            cells=[c for c in matrix() if c['stage']=='inner' and c['family']=='silu_monotone'][:4]
            for i,cell in enumerate(cells):
                target=study/'runs'/cell['id']/'attempt-001'
                if target!=run:shutil.copytree(run,target)
                for name in ('RUN.json','RESULT.json'):
                    path=target/name;value=json.loads(path.read_text());value['cell']=cell
                    if name=='RUN.json':value['execution_host']='r5600g' if i<2 else 'i134'
                    path.write_bytes(encoded(value))
            args=SimpleNamespace(study=study,output=root/'output',remote_source=root/'source',watch=False,
                                 binary_dir=root/'bin',repair_binary_dir=root/'repair',inspector=root/'inspect')
            calls=[]
            def remote(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=int(command[0]=='ssh' and command[-1].startswith('test -f ')))
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})), \
                 patch('shaping_evaluation_worker.subprocess.run',side_effect=remote):
                for sweep in range(2):
                    status=watch(args)
                    self.assertEqual(set(status['waiting']),{c['id'] for c in cells})
                    self.assertEqual(status['capacity_wait_hosts'],['i134','r5600g'])
                    self.assertFalse(status['requires_reconciliation'])
                    work=[c for c in calls if c[0]=='ssh' and '--work' in c[-1]]
                    self.assertEqual(len(work),2*(sweep+1))

    def relu_fixture(self,root):
        study,run=self.fixture(root)
        cell=next(c for c in matrix() if c['stage']=='inner' and c['family']=='relu')
        target=run.parent.with_name(cell['id']);run.parent.rename(target);run=target/run.name
        for name in ('RUN.json','RESULT.json'):
            path=run/name;value=json.loads(path.read_text());value['cell']=cell;path.write_bytes(encoded(value))
        (run/'keep.txt').write_text('0\n1\n')
        return study,run

    def test_cross_host_request_keeps_training_origin_and_pins_portable_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study,run=self.relu_fixture(Path(directory));before=(run/'RUN.json').read_bytes()
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})):
                record=request(study,run,target_host='r3800x')
                with self.assertRaisesRegex(ValueError,'approved'):request(study,run,target_host='dev')
            self.assertEqual(record['host'],'r3800x');self.assertEqual(record['fit_execution_host'],'r5600g')
            self.assertEqual((run/'RUN.json').read_bytes(),before)
            self.assertTrue(any(n.endswith('/WORKER_COLLECTION.json') for n in record['source_sha256']))
            self.assertTrue(any(n.endswith('/keep.txt') for n in record['source_sha256']))
            self.assertEqual(set(record['input_sizes']),set(record['source_sha256']))

    def test_cross_host_scope_refuses_unqualified_family(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study,run=self.fixture(Path(directory))
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})):
                with self.assertRaisesRegex(ValueError,'ReLU controls only'):request(study,run,target_host='r3800x')

    def test_missing_transfer_inputs_are_preflight_only_and_drift_or_origin_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study,run=self.relu_fixture(Path(directory))
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})):
                record=request(study,run,target_host='r3800x')
            with patch('shaping_reduced_scope.admit_compute'),patch('shaping_evaluation_worker.socket.gethostname',return_value='r3800x'):
                verify_request(study,record)
                model=next((run/'checkpoints').glob('*.bin'));data=model.read_bytes();model.unlink()
                verify_request(study,record,allow_missing=True)
                with self.assertRaises(FileNotFoundError):verify_request(study,record)
                model.write_bytes(data+b'drift')
                with self.assertRaisesRegex(ValueError,'drift'):verify_request(study,record,allow_missing=True)
                model.write_bytes(data)
                record['fit_execution_host']='i134'
                with self.assertRaisesRegex(ValueError,'training origin'):verify_request(study,record)

    def test_transfer_refuses_overwrites_and_uses_bounded_receiver(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run=self.relu_fixture(root)
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})):
                record=request(study,run,target_host='r3800x')
            args=SimpleNamespace(study=study,remote_source=root/'export');path=root/'REQUEST.json';path.write_bytes(encoded(record))
            with patch('shaping_evaluation_worker.subprocess.run') as call:
                transfer_inputs(args,record,path,path)
            self.assertIn('--verify-inputs',call.call_args_list[0].args[0][-1])
            command=call.call_args_list[1].args[0]
            self.assertEqual(command[0],'rsync')
            for flag in ('--ignore-existing','--no-implied-dirs','--omit-dir-times'):self.assertIn(flag,command)
            receiver=next(v for v in command if v.startswith('--rsync-path='))
            self.assertIn('--mem 512M',receiver);self.assertIn('--receive',receiver)

    def test_cross_host_payload_and_symlink_guards_apply_during_preflight(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study,run=self.relu_fixture(Path(directory))
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})):
                record=request(study,run,target_host='r3800x')
            with patch('shaping_reduced_scope.admit_compute'),patch('shaping_evaluation_worker.socket.gethostname',return_value='r3800x'):
                name=next(iter(record['input_sizes']));original=record['input_sizes'][name]
                record['input_sizes'][name]=128*1024**2+1
                with self.assertRaisesRegex(ValueError,'input scope'):verify_request(study,record,allow_missing=True)
                record['input_sizes'][name]=original
                keep=run/'keep.txt';keep.unlink();keep.symlink_to(run/'train.log')
                with self.assertRaisesRegex(ValueError,'symlink'):verify_request(study,record,allow_missing=True)

    def test_nested_rescue_request_pins_native_phase_without_replaying_it(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run=self.fixture(root)
            import shutil
            fit=run/'fit';fit.mkdir();shutil.move(str(run/'checkpoints'),str(fit/'checkpoints'));shutil.move(str(run/'train.log'),str(fit/'train.log'))
            cell=next(c for c in matrix() if c['stage']=='rescue' and c['family']=='silu_monotone')
            for name in ('RUN.json','RESULT.json'):
                path=run/name;value=json.loads(path.read_text());value.update(cell=cell,fit_directory='fit');path.write_bytes(encoded(value))
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})):
                record=request(study,run)
            self.assertTrue(any('/fit/checkpoints/' in name for name in record['source_sha256']))
            self.assertTrue(any(name.endswith('/fit/train.log') for name in record['source_sha256']))

    def fixture(self,root):
        study=root/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
        (study/'REDUCED_SCOPE.json').write_text('{}')
        cell=next(c for c in matrix() if c['stage']=='inner' and c['family']=='silu_monotone')
        run=study/'runs'/cell['id']/'attempt-001';models=run/'checkpoints';models.mkdir(parents=True)
        for epoch in {239,*diagnostic_epochs(240)}:(models/f'ckpt_epoch{epoch:03}.bin').write_text(str(epoch))
        identity=dict(cell=cell,execution_host='r5600g',epochs=240)
        (run/'RUN.json').write_bytes(encoded(identity))
        (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',exit_code=0,epochs=240,bake_sha256=sha(models/'ckpt_epoch239.bin'))))
        (run/'WORKER_COLLECTION.json').write_text('{}')
        (run/'train.log').write_text('training evidence')
        return study,run

    def test_request_requires_collected_fit_and_refuses_prediction_replay(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,run=self.fixture(Path(d))
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})):
                record=request(study,run);self.assertEqual(record['host'],'r5600g')
                (run/'evaluation').mkdir()
                with self.assertRaisesRegex(ValueError,'never replay'):request(study,run)

    def test_worker_export_publishes_report_tree_and_binds_original_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study,run=self.fixture(root)
            with patch('shaping_execution.load_registration',return_value=({},matrix(),{})):record=request(study,run)
            request_path=root/'request.json';request_path.write_bytes(encoded(record))
            def evaluate(args):
                self.assertEqual(args.cells,(record['cell']['id'],))
                directory=run/'evaluation';directory.mkdir();(directory/'REPORT.json').write_text('report')
                (directory/'COMPLETE.json').write_text('marker')
                (run/'CONVERGENCE.json').write_text('convergence evidence')
            args=SimpleNamespace(study=study,request=request_path,bundle=root/'evaluation.tar',binary_dir=root/'bin')
            with patch('shaping_evaluation_worker.verify_request'),patch('shaping_evaluation_worker.capacity',return_value=True),patch('shaping_supervision.sweep',side_effect=evaluate),patch('shaping_evaluation_worker.socket.gethostname',return_value='r5600g'):
                result=_work(args)
            other=root/'coordinator';other.mkdir()
            import shutil
            for name in record['source_sha256']:
                target=other/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(study/name,target)
            receipt=publish(other,request_path,args.bundle,result['bundle_sha256'],root/'verified')
            target=other/'runs'/record['cell']['id']/'attempt-001/evaluation'
            self.assertEqual((target/'COMPLETE.json').read_text(),'marker')
            self.assertEqual(receipt['source_host'],'r5600g')
            self.assertTrue((target.parent/'EVALUATION_WORKER_IMPORT.json').is_file())
            with self.assertRaisesRegex(ValueError,'preserve existing'):publish(other,request_path,args.bundle,result['bundle_sha256'],root/'verified-2')
            (other/next(iter(record['source_sha256']))).write_text('changed')
            with self.assertRaisesRegex(ValueError,'fit changed'):publish(other,request_path,args.bundle,result['bundle_sha256'],root/'verified-3')

    def test_capacity_wait_performs_no_evaluation_or_export(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);p=root/'request.json';p.write_text('{}');args=SimpleNamespace(request=p,study=root,bundle=root/'export.tar')
            with patch('shaping_evaluation_worker.verify_request'),patch('shaping_evaluation_worker.capacity',return_value=False),patch('shaping_supervision.sweep') as evaluate:
                self.assertEqual(_work(args)['stage'],'waiting_for_capacity');evaluate.assert_not_called()
            self.assertFalse(args.bundle.exists())


if __name__=='__main__':unittest.main()
