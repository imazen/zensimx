import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,encoded,sha
from shaping_evaluation import diagnostic_epochs
from shaping_evaluation_worker import request,_work,publish


class WorkerEvaluationTests(unittest.TestCase):
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
