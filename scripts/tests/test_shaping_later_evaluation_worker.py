import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_evaluation_worker as worker
import shaping_supervision as supervision
from shaping_completion import encoded,sha


class LaterEvaluationWorkerTests(unittest.TestCase):
    def fixture(self,root,*,repaired=False,variant=False):
        study=root/'study';run=study/'runs/cell/attempt-001';models=run/'fit/checkpoints';models.mkdir(parents=True)
        for name in ('PLAN.json','REDUCED_SCOPE.json'):(study/name).write_text('{}')
        cell=dict(id='cell',stage='human_only' if variant else 'fresh_control_outer_matched',
                  family='best_analytic' if variant else 'relu',partition='outer-0')
        model=models/'ckpt_epoch239.bin';model.write_bytes(b'exact final epoch')
        identity=dict(cell=cell,execution_host='r5600g',epochs=240,fit_directory='fit')
        (run/'RUN.json').write_bytes(encoded(identity))
        (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',exit_code=0,epochs=240,bake_sha256=sha(model))))
        (run/'WORKER_COLLECTION.json').write_text('{}');(run/'fit/train.log').write_text('native log')
        family='softplus' if variant else 'relu'
        context=dict(admission_sha256='frozen-admission',schedule=dict(phase_epochs=[240]),
                     admission=dict(evaluation_representations={family:dict(representation='research-relu-max-pairs-v1' if repaired else 'original')}))
        if variant:context['variant']=dict(resolved_family=family)
        return study,run,cell,context

    def test_later_requests_pin_final_only_and_reproduce_the_original_scope(self):
        for variant in (False,True):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                study,run,cell,context=self.fixture(Path(directory),variant=variant)
                with patch('shaping_execution.load_registration',return_value=({},[],{})), \
                     patch('shaping_evaluation.outer_context',return_value=context):
                    record=worker.request(study,run)
                    checkpoints=[n for n in record['source_sha256'] if '/checkpoints/' in n]
                    self.assertEqual(checkpoints,[str((run/'fit/checkpoints/ckpt_epoch239.bin').relative_to(study))])
                    self.assertEqual(record['later_scope']['resolved_family'],'softplus' if variant else 'relu')
                    with patch('shaping_reduced_scope.admit_compute'),patch.object(worker.socket,'gethostname',return_value='r5600g'):
                        worker.verify_request(study,record)
                        record['later_scope']['admission_sha256']='changed'
                        with self.assertRaisesRegex(ValueError,'original admission'):worker.verify_request(study,record)
                    with self.assertRaisesRegex(ValueError,'primary ReLU'):worker.request(study,run,target_host='r3800x')

    def test_missing_outer_admission_cannot_become_a_later_request(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study,run,cell,context=self.fixture(Path(directory))
            with patch('shaping_execution.load_registration',return_value=({},[],{})),patch('shaping_evaluation.outer_context',return_value=None):
                with self.assertRaisesRegex(ValueError,'original outer admission'):worker.request(study,run)

    def test_worker_uses_later_owner_and_import_preserves_outer_report_scope(self):
        for repaired in (False,True):
            with self.subTest(repaired=repaired),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);study,run,cell,context=self.fixture(root,repaired=repaired,variant=True)
                with patch('shaping_execution.load_registration',return_value=({},[],{})),patch('shaping_evaluation.outer_context',return_value=context):record=worker.request(study,run)
                request=root/'REQUEST.json';request.write_bytes(encoded(record))
                base=Path('reexport_max_pairs/cell/attempt-001/final') if repaired else Path('runs/cell/attempt-001/evaluation')
                report=dict(outer_results_read=True,admission_sha256='frozen-admission',cell=cell,outer_test={'saved':'evidence'})
                def evaluate(args):
                    self.assertEqual(args.cells,('cell',))
                    target=study/base;target.mkdir(parents=True)
                    (target/'REPORT.json').write_bytes(encoded(report))
                    (target/'COMPLETE.json').write_bytes(encoded(dict(cell='cell',report_sha256=sha(target/'REPORT.json'))))
                args=SimpleNamespace(study=study,request=request,bundle=root/'evaluation.tar',binary_dir=root/'bin')
                with patch.object(worker,'verify_request'),patch.object(worker,'capacity',return_value=True), \
                     patch.object(supervision,'later_sweep',side_effect=evaluate) as later, \
                     patch.object(supervision,'sweep') as primary,patch('shaping_reexport_supervision.sweep') as supplemental, \
                     patch.object(worker.socket,'gethostname',return_value='r5600g'):
                    result=worker._work(args)
                    later.assert_called_once();primary.assert_not_called();supplemental.assert_not_called()
                coordinator=root/'coordinator';coordinator.mkdir()
                import shutil
                for name in record['source_sha256']:
                    target=coordinator/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(study/name,target)
                receipt=worker.publish(coordinator,request,args.bundle,result['bundle_sha256'],root/'verified')
                self.assertEqual(json.loads((coordinator/base/'REPORT.json').read_text()),report)
                self.assertEqual(receipt['source_host'],'r5600g')

    def test_later_terminal_requires_final_report_or_the_actual_failed_gate(self):
        for repaired in (False,True):
            record=dict(cell=dict(id='c'),attempt='attempt-001',later_scope=dict(representation=dict(representation='research-relu-max-pairs-v1' if repaired else 'original')))
            parent='reexport_max_pairs/c/attempt-001' if repaired else 'runs/c/attempt-001'
            final=parent+('/final' if repaired else '/evaluation')
            with self.assertRaisesRegex(ValueError,'later-stage'):worker.require_terminal(record,{parent+'/CONVERGENCE.json':{}})
            with self.assertRaisesRegex(ValueError,'later-stage'):worker.require_terminal(record,{parent+'/LATER_EVALUATION_FAILURE.json':{}})
            worker.require_terminal(record,{parent+'/LATER_EVALUATION_FAILURE.json':{},final+'/FAILED_GATE.json':{}})
            worker.require_terminal(record,{final+'/REPORT.json':{},final+'/COMPLETE.json':{}})

    def test_later_sweep_cannot_open_an_unselected_run(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study=Path(directory)
            other=study/'runs/a-unselected/attempt-001';other.mkdir(parents=True)
            (other/'RESULT.json').write_bytes(encoded(dict(cell=dict(id='a-unselected',stage='outer'),stage='trained')))
            selected=study/'runs/z-selected/attempt-001';selected.mkdir(parents=True)
            binary=study/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            cell=dict(id='z-selected',stage='outer',family='silu_monotone')
            (selected/'RUN.json').write_bytes(encoded(dict(cell=cell,build_sha256=sha(binary/'BUILD.json'))))
            (selected/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained')))
            context=dict(admission=dict(evaluation_representations={'silu_monotone':dict(representation='original')}))
            args=SimpleNamespace(study=study,binary_dir=binary,cells=('z-selected',))
            def evaluate(a):
                self.assertEqual(a.run,selected);self.assertIsNone(a.checkpoint)
                dest=selected/'evaluation';dest.mkdir();(dest/'REPORT.json').write_text('{}')
                (dest/'COMPLETE.json').write_bytes(encoded(dict(cell='z-selected',report_sha256=sha(dest/'REPORT.json'))))
            with patch.object(supervision,'load_registration',return_value=({},[cell],{})), \
                 patch('shaping_evaluation.outer_context',return_value=context) as gate,patch.object(supervision,'evaluate',side_effect=evaluate):
                self.assertEqual(supervision.later_sweep(args),1)
                self.assertEqual(gate.call_count,1)


if __name__=='__main__':unittest.main()
