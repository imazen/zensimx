import os,sys,tempfile,unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,FAMILIES,NEURAL,encoded,sha
from shaping_outer_evidence import primary_cells,collect
from shaping_prediction_evidence import save


class OuterEvidenceTests(unittest.TestCase):
    def test_complete_fixture_and_changed_prediction_table(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}');plan_pin=sha(root/'PLAN.json')
            cells=matrix();admissions={};partitions={}
            for fold in range(3):
                path=root/f'admission-{fold}.json';path.write_text('{}')
                schedules={family:dict(setting=0,optimizer_choice='base',phase_epochs=[1] if family in NEURAL else [],total_pair_draws=50000 if family in NEURAL else 0,exposure_extended=False) for family in FAMILIES}
                representations={f:dict(representation='original') for f in NEURAL}
                representations['relu']=dict(representation='research-relu-max-pairs-v1',evaluation_build_sha256='eval-build')
                admissions[fold]=(dict(qualifications={},schedules=schedules,evaluation_representations=representations),path)
                partitions[f'outer-{fold}']=dict(train=[0,1],test=[2])
            last=None
            for cell in primary_cells(cells):
                run=root/'runs'/cell['id']/'attempt-001';run.mkdir(parents=True)
                fold=int(cell['partition'][-1]);admission,path=admissions[fold]
                record=dict(cell=cell,plan_sha256=plan_pin,admission_sha256=sha(path),build_sha256='training-build')
                if cell['family'] in NEURAL:
                    record.update(schedule={**admission['schedules'][cell['family']],'fresh':False},epochs=1)
                    model=run/'fit/checkpoints/ckpt_epoch000.bin';model.parent.mkdir(parents=True);model.write_bytes(b'fixture-model')
                    result=dict(record,stage='trained',exit_code=0,bake_sha256=sha(model))
                else:
                    record['selected_setting']=0;result=dict(record,stage='complete',exit_code=0)
                (run/'RUN.json').write_bytes(encoded(record));(run/'RESULT.json').write_bytes(encoded(result))
                report=dict(cell=cell,plan_sha256=plan_pin,admission_sha256=sha(path),outer_results_read=True,outer_test={})
                if cell['family'] in NEURAL:report['checkpoint_sha256']=result['bake_sha256']
                evaluation=run/'evaluation'
                if cell['family']=='relu':
                    evaluation=root/'reexport_max_pairs'/cell['id']/run.name/'final'
                    repair=root/'reexports'/cell['id'];repair.mkdir(parents=True)
                    bake=repair/'repaired.bin';bake.write_bytes(b'repaired-fixture')
                    receipt=repair/'RECEIPT.json';receipt.write_bytes(encoded(dict(bake=str(bake),file_sha256={str(bake):sha(bake)},original_sha256=result['bake_sha256'],training_build_sha256='training-build',evaluation_build_sha256='eval-build',representation='research-relu-max-pairs-v1')))
                    report.update(checkpoint_sha256=sha(bake),reexport_receipt=str(receipt),reexport_receipt_sha256=sha(receipt),evaluation_build_sha256='eval-build')
                evaluation.mkdir(parents=True);save(evaluation,report,np.arange(3),np.array([90.,80.,70.]))
                (evaluation/'REPORT.json').write_bytes(encoded(report));(evaluation/'COMPLETE.json').write_bytes(encoded(dict(cell=cell['id'],report_sha256=sha(evaluation/'REPORT.json'))));last=evaluation
            with patch('shaping_outer_evidence.load_registration',return_value=({},cells,partitions)),patch('shaping_outer_evidence.resolve',side_effect=lambda root,fold:admissions[fold]):
                _,_,outputs,pins,_=collect(root);self.assertEqual(len(outputs),102)
                self.assertTrue(all(len(o['predictions'])==3 for o in outputs))
                self.assertIn(str(last/'predictions.npz'),pins)
                save(last,{},np.arange(3),np.array([90.,80.,69.]))
                with self.assertRaisesRegex(ValueError,'not bound'):collect(root)

    def test_requires_all_families_folds_and_paired_seeds(self):
        cells=matrix();outer=primary_cells(cells);self.assertEqual(len(outer),102)
        with self.assertRaises(ValueError):primary_cells(outer[:-1])
        duplicate=outer[:-1]+[outer[0]]
        with self.assertRaises(ValueError):primary_cells(duplicate)

    def test_all_admission_gates_precede_response_reader(self):
        with patch('shaping_outer_evidence.load_registration',return_value=({},matrix(),{})),patch('shaping_outer_evidence.sha',return_value='plan'),patch('shaping_outer_evidence.read') as reader:
            def admission(root,fold):
                if fold==2:raise ValueError('third fold not admitted')
                return {},Path('admission')
            with patch('shaping_outer_evidence.resolve',side_effect=admission) as gate:
                with self.assertRaisesRegex(ValueError,'third fold'):collect(Path('unopened'))
                self.assertEqual(gate.call_count,3);reader.assert_not_called()

    def test_missing_outer_fit_does_not_open_prediction_tables(self):
        admitted=dict(qualifications={},schedules={})
        with patch('shaping_outer_evidence.load_registration',return_value=({},matrix(),{})),patch('shaping_outer_evidence.sha',return_value='plan'),patch('shaping_outer_evidence.resolve',return_value=(admitted,Path('admission'))),patch('shaping_outer_evidence.read') as reader:
            with self.assertRaisesRegex(ValueError,'exactly one complete'):collect(Path('/nonexistent-shaping-evidence-fixture'))
            reader.assert_not_called()


if __name__=='__main__':unittest.main()
