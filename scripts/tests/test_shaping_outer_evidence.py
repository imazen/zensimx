import os,sys,tempfile,unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,FAMILIES,NEURAL,encoded,sha
from shaping_outer_evidence import primary_cells,collect
from shaping_prediction_evidence import save


class OuterEvidenceTests(unittest.TestCase):
    def test_reduced_primary_requires_all_role_seeds_and_consistent_scope(self):
        from shaping_completion import ROLES
        roles=dict(zip(ROLES,('relu','extra_trees','shaped_additive','softplus',
                              'product_joint_identity','separate_joint_identity')))
        admissions={o:dict(reduced_scope_sha256='approved',finalist_roles=dict(roles)) for o in range(3)}
        admissions[1]['finalist_roles']['best_analytic']='silu_monotone'
        selected=primary_cells(matrix(),admissions)
        self.assertEqual(len(selected),48)
        self.assertEqual({c['family'] for c in selected if c['partition']=='outer-1'},set(admissions[1]['finalist_roles'].values()))
        with self.assertRaisesRegex(ValueError,'48-cell'):primary_cells(selected[:-1],admissions)
        with self.assertRaisesRegex(ValueError,'48-cell'):primary_cells(selected[:-1]+[selected[0]],admissions)
        admissions[2]['reduced_scope_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'mixed reduced'):primary_cells(matrix(),admissions)

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


class MatchedCellTests(unittest.TestCase):
    def admissions(self,cells):
        from shaping_outer_plan import plan
        values={}
        for outer in range(3):
            decisions={f:dict(family=f,outer=outer,outer_results_read=False,setting=0,optimizer_choice='base',inner_score=.8) for f in FAMILIES}
            values[outer]=plan(decisions,[],outer,cells);values[outer]['outer_evaluation_admitted']=True
        return values

    def test_variant_pairs_use_inner_finalist_and_all_seeds(self):
        from shaping_outer_evidence import matched_cells
        cells=matrix();admissions=self.admissions(cells)
        pairs,seeds_=matched_cells(cells,admissions,role='best_analytic',condition=dict(stage='source_curve',fraction=.25,order_seed=20261107))
        self.assertEqual(len(pairs),9);self.assertEqual(seeds_,(0,1,2))
        self.assertTrue(all(p['control']['family']=='softplus' for p in pairs))
        missing=[c for c in cells if c!=pairs[-1]['candidate']]
        with self.assertRaisesRegex(ValueError,'exactly one'):matched_cells(missing,admissions,role='best_analytic',condition=dict(stage='source_curve',fraction=.25,order_seed=20261107))
        pairs,seeds_=matched_cells(cells,admissions,role='shaped_additive',condition=dict(stage='human_only'))
        self.assertEqual(len(pairs),3);self.assertEqual(seeds_,(None,))

    def test_pretrained_controls_match_actual_exposure_and_settings(self):
        from shaping_outer_evidence import matched_cells
        cells=matrix();admissions=self.admissions(cells)
        pairs,seeds_=matched_cells(cells,admissions,fresh_family='product_pretrained_frozen')
        self.assertEqual(len(pairs),9)
        for admission in admissions.values():
            next(v for v in admission['fresh_controls'] if 'product_pretrained_frozen' in v['paired_families'])['epochs']=240
        with self.assertRaisesRegex(ValueError,'exposure differs'):matched_cells(cells,admissions,fresh_family='product_pretrained_frozen')


class MatchedCollectorTests(unittest.TestCase):
    def test_complete_control_pair_and_pre_read_corruption_refusal(self):
        from shaping_outer_evidence import matched_cells,collect_matched
        from shaping_prediction_evidence import read as real_read
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells=matrix();(root/'PLAN.json').write_text('{}');plan_pin=sha(root/'PLAN.json')
            admissions=MatchedCellTests().admissions(cells);gates={};partitions={};resolutions={}
            for outer in range(3):
                admissions[outer]['qualifications']={};gate=root/f'admission-{outer}.json';gate.write_bytes(encoded(admissions[outer]));gates[outer]=(admissions[outer],gate)
                partitions[f'outer-{outer}']=dict(train=[0],test=[1,2])
            pairs,seed_ids=matched_cells(cells,admissions,role='shaped_additive',condition=dict(stage='human_only'))
            for pair in pairs:
                for cell in (pair['candidate'],pair['control']):
                    run=root/'runs'/cell['id']/'attempt-001';run.mkdir(parents=True)
                    outer=int(cell['partition'][-1]);gate=gates[outer][1]
                    record=dict(cell=cell,plan_sha256=plan_pin,admission_sha256=sha(gate),selected_setting=0)
                    if cell['stage']=='human_only':
                        resolution=dict(cell=cell,resolved_family='shaped_additive',schedule={'setting':0},admission_sha256=sha(gate))
                        record['variant']=resolution;resolutions[cell['id']]=resolution
                    (run/'RUN.json').write_bytes(encoded(record));(run/'RESULT.json').write_bytes(encoded(dict(record,stage='complete',exit_code=0)))
                    model=run/'native/model.bin';model.parent.mkdir();model.write_bytes(b'qualified-native-model')
                    report=dict(cell=cell,plan_sha256=plan_pin,admission_sha256=sha(gate),outer_results_read=True,outer_test={},qualification={'model_sha256':sha(model)})
                    evaluation=run/'evaluation';evaluation.mkdir();save(evaluation,report,np.arange(3),np.array([90.,80.,70.]))
                    (evaluation/'REPORT.json').write_bytes(encoded(report));(evaluation/'COMPLETE.json').write_bytes(encoded(dict(cell=cell['id'],report_sha256=sha(evaluation/'REPORT.json'))))
            kwargs=dict(role='shaped_additive',condition={'stage':'human_only'})
            with patch('shaping_outer_evidence.load_registration',return_value=({},cells,partitions)),patch('shaping_outer_evidence.resolve',side_effect=lambda root,fold:gates[fold]),patch('shaping_variant_execution.resolve',side_effect=lambda root,cell:resolutions[cell]),patch('shaping_variant_inputs.load_variant',return_value=({},None,None,dict(train=[0],test=[1,2]))),patch('shaping_outer_evidence.read',wraps=real_read) as reader:
                _,_,arms,pins,_,ids=collect_matched(root,**kwargs)
                self.assertEqual(ids,(None,));self.assertEqual(len(arms['candidate']),3);self.assertEqual(reader.call_count,6)
                reader.reset_mock();model.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'model drift'):collect_matched(root,**kwargs)
                reader.assert_not_called();model.write_bytes(b'qualified-native-model')
                (run/'RESULT.json').unlink()
                with self.assertRaisesRegex(ValueError,'unfinished'):collect_matched(root,**kwargs)
                reader.assert_not_called()

    def test_all_admissions_precede_matched_response_reads(self):
        from shaping_outer_evidence import collect_matched
        with patch('shaping_outer_evidence.load_registration',return_value=({},matrix(),{})),patch('shaping_outer_evidence.sha',return_value='plan'),patch('shaping_outer_evidence.read') as reader:
            def admission(root,fold):
                if fold==2:raise ValueError('third fold not admitted')
                return {},Path('gate')
            with patch('shaping_outer_evidence.resolve',side_effect=admission):
                with self.assertRaisesRegex(ValueError,'third fold'):collect_matched(Path('unopened'),fresh_family='product_pretrained_frozen')
                reader.assert_not_called()

    def test_neural_fresh_pair_requires_exact_phase_exposure_before_reads(self):
        from shaping_outer_evidence import matched_cells,collect_matched
        from shaping_prediction_evidence import read as real_read
        from shaping_outer_execution import resolve_cell
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells=matrix();(root/'PLAN.json').write_text('{}');plan_pin=sha(root/'PLAN.json')
            admissions=MatchedCellTests().admissions(cells);gates={};partitions={};contexts={}
            for outer in range(3):
                admissions[outer]['qualifications']={};gate=root/f'admission-{outer}.json';gate.write_bytes(encoded(admissions[outer]));gates[outer]=(admissions[outer],gate)
                partitions[f'outer-{outer}']=dict(train=[0],test=[1,2])
            pairs,_=matched_cells(cells,admissions,fresh_family='product_pretrained_frozen')
            for pair in pairs:
                for cell in (pair['candidate'],pair['control']):
                    run=root/'runs'/cell['id']/'attempt-001';run.mkdir(parents=True)
                    outer=int(cell['partition'][-1]);admission,gate=gates[outer];schedule=resolve_cell(cell,admission)
                    contexts[cell['id']]=dict(admission=admission,admission_sha256=sha(gate),schedule=schedule)
                    record=dict(cell=cell,plan_sha256=plan_pin,admission_sha256=sha(gate),schedule=schedule,
                                epochs=schedule['phase_epochs'][-1],training_exposure_pair_draws=schedule['total_pair_draws'],fit_directory='fit')
                    (run/'RUN.json').write_bytes(encoded(record))
                    model=run/'fit/checkpoints'/f"ckpt_epoch{record['epochs']-1:03}.bin";model.parent.mkdir(parents=True);model.write_bytes(b'neural-model')
                    result=dict(record,stage='trained',exit_code=0,bake_sha256=sha(model),phases=[dict(epochs=e,stage='trained',exit_code=0) for e in schedule['phase_epochs']])
                    (run/'RESULT.json').write_bytes(encoded(result))
                    report=dict(cell=cell,plan_sha256=plan_pin,admission_sha256=sha(gate),outer_results_read=True,outer_test={},checkpoint_epoch=record['epochs']-1,checkpoint_sha256=sha(model))
                    evaluation=run/'evaluation';evaluation.mkdir();save(evaluation,report,np.arange(3),np.array([90.,80.,70.]))
                    (evaluation/'REPORT.json').write_bytes(encoded(report));(evaluation/'COMPLETE.json').write_bytes(encoded(dict(cell=cell['id'],report_sha256=sha(evaluation/'REPORT.json'))))
            with patch('shaping_outer_evidence.load_registration',return_value=({},cells,partitions)),patch('shaping_outer_evidence.resolve',side_effect=lambda root,fold:gates[fold]),patch('shaping_evaluation.outer_context',side_effect=lambda root,record,cells:contexts[record['cell']['id']]),patch('shaping_outer_evidence.read',wraps=real_read) as reader:
                _,_,arms,_,_,ids=collect_matched(root,fresh_family='product_pretrained_frozen')
                self.assertEqual(ids,(0,1,2));self.assertEqual(len(arms['candidate']),9);self.assertEqual(reader.call_count,18)
                reader.reset_mock();result['phases'][0]['epochs']-=1;(run/'RESULT.json').write_bytes(encoded(result))
                with self.assertRaisesRegex(ValueError,'phase exposure'):collect_matched(root,fresh_family='product_pretrained_frozen')
                reader.assert_not_called()
