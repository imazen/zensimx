from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_archive_supervision import later_control_bundles


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))

class ControlArchiveTests(unittest.TestCase):
    def check_stage(self,stage,family):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study=Path(d);write(study/'PLAN.json',{})
            cell=dict(id='cell',stage=stage,family=family,partition='outer-0')
            context=dict(admission_sha256='admission',selected_setting=2)
            resolver='selected_cells';role='outer_test'
            if stage=='full_fit_selection':
                resolver='full_fit_cells';role='full_fit_validation';context.update(full_fit=True,selection_scope='full_fit_only',development_results_read=False,resolved_family=family)
            elif stage=='full_fit':
                resolver='final_refit_cells';role='exposed_development';context.update(full_fit=True,descriptive=True,selection_scope='full_fit_refit',development_results_read=False,resolved_family=family)
            elif stage=='human_only':
                variant=dict(cell=cell,resolved_family=family,admission_sha256='admission',schedule=dict(setting=2))
                context['variant']=variant
            run=study/'runs/cell/attempt-001'
            identity=dict(cell=cell,plan_sha256=sha(study/'PLAN.json'),build_sha256='build',**context)
            write(run/'RUN.json',identity);write(run/'RESULT.json',dict(identity,stage='complete',exit_code=0))
            model=run/('model.opaque-vector.zcth' if family=='extra_trees' else 'native/model.bin');model.parent.mkdir(parents=True,exist_ok=True);model.write_bytes(b'model')
            qualification=dict(native_tree_gate=dict(model_sha256=sha(model))) if family=='extra_trees' else dict(model_sha256=sha(model))
            panel=run/'evaluation';panel.mkdir();(panel/'predictions.npz').write_bytes(b'predictions')
            report=dict(cell=cell,plan_sha256=sha(study/'PLAN.json'),admission_sha256='admission',outer_results_read=True,qualification=qualification,predictions_sha256=sha(panel/'predictions.npz'),**{role:{}})
            def save():
                write(panel/'REPORT.json',report);write(panel/'COMPLETE.json',dict(cell='cell',report_sha256=sha(panel/'REPORT.json')))
            save()
            with patch('shaping_controls.'+resolver,return_value=([cell],context)) as admission,patch('shaping_variant_execution.resolve',return_value=context.get('variant')) as variant_resolver,patch('shaping_variant_inputs.load_variant'):
                def check():return later_control_bundles(study,run.parent,[cell])
                self.assertEqual(check(),[('control',run.parent)])
                gate=variant_resolver if stage=='human_only' else admission
                gate.side_effect=ValueError('admission drift')
                with self.assertRaisesRegex(ValueError,'admission drift'):check()
                gate.side_effect=None
                model.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'native model drift'):check()
                model.write_bytes(b'model');(panel/'predictions.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'prediction drift'):check()
                (panel/'predictions.npz').write_bytes(b'predictions')
                report['inner_validation']={};save()
                with self.assertRaisesRegex(ValueError,'response role'):check()
                del report['inner_validation'];save()
                identity['admission_sha256']='changed';write(run/'RUN.json',identity)
                with self.assertRaisesRegex(ValueError,'archive admission differs'):check()

    def test_original_outer_controls(self):
        for family in ('extra_trees','shaped_additive'):
            with self.subTest(family=family):self.check_stage('outer',family)

    def test_full_fit_search_and_refit_controls(self):
        for stage in ('full_fit_selection','full_fit'):
            for family in ('extra_trees','shaped_additive'):
                with self.subTest(stage=stage,family=family):self.check_stage(stage,family)

    def test_variant_controls(self):
        for family in ('extra_trees','shaped_additive'):
            with self.subTest(family=family):self.check_stage('human_only',family)
