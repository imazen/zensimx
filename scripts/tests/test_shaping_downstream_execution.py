"""Exercise admitted controllers through terminal receipts without native fitting."""
import copy,json,sys,tempfile,unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_outer_execution as outer
import shaping_variant_execution as variant
from shaping_completion import sha


class DownstreamExecutionTests(unittest.TestCase):
    def run_case(self,module,family):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value));return path
            write(root/'PLAN.json',{})
            binaries=root/'binaries';write(binaries/'BUILD.json',{})
            source=root/'source';parent=root/'parent'
            write(source/'RUN.json',{'template':'selected head'})
            write(parent/'RUN.json',{'template':'selected shaper'})
            original={'command':['selected-head-command']}
            parent_run={'command':['selected-shaper-command']}
            pretrained='_pretrained_' in family
            cell=dict(id='cell',family=family,partition='outer-0',stage='outer' if module is outer else 'source_curve')
            schedule=dict(setting=2,optimizer_choice='constant_lr',phase_epochs=[120,120] if pretrained else [240],total_pair_draws=12000000)
            shape=dict(terms=[dict(feature=0,knots=[0.,1.],slopes=[1.])],trainable=True)
            shape_path=write(root/'shapes/outer-0/product/INITIAL.json',shape)
            admission_path=write(root/'admission.json',{})
            resolution=dict(cell=cell,initial_shape=str(shape_path) if pretrained else None)
            context=dict(plan=dict(minimum_ssd_free_bytes=0),cells=[],admission=dict(decision_sha256={str(parent/'RUN.json'):sha(parent/'RUN.json')}),
                admission_path=admission_path,cell=cell,family=family,schedule=schedule,record=dict(groups=[],explicit_absent_fit_groups=[]),
                groups=[],source=source,original=original,analytic=False,outer=0,resolution=resolution,
                qualification_pins={},templates=[(source,original),*([(parent,parent_run)] if pretrained else [])])
            args=SimpleNamespace(study=root,binary_dir=binaries,outer=0,cell='cell',attempt=1)
            calls=[]
            def phase(study,binary_dir,selected,selected_run,dest,epochs,choice,groups,shaping,**kwargs):
                calls.append(dict(source=selected,original=selected_run,phase=dest.name,epochs=epochs,shaping=copy.deepcopy(shaping)))
                return dict(stage='trained',exit_code=0,bake_sha256='final-model',epochs=epochs)
            with patch.object(module,'admit',return_value=context),patch.object(module,'reservation',return_value=nullcontext()), \
                 patch.object(module,'phase',side_effect=phase),patch.object(module,'optimizer',return_value={}), \
                 patch.object(module.subprocess,'check_output',return_value=json.dumps(shape)), \
                 patch.object(module,'primary_source',side_effect=AssertionError('execution must reuse admitted templates')):
                if module is variant:
                    with patch.object(module,'resolve',return_value=resolution):module.execute(args)
                else:module.execute(args)
            result=json.loads((root/'runs/cell/attempt-001/RESULT.json').read_text())
            self.assertEqual(result['stage'],'trained');self.assertEqual(result['exit_code'],0)
            self.assertEqual(result['build_sha256'],sha(binaries/'BUILD.json'))
            self.assertEqual(result['training_exposure_pair_draws'],12000000)
            self.assertEqual([c['phase'] for c in calls],['pretraining','fit'] if pretrained else ['fit'])
            self.assertEqual(calls[-1]['source'],source);self.assertEqual(calls[-1]['original'],original)
            if pretrained:
                self.assertEqual(calls[0]['source'],parent);self.assertEqual(calls[0]['original'],parent_run)
                self.assertEqual(calls[-1]['shaping']['trainable'],family.endswith('_joint'))
            self.assertEqual(result['outer_results_read'],False)

    def test_outer_selected_templates_reach_terminal_receipt(self):
        for family in ('relu','product_pretrained_frozen','product_pretrained_joint'):
            with self.subTest(family=family):self.run_case(outer,family)

    def test_variants_preserve_build_identity_and_selected_templates(self):
        for family in ('relu','product_pretrained_frozen','product_pretrained_joint'):
            with self.subTest(family=family):self.run_case(variant,family)


if __name__=='__main__':unittest.main()
