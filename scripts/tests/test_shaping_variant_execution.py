import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import sha
from shaping_variant_execution import command,optimizer,resolve,POPULATIONS


class VariantExecutionTests(unittest.TestCase):
    def test_reduced_population_and_human_only_commands_preserve_selected_settings(self):
        original=['run-heavy','--mem','1G','--jobs','1','--','/old/zensim_mlp_train',
                  '--group','old:training','--epochs','240','--out','old','--dump-checkpoints-dir','old',
                  '--lr','.0003','--l2','.00001','--init-seed','1001101','--sample-seed','1000101',
                  '--keep-features','frozen','--mse-weight','1']
        saved=copy.deepcopy(original)
        for absent in (['human_tid2013'],['cid22'],[]):
            groups=[(n,'/new/'+n,1.,0.,'withinref,both' if n=='cid22' else 'withinref,rank') for n in sorted(POPULATIONS-set(absent))]
            for epochs in (120,240,480):
                result=command(original,Path('/bin'),Path('/out'),epochs,groups,absent)
                self.assertNotIn('old:training',result)
                self.assertEqual(result.count('--group'),len(groups))
                self.assertEqual(result[result.index('--epochs')+1],str(epochs))
                self.assertEqual(result[result.index('--mse-weight')+1],'0' if 'cid22' in absent else '1')
                for flag in ('--lr','--l2','--init-seed','--sample-seed','--keep-features','--jobs'):
                    self.assertEqual(result[result.index(flag)+1],original[original.index(flag)+1])
            with self.assertRaises(ValueError):command(original,Path('/bin'),Path('/out'),240,groups[:-1],absent)
            bad=copy.deepcopy(groups);g=bad[0];bad[0]=(*g[:3],1.,g[4])
            with self.assertRaises(ValueError):command(original,Path('/bin'),Path('/out'),240,bad,absent)
        self.assertEqual(original,saved)

    def test_analytic_optimizer_excludes_teacher_only_when_absent(self):
        human=[dict(name=n,families=[0,1]) for n in sorted(POPULATIONS-{'cid22'})]
        for choice,scale in [('base',1.),('init_half',.5),('init_double',2.)]:
            result=optimizer(choice,True,human)
            self.assertTrue(result['human_only']);self.assertEqual(result['groups'],human)
            self.assertEqual(result['init_scale'],scale);self.assertEqual(result['bias_scale'],scale)
        self.assertNotIn('human_only',optimizer('base',True,[*human,dict(name='cid22',families=[0])]))

    def test_resolution_binds_audited_view_and_new_shape_not_inner_learned_shape(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
            for name in ('PLAN.json','VARIANT_MEMBERSHIPS.json','VARIANT_INPUT_PROGRESS.json','OUTER_ADMISSION.json'):write(root/name,{})
            cell=dict(id='c',stage='source_curve',partition='outer-0',family='best_product',seed=1)
            view=dict(train=[1,2],test=[3],teacher_training_rows=1);identifier='abcdef'
            groups=root/'comparators/variant-abcdef-true-0/GROUPS.json'
            write(groups.with_name('ADMISSION.json'),dict(cell=view))
            record=dict(view_identifier=identifier,permutation='true',permutation_seed=0,
                view_sha256=sha(root/'VARIANT_MEMBERSHIPS.json'),exposed_development_passed_to_trainer=False,
                outer_results_read=False,file_sha256={str(groups.with_name('ADMISSION.json')):sha(groups.with_name('ADMISSION.json'))})
            write(groups,record)
            audit=root/'VARIANT_INPUT_AUDIT.json'
            write(audit,dict(plan_sha256=sha(root/'PLAN.json'),progress_sha256=sha(root/'VARIANT_INPUT_PROGRESS.json'),
                feature_bits_and_keys_unchanged=True,training_labels_match_registration=True,outer_responses_used=False,
                views=[dict(path=str(groups),receipt_sha256=sha(groups))]))
            initial=root/'shapes/variant-abcdef/product/INITIAL.json';write(initial,{'trainable':True})
            write(initial.with_name('RECEIPT.json'),{})
            shapes=dict(plan_sha256=sha(root/'PLAN.json'),audit_sha256=sha(audit),membership_sha256=sha(root/'VARIANT_MEMBERSHIPS.json'),
                cells={'c':identifier},label_independent_initializations_only=True,learned_shapes_reused=False,outer_responses_used=False,
                views={identifier:dict(manifests={'product':dict(path=str(initial),sha256=sha(initial),receipt_sha256=sha(initial.with_name('RECEIPT.json')))})})
            write(root/'VARIANT_SHAPES.json',shapes)
            admission=dict(outer_evaluation_admitted=True,finalist_roles={'best_product':'product_pretrained_frozen'},
                schedules={'product_pretrained_frozen':dict(setting=3,optimizer_choice='constant_lr',phase_epochs=[240,240],total_pair_draws=24000000)})
            with patch('shaping_variant_execution.load_variant',return_value=({},cell,identifier,view)),\
                 patch('shaping_variant_execution.resolve_outer',return_value=(admission,root/'OUTER_ADMISSION.json')):
                result=resolve(root,'c')
                self.assertEqual(result['resolved_family'],'product_pretrained_frozen')
                self.assertEqual(result['initial_shape'],str(initial));self.assertEqual(result['schedule']['setting'],3)
                write(initial,{'trainable':False})
                with self.assertRaisesRegex(ValueError,'initial shape drift'):resolve(root,'c')
                write(initial,{'trainable':True})
                shapes['learned_shapes_reused']=True;write(root/'VARIANT_SHAPES.json',shapes)
                with self.assertRaisesRegex(ValueError,'shape preparation'):resolve(root,'c')
                admission['outer_evaluation_admitted']=False
                with self.assertRaisesRegex(ValueError,'not admitted'):resolve(root,'c')


if __name__=='__main__':unittest.main()
