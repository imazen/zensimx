import hashlib,json,sys,tempfile,unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_full_fit_execution import search_cell,prepared,execute

class FullFitExecutionTests(unittest.TestCase):
    def fixture(self,root,family='product_pretrained_frozen'):
        def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(encoded(d))
        write(root/'PLAN.json',{})
        cell=dict(id='search',stage='full_fit_selection',family='best_product',partition='outer-2',setting=3,seed=2)
        member=dict(train=[1,2],test=[3]);partitions={'outer-2':member}
        phases=[120,120] if '_pretrained_' in family else [240]
        admission=dict(selection_cells=[cell],finalist_roles={'best_product':family},schedules={family:dict(phase_epochs=phases,total_pair_draws=sum(phases)*50000,optimizer_choice='init_double')},evidence_sha256={})
        source=root/'primary';write(source/'RUN.json',{})
        admission['evidence_sha256'][str(source/'RUN.json')]=sha(source/'RUN.json')
        path=root/'full_fit/PLAN.json';write(path,admission)
        groups=[(n,str(root/(n+'.parquet')),1.,0.,'withinref,both' if n=='cid22' else 'withinref,rank') for n in ['human_kadid','human_tid2013','human_konfig','human_cid22_a25','cid22']]
        group=root/'comparators/outer-2/GROUPS.json'
        write(group,dict(groups=groups,membership_sha256=hashlib.sha256(encoded(member)).hexdigest(),exposed_development_passed_to_trainer=False))
        shape=root/'shapes/outer-2/product/INITIAL.json';write(shape,dict(terms=[],trainable=True))
        receipt=dict(partition='outer-2',admission_sha256=sha(path),plan_sha256=sha(root/'PLAN.json'),file_sha256={str(group):sha(group),str(shape):sha(shape)},selection_scope='full_fit_only',development_results_read=False)
        write(root/'full_fit/inputs/outer-2/PREPARED.json',receipt)
        write(root/'binaries/BUILD.json',dict(binary_sha256={}))
        return cell,admission,path,partitions,source,receipt

    def test_native_search_retrains_shapes_then_fresh_head(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cell,admission,path,partitions,source,receipt=self.fixture(root)
            calls=[]
            def primary(root,cells,template,setting,family=None):
                self.assertEqual(template['partition'],'inner-0-0');self.assertEqual(template['seed'],2);self.assertEqual(setting,3)
                return source,{'command':[], 'epochs':240}
            def fit(*args,**kwargs):
                calls.append((args,kwargs));return {'bake_sha256':'final'}
            args=SimpleNamespace(study=root,cell='search',binary_dir=root/'binaries',neutral_qualification=root/'qualification',attempt=1)
            with patch('shaping_full_fit_execution.load_registration',return_value=({'minimum_ssd_free_bytes':0},[cell],partitions)),\
                 patch('shaping_full_fit_execution.resolve_admission',return_value=(admission,path)),\
                 patch('shaping_full_fit_execution.primary_source',side_effect=primary),\
                 patch('shaping_full_fit_execution.admitted_families',return_value=[{'name':'cid22','families':[0]}]),\
                 patch('shaping_full_fit_execution.reservation',return_value=nullcontext()),\
                 patch('shaping_full_fit_execution.phase',side_effect=fit),\
                 patch('shaping_full_fit_execution.subprocess.check_output',return_value='{"terms":[],"trainable":true}'):
                execute(args)
                self.assertEqual(len(calls),2)
                self.assertEqual([c[0][4].name for c in calls],['pretraining','fit'])
                self.assertEqual([c[0][5] for c in calls],[120,120])
                self.assertTrue(calls[0][0][8]['trainable']);self.assertFalse(calls[1][0][8]['trainable'])
                result=json.loads((root/'runs/search/attempt-001/RESULT.json').read_text())
                self.assertEqual(result['training_exposure_pair_draws'],12000000)
                self.assertEqual(result['selection_scope'],'full_fit_only');self.assertFalse(result['development_results_read'])
                self.assertEqual(result['resolved_family'],'product_pretrained_frozen')
                with self.assertRaisesRegex(ValueError,'preserve'):execute(args)

    def test_input_membership_and_admission_drift_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cell,a,path,partitions,source,receipt=self.fixture(root)
            with patch('shaping_full_fit_execution.load_registration',return_value=({},[cell],partitions)),patch('shaping_full_fit_execution.resolve_admission',return_value=(a,path)):
                self.assertEqual(prepared(root,'search'),receipt)
                partitions['outer-2']['train']=[1,3]
                with self.assertRaisesRegex(ValueError,'membership'):prepared(root,'search')
                partitions['outer-2']['train']=[1,2]
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'admission'):prepared(root,'search')

    def test_final_refit_cannot_be_run_as_selection(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cell,a,path,partitions,source,receipt=self.fixture(root)
            cell['stage']='full_fit'
            with patch('shaping_full_fit_execution.load_registration',return_value=({},[cell],partitions)),patch('shaping_full_fit_execution.resolve_admission',return_value=(a,path)):
                with self.assertRaisesRegex(ValueError,'scope'):search_cell(root,'search')
            self.assertFalse((root/'runs').exists())

if __name__=='__main__':unittest.main()
