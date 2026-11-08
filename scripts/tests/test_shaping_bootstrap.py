import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_bootstrap import compute,compute_matched,verify_draws,run
from shaping_bootstrap_plan import population_groups,component_draws,SEED,REPLICATES
from shaping_bootstrap_reduce import POPULATIONS
from shaping_outer_evidence import primary_cells
from shaping_completion import matrix,encoded,sha


class BootstrapRunnerTests(unittest.TestCase):
    def test_reduced_roles_allow_fold_specific_families_and_require_every_seed(self):
        from shaping_completion import ROLES
        outputs,registration,draws,labels,*_=self.fixture()
        families=dict(zip(ROLES,('relu','extra_trees','shaped_additive','softplus',
                                 'product_joint_identity','separate_joint_identity')))
        selected=[]
        for row in outputs:
            mapping=dict(families)
            if row['cell']['partition']=='outer-1':mapping['best_analytic']='silu_monotone'
            for role,family in mapping.items():
                if row['cell']['family']==family:selected.append(dict(row,role=role))
        samples,points,summary=compute(selected,registration,draws,labels)
        self.assertEqual(set(samples),set(ROLES));self.assertEqual(len(selected),48)
        self.assertEqual(samples['shaped_additive'].shape,(3,1,5,2,REPLICATES))
        self.assertLess(summary['scopes']['mean_outer']['best_analytic']['equal_human_composite']['difference_from_relu'],0)
        with self.assertRaisesRegex(ValueError,'role/fold/seed'):compute(selected[:-1],registration,draws,labels)
        with self.assertRaisesRegex(ValueError,'role/fold/seed'):compute(selected[:-1]+[selected[0]],registration,draws,labels)

    def fixture(self):
        populations=[];components=[];labels=[];partitions={};strata=[];draws={};rng=np.random.default_rng(SEED)
        for fold in range(3):
            start=len(labels)
            for population in POPULATIONS:
                for source in range(2):
                    populations.extend([population]*3);components.extend([f'{fold}-{population}-{source}']*3);labels.extend([0.,1.,2.])
            ids=list(range(start,len(labels)));partitions[f'outer-{fold}']=dict(test=ids)
            groups=population_groups(ids,populations,components)
            for pi,(population,group) in enumerate(groups.items()):
                key=f'outer{fold}_population{pi}';draws[key]=component_draws(group,rng)
                strata.append(dict(outer=fold,population=population,array=key,groups=group,source_components=2,rows=6))
        registration=dict(seed=SEED,replicates=REPLICATES,scores_read=False,strata=strata)
        outputs=[]
        y=np.array([0.,1.,2.,0.,1.,2.]);base=np.array([0.,2.,1.,0.,1.,2.])
        for cell in primary_cells(matrix()):
            sign=-1 if cell['family']=='softplus' else 1;pred=sign*base
            point=dict(pooled=float(spearmanr(pred,y).statistic),within=float(np.mean([spearmanr(pred[:3],y[:3]).statistic,spearmanr(pred[3:],y[3:]).statistic])))
            outputs.append(dict(cell=cell,indices=partitions[cell['partition']]['test'],predictions=np.tile(pred,5),report=dict(outer_test=dict(summary={p:point for p in POPULATIONS}))))
        return outputs,registration,draws,np.array(labels),partitions,np.array(populations),np.array(components)

    def test_native_matched_comparison_keeps_pairing_and_rejects_missing_seed(self):
        outputs,registration,draws,labels,*_=self.fixture()
        arms={'candidate':[r for r in outputs if r['cell']['family']=='softplus'],
              'control':[r for r in outputs if r['cell']['family']=='relu']}
        samples,points,summary=compute_matched(arms,registration,draws,labels,(0,1,2))
        self.assertEqual(samples['candidate'].shape,(3,3,5,2,REPLICATES))
        point=summary['scopes']['mean_outer']['equal_human_composite']
        self.assertLess(point['difference_from_control'],0)
        self.assertFalse(summary['teacher_in_human_headline'])
        arms['candidate'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):compute_matched(arms,registration,draws,labels,(0,1,2))

    def test_complete_native_primary_computation(self):
        if not Path(os.environ['ZEN_PANEL_BIN']).is_file():raise ValueError('native panel binary missing')
        outputs,registration,draws,labels,partitions,pops,components=self.fixture()
        verify_draws(registration,draws,partitions,pops,components)
        completed=[];samples,points,summary=compute(outputs,registration,draws,labels,lambda s,r:completed.append(s['array']))
        self.assertEqual(len(completed),15)
        self.assertEqual(samples['relu'].shape,(3,3,5,2,2000))
        np.testing.assert_allclose(samples['softplus'],-samples['relu'],atol=1e-9)
        row=summary['scopes']['outer-0']
        self.assertAlmostEqual(row['relu']['populations']['kadid']['composite']['point'],.75)
        np.testing.assert_allclose(row['relu']['populations']['kadid']['composite']['percentile95'],[.5,1.],atol=1e-9)
        self.assertEqual(row['product_joint_identity']['equal_human_composite']['difference_percentile95'],[0.,0.])

    def test_registered_draw_and_membership_drift_refused(self):
        _,r,d,_,partitions,pops,components=self.fixture()
        altered={k:v.copy() for k,v in d.items()};key=next(iter(altered));altered[key][0,0]=1-altered[key][0,0]
        with self.assertRaisesRegex(ValueError,'cannot be reproduced'):verify_draws(r,altered,partitions,pops,components)
        changed=copy.deepcopy(r);changed['strata'][0]['rows']+=1
        with self.assertRaisesRegex(ValueError,'membership'):verify_draws(changed,d,partitions,pops,components)

    def test_runner_does_not_load_arrays_before_admission(self):
        with patch('shaping_bootstrap.collect',side_effect=ValueError('not admitted')),patch('shaping_bootstrap.np.load') as load:
            with self.assertRaisesRegex(ValueError,'not admitted'):run(SimpleNamespace(study=Path('not-admitted')))
            load.assert_not_called()

    def test_matched_runner_requires_admission_before_loading_arrays(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
            spec=Path(d)/'comparison.json';spec.write_bytes(encoded({'fresh_family':'product_pretrained_frozen'}))
            with patch('shaping_bootstrap_plan.load_comparison',return_value=({'fresh_family':'product_pretrained_frozen'},{})),patch('shaping_bootstrap.collect_matched',side_effect=ValueError('matched not admitted')),patch('shaping_bootstrap.np.load') as load:
                with self.assertRaisesRegex(ValueError,'not admitted'):run(SimpleNamespace(study=Path(d),comparison=spec))
                load.assert_not_called()

    def test_runner_writes_pinned_summary_after_complete_native_calculation(self):
        outputs,registration,draws,labels,partitions,pops,components=self.fixture()
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            root=Path(directory);study=root/'study';study.mkdir();old=root/'old';(old/'data').mkdir(parents=True)
            original=root/'original';(original/'data').mkdir(parents=True)
            np.save(old/'data/y.npy',labels);np.save(old/'data/pop.npy',pops);np.save(original/'data/component.npy',components)
            paths=[old/'data/y.npy',old/'data/pop.npy',original/'data/component.npy']
            plan=dict(original_plan=str(original/'PLAN.json'),minimum_ssd_free_bytes=30*1024**3,input_pins={str(p):sha(p) for p in paths})
            (study/'PLAN.json').write_bytes(encoded(plan));draw_dir=root/'draws';draw_dir.mkdir()
            np.savez_compressed(draw_dir/'draws.npz',**draws)
            registration.update(plan_sha256=sha(study/'PLAN.json'),draws_sha256=sha(draw_dir/'draws.npz'),input_sha256={str(p):sha(p) for p in paths[1:]})
            (draw_dir/'REGISTRATION.json').write_bytes(encoded(registration))
            args=SimpleNamespace(study=study,draws=draw_dir,binary_dir=Path(os.environ['ZEN_PANEL_BIN']).parent,output=root/'result')
            with patch('shaping_bootstrap.collect',return_value=(plan,partitions,outputs,{str(study/'PLAN.json'):sha(study/'PLAN.json')},{})),patch('shaped_linear_study.OLD',old):
                run(args)
            complete=json.loads((args.output/'COMPLETE.json').read_text())
            self.assertEqual(complete['summary_sha256'],sha(args.output/'SUMMARY.json'))
            summary=json.loads((args.output/'SUMMARY.json').read_text())
            self.assertEqual(summary['samples_sha256'],sha(args.output/'samples.npz'))
            self.assertEqual(len(summary['outer_population_reports']),102)
            self.assertFalse(summary['teacher_in_human_headline']);self.assertFalse(summary['final_holdout_read'])
            # Exercise the same pinned runner for a separate matched comparison.
            arms={'candidate':[r for r in outputs if r['cell']['family']=='softplus'],
                  'control':[r for r in outputs if r['cell']['family']=='relu']}
            from shaping_bootstrap_plan import register_matched
            with patch('shaping_bootstrap_plan.load_registration',return_value=(plan,matrix(),partitions)):
                registered=register_matched(study,root/'comparisons')
            args.comparison=root/'comparisons'/next(r['file'] for r in registered['comparisons'] if r['spec']==dict(role='best_analytic',condition={'stage':'human_only'}))
            args.output=root/'matched-result'
            with patch('shaping_bootstrap_plan.load_registration',return_value=(plan,matrix(),partitions)),patch('shaping_bootstrap.collect_matched',return_value=(plan,partitions,arms,{str(study/'PLAN.json'):sha(study/'PLAN.json')},{},(0,1,2))),patch('shaped_linear_study.OLD',old):
                run(args)
            matched=json.loads((args.output/'SUMMARY.json').read_text())
            self.assertEqual(matched['comparison'],json.loads(args.comparison.read_text()))
            self.assertEqual(matched['seed_ids'],[0,1,2]);self.assertEqual(len(matched['outer_population_reports']),18)
            self.assertIn(str(args.comparison),matched['input_sha256'])
            self.assertFalse(matched['teacher_in_human_headline'])



if __name__=='__main__':unittest.main()
