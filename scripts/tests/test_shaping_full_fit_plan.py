import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,FAMILIES
from shaping_full_fit_plan import plan,select,HUMANS


def evidence():
    return {o:{f:dict(family=f,outer=o,outer_results_read=False,inner_score=.8,
                     choice_scores={'base':.8,'init_half':.81},
                     phase_epochs=[120,120] if '_pretrained_' in f else [240]) for f in FAMILIES} for o in range(3)}


class FullFitPlanningTests(unittest.TestCase):
    def test_family_ties_common_optimizer_evidence_and_longest_registered_schedule(self):
        folds=evidence();folds[2]['relu']['choice_scores']={'base':.8}
        folds[1]['softplus']['phase_epochs']=[480]
        result=plan(folds,matrix())
        self.assertEqual(result['finalist_roles']['best_analytic'],'softplus')
        self.assertEqual(result['finalist_roles']['best_product'],'product_joint_identity')
        self.assertEqual(result['schedules']['relu']['optimizer_choice'],'base')
        self.assertEqual(result['schedules']['softplus']['optimizer_choice'],'init_half')
        self.assertEqual(result['schedules']['softplus']['phase_epochs'],[480])
        self.assertEqual(len(result['selection_cells']),192);self.assertEqual(len(result['final_cells']),16)
        self.assertFalse(result['development_results_read'])
        folds[0]['softplus']['outer_results_read']=True
        with self.assertRaisesRegex(ValueError,'unexposed'):plan(folds,matrix())

    def test_incomplete_pairing_and_unregistered_exposure_refuse(self):
        folds=evidence()
        with self.assertRaises(ValueError):plan({0:folds[0]},matrix())
        folds[0]['relu']['phase_epochs']=[120]
        with self.assertRaisesRegex(ValueError,'exposure'):plan(folds,matrix())
        with self.assertRaisesRegex(ValueError,'coverage'):plan(evidence(),[c for c in matrix() if c['stage']!='full_fit_selection'])

    def test_full_fit_selection_uses_all_pairs_humans_only_and_explicit_scope(self):
        cells=matrix();reports={}
        for c in cells:
            if c['stage']!='full_fit_selection' or c['family']!='best_analytic':continue
            value=(.8,.81,.81+1e-13,.79)[c['setting']]
            summary={p:dict(pooled=value,within=value) for p in HUMANS}
            summary['cid22_ssim2']=dict(pooled=100 if c['setting']==3 else 0,within=100)
            reports[c['id']]=dict(cell=c,resolved_family='softplus',selection_scope='full_fit_only',
                development_results_read=False,full_fit_validation=dict(summary=summary,selection_score=value))
        result=select(cells,reports,'best_analytic','softplus');self.assertEqual(result['selected_setting'],1)
        key=next(iter(reports));missing=copy.deepcopy(reports);missing.pop(key)
        with self.assertRaisesRegex(ValueError,'missing'):select(cells,missing,'best_analytic','softplus')
        for field,value in [('selection_scope','outer_comparison'),('development_results_read',True),('resolved_family','relu')]:
            altered=copy.deepcopy(reports);altered[key][field]=value
            with self.assertRaisesRegex(ValueError,'scope'):select(cells,altered,'best_analytic','softplus')


if __name__=='__main__':unittest.main()
