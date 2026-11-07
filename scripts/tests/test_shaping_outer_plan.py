import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import FAMILIES,matrix
from shaping_outer_plan import plan,finalist_roles


def decisions():
    return {f:dict(family=f,outer=0,outer_results_read=False,setting=0,optimizer_choice='base',inner_score=.8) for f in FAMILIES}


class OuterPlanTests(unittest.TestCase):
    def test_exact_fresh_matches_only_and_original_cells_preserved(self):
        d=decisions();r=plan(d,[],0,matrix())
        self.assertEqual(len(r['fresh_controls']),3);self.assertEqual(r['additional_cells'],[])
        self.assertTrue(all(len(c['paired_families'])==4 for c in r['fresh_controls']))
        d['product_pretrained_frozen']['setting']=2
        d['product_pretrained_joint']['setting']=2
        r=plan(d,[],0,matrix());self.assertEqual(len(r['additional_cells']),3)
        extra=[v for v in r['fresh_controls'] if not v['original_registered_cell']]
        self.assertTrue(all(len(v['paired_families'])==2 for v in extra))
        d['product_pretrained_joint']['optimizer_choice']='constant_lr'
        self.assertEqual(len(plan(d,[],0,matrix())['additional_cells']),6)

    def test_extension_length_is_distinct_and_no_retuning(self):
        d=decisions();original=copy.deepcopy(d)
        r=plan(d,['relu','product_pretrained_joint'],0,matrix())
        self.assertEqual(d,original);self.assertEqual(r['schedules']['relu']['phase_epochs'],[480])
        self.assertEqual(r['schedules']['product_pretrained_joint']['phase_epochs'],[240,240])
        self.assertEqual(r['schedules']['product_pretrained_frozen']['phase_epochs'],[120,120])
        self.assertEqual(len(r['additional_cells']),3)
        self.assertTrue(all(c['epochs']==240 for c in r['additional_cells']))
        self.assertEqual(len({c['id'] for c in r['additional_cells']}),3)
        self.assertFalse(r['outer_evaluation_admitted'])
        with self.assertRaises(ValueError):plan(d,['product_pretrained_joint'],0,matrix())

    def test_inner_only_role_ties_and_incomplete_refusal(self):
        d=decisions();self.assertEqual(finalist_roles(d)['best_analytic'],'softplus')
        d['silu_monotone']['inner_score']=.8+5e-13
        self.assertEqual(finalist_roles(d)['best_analytic'],'softplus')
        d['silu_monotone']['inner_score']=.81
        self.assertEqual(finalist_roles(d)['best_analytic'],'silu_monotone')
        d['relu']['outer_results_read']=True
        with self.assertRaisesRegex(ValueError,'exposure'):plan(d,[],0,matrix())
        del d['relu']
        with self.assertRaises(ValueError):plan(d,[],0,matrix())

if __name__=='__main__':unittest.main()
