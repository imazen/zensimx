import copy
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import NEURAL
from shaping_extension import matrix


def selections():
    return {family:dict(family=family,outer=0,outer_results_read=False,base_setting=1,
                       selected_choice='constant_lr',selected_convergence={
                           f'{fold}-{seed}':dict(cell=dict(partition=f'inner-0-{fold}',seed=seed),reasons=[])
                           for fold in range(2) for seed in range(3)}) for family in NEURAL}


class ExtensionTests(unittest.TestCase):
    def test_plateau_collapse_is_not_mislabeled_as_continuing_improvement(self):
        d=selections();d['softplus']['selected_convergence']['0-0']['reasons']=['one_output_coefficient']
        self.assertEqual(matrix(d,0)['cells'],[])
        del d['relu']
        with self.assertRaisesRegex(ValueError,'all neural'):matrix(d,0)

    def test_improvement_extends_every_pair_and_relu_control(self):
        d=selections();d['softplus']['selected_convergence']['0-0']['reasons']=['loss_improvement_over_one_percent_or_rounding_boundary']
        r=matrix(d,0)
        self.assertEqual(r['extended_families'],['relu','softplus']);self.assertEqual(len(r['cells']),12)
        self.assertTrue(all(c['total_pair_draws']==24000000 and c['phase_epochs']==[480] for c in r['cells']))
        self.assertTrue(all(c['optimizer_choice']=='constant_lr' and c['setting']==1 for c in r['cells']))
        d['softplus']['selected_convergence'].pop('1-2')
        with self.assertRaisesRegex(ValueError,'pairing'):matrix(d,0)

    def test_pretrained_exposure_and_fresh_controls_are_paired_and_deduplicated(self):
        d=selections()
        for family in ('product_pretrained_frozen','product_pretrained_joint'):
            d[family]['selected_convergence']['1-2']['reasons']=['same_cycle_inner_composite_improvement_over_0.002']
        original=copy.deepcopy(d);r=matrix(d,0)
        self.assertEqual(d,original);self.assertEqual(len(r['cells']),24)
        fresh=[c for c in r['cells'] if c['role']=='fresh_control']
        self.assertEqual(len(fresh),6)
        self.assertTrue(all(c['phase_epochs']==[240] and c['total_pair_draws']==12000000 for c in fresh))
        self.assertTrue(all(len(c['paired_families'])==2 for c in fresh))
        self.assertTrue(all(c['phase_epochs']==[240,240] for c in r['cells'] if '_pretrained_' in c['family']))
        d['product_pretrained_joint']['selected_choice']='init_double'
        self.assertEqual(len(matrix(d,0)['cells']),30)


if __name__=='__main__':unittest.main()
