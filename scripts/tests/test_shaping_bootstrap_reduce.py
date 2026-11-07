import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import FAMILIES,seeds
from shaping_bootstrap_plan import REPLICATES
from shaping_bootstrap_reduce import reduce,reduce_matched


class BootstrapReductionTests(unittest.TestCase):
    def fixture(self):
        points={f:np.zeros((3,len(tuple(seeds(f))),5,2)) for f in FAMILIES}
        samples={f:np.repeat(p[...,None],REPLICATES,axis=-1) for f,p in points.items()}
        return samples,points
    def test_teacher_changes_do_not_move_human_headline(self):
        s,p=self.fixture();before=reduce(s,p)
        s['softplus'][:,:,1,:,:]=1.;p['softplus'][:,:,1,:]=1.
        after=reduce(s,p)
        for scope in before['scopes']:
            self.assertEqual(before['scopes'][scope]['softplus']['equal_human_composite'],after['scopes'][scope]['softplus']['equal_human_composite'])
        self.assertEqual(after['scopes']['mean_outer']['softplus']['populations']['cid22_ssim2']['composite']['point'],1.)
    def test_equal_population_and_fold_weights_with_seed_performance(self):
        s,p=self.fixture()
        # One of four human populations, one of three folds and one of three
        # paired seeds has both rank metrics equal to one; all others zero.
        s['softplus'][0,0,0,:,:]=1.;p['softplus'][0,0,0,:]=1.
        r=reduce(s,p)['scopes'];self.assertAlmostEqual(r['mean_outer']['softplus']['equal_human_composite']['point'],1/36)
        self.assertAlmostEqual(r['outer-0']['softplus']['equal_human_composite']['point'],1/12)
    def test_paired_replicates_cancel_shared_source_variation(self):
        s,p=self.fixture();variation=np.linspace(-.25,.25,REPLICATES)
        s['relu'][...]=variation;s['softplus'][...]=variation+.125;p['softplus'][...]=.125
        r=reduce(s,p)['scopes']['mean_outer']
        np.testing.assert_allclose(r['softplus']['equal_human_composite']['difference_percentile95'],[.125,.125])
        self.assertEqual(r['relu']['equal_human_composite']['difference_percentile95'],[0.,0.])
    def test_incomplete_axes_nonfinite_or_missing_family_refused(self):
        s,p=self.fixture();s['relu']=s['relu'][:2]
        with self.assertRaises(ValueError):reduce(s,p)
        s,p=self.fixture();s['relu'][0,0,0,0,0]=np.nan
        with self.assertRaises(ValueError):reduce(s,p)
        s,p=self.fixture();s.pop('extra_trees')
        with self.assertRaises(ValueError):reduce(s,p)


class MatchedReductionTests(unittest.TestCase):
    def fixture(self):
        p={k:np.zeros((3,3,5,2)) for k in ('candidate','control')}
        return {k:np.repeat(v[...,None],REPLICATES,axis=-1) for k,v in p.items()},p

    def test_matched_source_variation_cancels_and_teacher_is_separate(self):
        s,p=self.fixture();variation=np.linspace(-.2,.2,REPLICATES)
        s['control'][...]=variation;s['candidate'][...]=variation+.125;p['candidate'][...]=.125
        result=reduce_matched(s,p,(0,1,2));human=result['scopes']['mean_outer']['equal_human_composite']
        np.testing.assert_allclose(human['difference_percentile95'],[.125,.125])
        self.assertEqual(human['difference_from_control'],.125)
        s['candidate'][:,:,1,:,:]=1;p['candidate'][:,:,1,:]=1
        self.assertEqual(human,reduce_matched(s,p,(0,1,2))['scopes']['mean_outer']['equal_human_composite'])

    def test_missing_seed_fold_or_nonfinite_rank_refused(self):
        s,p=self.fixture()
        with self.assertRaises(ValueError):reduce_matched(s,p,(0,1))
        s['candidate']=s['candidate'][:2]
        with self.assertRaises(ValueError):reduce_matched(s,p,(0,1,2))
        s,p=self.fixture();s['control'][0,0,0,0,0]=np.nan
        with self.assertRaises(ValueError):reduce_matched(s,p,(0,1,2))


if __name__=='__main__':unittest.main()
