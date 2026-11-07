import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_curve_stability import compare


class CurveStabilityTests(unittest.TestCase):
    def config(self,separate=False):
        return dict(schema='zensim-input-shaping-v1',terms=[dict(column=1,damage=1,reference=2 if separate else None,separate=separate,knots=[1.,2.],slopes=[1.,1.,1.],gate=[0.,.5,1.])])
    def test_linear_shape_is_knot_and_strength_invariant(self):
        a=self.config();b=copy.deepcopy(a);b['terms'][0].update(knots=[2.,3.],slopes=[2.,2.,2.])
        r=compare([a,b])[0];self.assertEqual(r['normalized_response_max_spread'],0.);self.assertEqual(r['strength_ratio'],2.)
    def test_equivalent_f32_spellings_have_identical_curves(self):
        a=self.config();b=copy.deepcopy(a);b['terms'][0]['slopes'][0]=1.00000001
        original=copy.deepcopy(b)
        self.assertEqual(compare([a,b])[0]['normalized_response_max_spread'],0.)
        self.assertEqual(b,original)
    def test_separate_gate_difference_is_not_hidden_by_damage_normalization(self):
        a=self.config(True);b=copy.deepcopy(a);b['terms'][0]['gate']=[1.,1.,0.]
        r=compare([a,b])[0];self.assertEqual(r['normalized_response_max_spread'],1.)
        b['terms'][0]['gate']=[0.,0.,0.];r=compare([a,b])[0];self.assertEqual(r['active_fits'],1);self.assertIsNone(r['normalized_response_max_spread'])
    def test_zero_terms_and_invalid_terms_are_not_stable_useful_candidates(self):
        a=self.config();a['terms'][0]['slopes']=[0.,0.,0.]
        r=compare([a,a])[0];self.assertEqual(r['active_fits'],0);self.assertIsNone(r['strength_ratio'])
        a['terms'][0]['slopes'][0]=-1.
        with self.assertRaises(ValueError):compare([a])


if __name__=='__main__':unittest.main()

class PairedSettingTests(unittest.TestCase):
    def test_two_settings_three_seeds_per_architecture_without_other_fold(self):
        from shaping_completion import matrix
        from shaping_curve_stability import selected_cells
        cells=matrix();chosen=selected_cells(cells,'direct_joint_both',0,[0,1],1)
        self.assertEqual(len(chosen),12)
        self.assertEqual({c['partition'] for c in chosen},{'inner-0-1'})
        self.assertEqual({c['setting'] for c in chosen},{0,1})
        with self.assertRaisesRegex(ValueError,'complete paired'):
            selected_cells([c for c in cells if c!=chosen[0]],'direct_joint_both',0,[0,1],1)
        with self.assertRaisesRegex(ValueError,'distinct settings'):
            selected_cells(cells,'direct_joint_both',0,[0,0],1)
        old=selected_cells(cells,'product_joint_identity',0,[0])
        self.assertEqual(len(old),6);self.assertEqual({c['partition'] for c in old},{'inner-0-0','inner-0-1'})
