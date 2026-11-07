import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_pretraining import matching_parent,structural_terms
from shaping_execution import registered_subset


class PrefixTests(unittest.TestCase):
    def test_native_knot_decimal_roundtrip_is_bit_exact(self):
        a={'terms':[dict(column=0,knots=[0.03746293485164642,0.17839227616786957],slopes=[1,1,1])]}
        b={'terms':[dict(column=0,knots=[0.037462935,0.17839228],slopes=[0,0,0])]}
        self.assertEqual(structural_terms(a),structural_terms(b))
        b['terms'][0]['knots'][0]=0.03746293857693672
        self.assertNotEqual(structural_terms(a),structural_terms(b))
        b['terms'][0]['knots'][0]=float('nan')
        with self.assertRaises(ValueError):structural_terms(b)

    def test_dispatch_subset_cannot_add_or_duplicate_registered_cells(self):
        cells=[dict(id='a'),dict(id='b'),dict(id='c')]
        self.assertEqual(registered_subset(cells,['c','a']),[cells[0],cells[2]])
        self.assertEqual(len(cells),3)
        for requested in (['a','a'],['outer-cell'],['a','foreign']):
            with self.assertRaises(ValueError):registered_subset(cells,requested)

    def test_parent_requires_same_fold_setting_seed_and_shape_family(self):
        child=dict(stage='inner',family='product_pretrained_frozen',partition='inner-0-1',setting=2,seed=1)
        parent={**child,'family':'product_joint_identity'}
        self.assertEqual(matching_parent(child,[parent]),parent)
        for key,value in [('partition','inner-1-0'),('setting',0),('seed',0),('family','separate_joint_identity'),('stage','outer')]:
            with self.assertRaises(ValueError):matching_parent(child,[{**parent,key:value}])
        with self.assertRaises(ValueError):matching_parent(child,[parent,parent])

    def test_outer_and_nonpretrained_children_are_refused(self):
        for child in [dict(stage='outer',family='product_pretrained_frozen'),dict(stage='inner',family='relu')]:
            with self.assertRaises(ValueError):matching_parent(child,[])


if __name__=='__main__':unittest.main()
