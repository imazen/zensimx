"""Nontrivial admission/leakage controls for the registered study controller."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaped_linear_study as owner

class StudyControls(unittest.TestCase):
    def test_training_only_basis_and_anchor(self):
        x=np.zeros((9,576),np.float32)
        x[:8,0]=np.arange(8);x[:8,1]=np.arange(8)[::-1]
        declaration=dict(direct=[0,1],products=[[0,1]])
        ids=np.arange(8);w=np.ones(8)
        a=owner.fit_spec(x,ids,w,'positive_shaped_products',declaration)
        x[8]=1e20
        b=owner.fit_spec(x,ids,w,'positive_shaped_products',declaration)
        self.assertEqual(a,b)
        self.assertTrue(np.array_equal(owner.apply_basis(np.zeros((1,576),np.float32),a),np.zeros((1,9))))
        x[8]=1
        z=owner.apply_basis(x,a)
        bigger=x.copy();bigger[:,0]+=1
        self.assertTrue(np.all(owner.apply_basis(bigger,a)>=z))

    def test_components_never_split_across_populations(self):
        c=np.array(['a','a','b','c','d','e','f','g','h','i','j','k'])
        pop=np.array(['p','q']+['p','q']*5)
        ids=np.arange(len(c));a=owner.split_components(c,pop,ids,3,'test')
        b=owner.split_components(c,pop,ids[::-1],3,'test')
        self.assertEqual(a,b)
        self.assertEqual(set(a),set(c))
        self.assertEqual(set(a.values()),{0,1,2})

    def test_permutation_does_not_change_outer_labels_or_family_distribution(self):
        y=np.arange(12,dtype=float);d=dict(pop=np.array(['p']*6+['q']*6),family=np.array(['a','a','b','b','a','b']*2))
        ids=np.array([0,1,2,3,6,7,8,9]);out=owner.permuted(y,d,ids,'conditional',13)
        np.testing.assert_array_equal(out[[4,5,10,11]],y[[4,5,10,11]])
        for p in ('p','q'):
            for f in ('a','b'):
                q=ids[(d['pop'][ids]==p)&(d['family'][ids]==f)]
                np.testing.assert_array_equal(np.sort(out[q]),np.sort(y[q]))

if __name__=='__main__':unittest.main()
