import sys,unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_variant_audit import same_table,shuffled_indices
from shaped_linear_study import permuted


class VariantAuditTests(unittest.TestCase):
    def test_numeric_bits_keys_and_nulls(self):
        original=pa.table({'feature':pa.array([0.,None,2.],type=pa.float32()),'key':['a','b','c'],'human_score':[1.,2.,3.]})
        same_table(original,original)
        for changed in (
            original.set_column(0,'feature',pa.array([-0.,None,2.],type=pa.float32())),
            original.set_column(0,'feature',pa.array([0.,1.,2.],type=pa.float32())),
            original.set_column(1,'key',pa.array(['b','a','c']))):
            with self.assertRaises(ValueError):same_table(changed,original)
        changed=original.set_column(2,'human_score',pa.array([3.,2.,1.]))
        same_table(changed,original,exclude=('human_score',))
        with self.assertRaises(ValueError):same_table(changed,original)

    def test_permutation_replay_and_strata(self):
        ids=np.array([0,2,3,4,6,7,8,10,11,12,14,15])
        pop=np.array(['a']*8+['b']*8);family=np.array(['x','y']*8);labels=np.arange(16.)
        for mode in ('true','population','conditional'):
            for seed in (20261107,20261108):
                result=shuffled_indices(ids,pop,family,mode,seed)
                expected=labels if mode=='true' else permuted(labels,dict(pop=pop,family=family),ids,mode,seed)
                np.testing.assert_array_equal(labels[result],expected[ids])
                self.assertEqual(set(result),set(ids));np.testing.assert_array_equal(pop[result],pop[ids])
                if mode=='conditional':np.testing.assert_array_equal(family[result],family[ids])
        with self.assertRaises(ValueError):shuffled_indices(ids,pop,family,'wrong',0)


if __name__=='__main__':unittest.main()
