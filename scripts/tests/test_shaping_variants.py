import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_variants import membership


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.component=np.repeat(np.array(['a','b','c','d','held']),8)
        self.pop=np.array(['human']*16+['cid22_ssim2']*16+['human']*8)
        self.base={'train':list(range(32)),'test':list(range(32,40))}
    def make(self,stage,fraction=.25):
        return membership(dict(stage=stage,fraction=fraction,partition='outer-0',order_seed=20261107),self.base,self.component,self.pop)
    def test_source_curves_are_nested_complete_components(self):
        previous=set()
        for f in (.25,.5,.75):
            m=self.make('source_curve',f);rows=set(m['train']);self.assertLessEqual(previous,rows)
            self.assertEqual(m['training_sources'],int(4*f));self.assertEqual(m['test'],self.base['test'])
            for c in set(self.component[m['train']]):self.assertEqual(sum(self.component[m['train']]==c),8)
            previous=rows
    def test_row_curves_keep_sources_and_exact_nested_budgets(self):
        previous=set()
        for f in (.25,.5,.75):
            m=self.make('row_curve',f);rows=set(m['train']);self.assertLessEqual(previous,rows)
            self.assertEqual(len(rows),int(32*f));self.assertEqual(m['training_sources'],4);previous=rows
    def test_human_only_and_shuffle_preserve_fixed_test(self):
        human=self.make('human_only');self.assertEqual(human['teacher_training_rows'],0)
        self.assertEqual(human['train'],list(range(16)));self.assertEqual(human['test'],self.base['test'])
        shuffle=self.make('shuffle');self.assertEqual(shuffle['train'],self.base['train']);self.assertTrue(shuffle['labels_permuted'])


if __name__=='__main__':unittest.main()
