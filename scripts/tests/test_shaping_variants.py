import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_variants import membership
from shaping_variant_inputs import verify_prepared
from shaping_completion import encoded,sha


class VariantTests(unittest.TestCase):
    def test_prepared_reuse_rejects_membership_and_input_drift(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);manifest=root/'VARIANT_MEMBERSHIPS.json';manifest.write_bytes(b'{}')
            source=root/'table';source.write_bytes(b'original')
            view=dict(train=[0,1],test=[2]);cell=dict(stage='human_only')
            admission=root/'ADMISSION.json';admission.write_bytes(encoded(dict(cell=view)))
            path=root/'GROUPS.json'
            receipt=dict(view_identifier='id',permutation='true',permutation_seed=0,
                         view_sha256=sha(manifest),exposed_development_passed_to_trainer=False,
                         outer_results_read=False,file_sha256={str(source):sha(source),str(admission):sha(admission)})
            path.write_bytes(encoded(receipt));verify_prepared(path,root,'id',view,cell)
            with self.assertRaisesRegex(ValueError,'membership differs'):
                verify_prepared(path,root,'id',dict(train=[0,2],test=[1]),cell)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'input drift'):verify_prepared(path,root,'id',view,cell)

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
