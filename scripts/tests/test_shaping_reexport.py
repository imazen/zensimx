from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_reexport import admit_family

class ReexportAdmissionTests(unittest.TestCase):
    def test_unsigned_checkpoint_is_preserved_exactly_without_native_repair(self):
        import tempfile
        from unittest.mock import patch
        from shaping_reexport import export_pairs,convert_export
        model=dict(layers=[dict(in_dim=2,out_dim=2,activation='relu',weights=[1.,0.,0.,2.],biases=[-1.,-2.]),
                           dict(in_dim=2,out_dim=1,weights=[1.,1.])])
        self.assertEqual(export_pairs(model),[])
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);original=root/'original';original.write_bytes(bytes(range(256)))
            with patch('shaping_reexport.subprocess.run') as native:
                kind=convert_export(original,root/'bake',root/'pairs',[],root/'native',root/'log')
            native.assert_not_called();self.assertEqual(kind,'unchanged-nonnegative-first-layer')
            self.assertEqual((root/'bake').read_bytes(),original.read_bytes())
            with self.assertRaisesRegex(ValueError,'preserve existing'):
                convert_export(original,root/'bake',root/'pairs',[],root/'native',root/'log')
        for value in (float('nan'),float('inf')):
            model['layers'][0]['weights'][0]=value
            with self.assertRaisesRegex(ValueError,'finite adjacent'):export_pairs(model)

    def test_signed_pairs_still_require_unambiguous_tied_native_candidates(self):
        from shaping_reexport import export_pairs
        model=dict(layers=[dict(in_dim=1,out_dim=2,activation='relu',weights=[2.,-1.],biases=[0.,-1.]),
                           dict(in_dim=2,out_dim=1,weights=[1.,1.])])
        self.assertEqual(export_pairs(model),[[0,1]])
        model['layers'][1]['weights'][1]=2.
        with self.assertRaisesRegex(ValueError,'ambiguous or unsupported'):export_pairs(model)

    def test_analytic_repair_requires_explicit_diagnostic_scope(self):
        admit_family('relu')
        with self.assertRaises(ValueError):admit_family('softplus')
        admit_family('softplus',True)
        for family in ('silu_monotone','silu_diagnostic','product_joint_identity','extra_trees'):
            with self.assertRaises(ValueError):admit_family(family,True)

    def test_comparison_refuses_reordered_or_nonfinite_rows(self):
        import tempfile
        from shaping_reexport import prediction_values
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            p=Path(d)/'pred.tsv';p.write_text('row_idx\tpred\n0\t1.5\n1\t2.0\n')
            self.assertEqual(prediction_values(p).tolist(),[1.5,2.0])
            for text in ('row_idx\tpred\n1\t1.5\n0\t2.0\n','row_idx\tpred\n0\tnan\n','row_idx\tpred\n'):
                p.write_text(text)
                with self.assertRaises(ValueError):prediction_values(p)

    def test_later_softplus_requires_inner_representation_admission(self):
        from shaping_reexport import admit_representation
        context={'admission':{'evaluation_representations':{'softplus':{'representation':'original'}}}}
        with self.assertRaises(ValueError):admit_representation('softplus',context,True)
        context['admission']['evaluation_representations']['softplus']['representation']='research-relu-max-pairs-v1'
        admit_representation('softplus',context)
        with self.assertRaises(ValueError):admit_representation('silu_monotone',context,True)


class RegisteredPairLayoutTests(unittest.TestCase):
    def test_declared_adjacent_layout_resolves_dead_ties_without_relaxing_generic_proposal(self):
        from shaping_reexport import export_pairs,registered_export_pairs
        model=dict(layers=[dict(in_dim=1,out_dim=4,activation='relu',weights=[2.,-1.,2.,-1.],biases=[0.,-1.,0.,-1.]),
                           dict(in_dim=4,out_dim=1,weights=[0.,0.,0.,0.])])
        command=['trainer','--hidden','4','--n-hidden-layers','2','--max-pairs','2',
                 '--nonneg-distance','--monotone-cbc','--nonneg-thresholds']
        with self.assertRaisesRegex(ValueError,'ambiguous'):export_pairs(model)
        pairs=registered_export_pairs(model,command)
        self.assertEqual({v for u,v in pairs},{1,3})
        self.assertEqual(len({u for u,v in pairs}),2)
        model['layers'][1]['weights'][1]=1.
        with self.assertRaisesRegex(ValueError,'invariants'):registered_export_pairs(model,command)
