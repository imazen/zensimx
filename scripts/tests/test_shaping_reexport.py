from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_reexport import admit_family

class ReexportAdmissionTests(unittest.TestCase):
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
