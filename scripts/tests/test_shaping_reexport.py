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
