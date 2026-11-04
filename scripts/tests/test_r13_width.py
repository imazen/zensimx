"""Capacity classification boundaries and proportional fitter argument checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('r13_width', Path(__file__).parents[1]/'byconstruction/r13_width.py')
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)


class Capacity(unittest.TestCase):
    def test_structural_dead_and_residual_are_disjoint_and_bias_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'head.json'
            # A zero column with positive bias is active; a zero output with
            # nonzero encoder is residual; equality at the cutoff is live.
            p.write_text(json.dumps(dict(layers=[dict(in_dim=2,out_dim=5,
                weights=[0,0,1,1,1,0,0,0,0,0],biases=[-1,1,0,0,0]),
                dict(weights=[1,0.0001,0,0.001,1])])) )
            d = OWNER.quick_capacity(p)
            self.assertEqual((d['dead'],d['residual'],d['live']),(1,2,2))
            self.assertEqual(d['dead_units'],[0])
            self.assertEqual(d['residual_units'],[1,2])


if __name__ == '__main__': unittest.main()
