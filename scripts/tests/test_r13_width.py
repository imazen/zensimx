"""Capacity classification boundaries and proportional fitter argument checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('r13_width', Path(__file__).parents[1]/'byconstruction/r13_width.py')
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)


class Capacity(unittest.TestCase):
    def test_owner_cut_keeps_only_seed_zero_and_appends_w4_after_w16(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(OWNER,'O',Path(directory)):
            self.assertEqual(OWNER.required_fits(),15)
            cut=dict(schema='r13-owner-cost-cut-v2',seeds=[0],not_run_seeds=[1,2],
                     widths=[4,16,32,64,256,512],additional_arm='W4-s0')
            p=Path(directory)/'OWNER_COST_CUT.json';p.write_text(json.dumps(cut))
            self.assertEqual(OWNER.registered_seeds(),(0,))
            self.assertEqual(OWNER.registered_widths(),(32,256,64,512,16,4))
            self.assertEqual(OWNER.required_fits(),6)
            cut['seeds']=[0,1];p.write_text(json.dumps(cut))
            with self.assertRaises(ValueError):OWNER.registered_seeds()

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
