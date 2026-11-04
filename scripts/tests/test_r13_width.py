"""Capacity classification boundaries and proportional fitter argument checks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('r13_width', Path(__file__).parents[1]/'byconstruction/r13_width.py')
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)
STATUS_SPEC=importlib.util.spec_from_file_location('r13_width_status',Path(__file__).parents[1]/'byconstruction/r13_width_status.py')
STATUS=importlib.util.module_from_spec(STATUS_SPEC)
with mock.patch.dict(sys.modules,{'r13_width':OWNER}):STATUS_SPEC.loader.exec_module(STATUS)


class Capacity(unittest.TestCase):
    def test_audit_binds_the_frozen_native_certifier(self):
        with mock.patch.dict('os.environ',{'VIZ_CERTIFIER':'/wrong/certifier'}):
            self.assertEqual(OWNER.env()['VIZ_CERTIFIER'],str(OWNER.O/'bin/bake_dial_refit'))

    def test_w4_full_recipe_parses_only_with_its_exact_amendment(self):
        import v2_common as common
        spec='sel:45acffa603e8@h32:H4:cv16:cf98:fx2:mono:thr:mp1'
        registration='397d67be879339e7014d3b29c2a99d3fe51df6b2'
        for unbound in ('r0@h32:H4',spec):
            with self.assertRaises(ValueError):common.recipe_of(unbound)
        parsed=common.recipe_of(spec,r13_w4_registration=registration)
        expected=common.recipe_of(spec.replace(':H4:',':H16:'));expected['hidden']=4
        self.assertEqual(parsed,expected)
        self.assertEqual((parsed['hidden'],parsed['max_pairs']),(4,1))
        for wrong in (spec.replace(':mp1',':mp2'),'r0@h32:H4'):
            with self.assertRaises(ValueError):common.recipe_of(wrong,r13_w4_registration=registration)
        with self.assertRaises(ValueError):common.recipe_of(spec,r13_w4_registration='25e772922fd5a40d576e12af47778ce81bed5e08')

    def test_failed_receipt_cannot_be_hidden_by_a_later_lock_holder(self):
        self.assertEqual(STATUS.execution_state({'rc':1},[{'pid':123}],False),'failed')
        self.assertEqual(STATUS.execution_state({'rc':0},[],False),'complete')
        self.assertEqual(STATUS.execution_state(None,[],False),'process-exited-without-receipt')
        self.assertEqual(STATUS.execution_state(None,[{'pid':123}],False),'waiting-for-shared-lock')
        self.assertEqual(STATUS.execution_state(None,[{'pid':123}],True),'running')

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
