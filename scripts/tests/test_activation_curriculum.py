"""Independent construction checks, not native bake/runtime qualification."""
import importlib.util
import itertools
import math
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/'rev4_featpot/activation_curriculum.py'
SPEC = importlib.util.spec_from_file_location('activation_curriculum', PATH)
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)


class AnchoredConstruction(unittest.TestCase):
    def test_bias_derivative_includes_anchor(self):
        # A detached anchor incorrectly gives a near-unit derivative here.
        z, b, eps = 0.1, 0.2, 1e-6
        numerical = (REF.anchored('softplus',z,b+eps)-REF.anchored('softplus',z,b-eps))/(2*eps)
        full = REF.activation('softplus',b+z)[1]-REF.activation('softplus',b)[1]
        detached = REF.activation('softplus',b+z)[1]
        self.assertAlmostEqual(numerical, full, places=8)
        self.assertGreater(abs(numerical-detached),0.5)

    def test_dropout_masks_preserve_last_hidden_identity_and_order(self):
        p = 0.15
        weights = [0.2,0.7,1.1]
        expected = 0.
        for bits in itertools.product((0,1),repeat=3):
            mask = [bit/(1-p) for bit in bits]
            probability = math.prod((1-p if bit else p) for bit in bits)
            def score(z):
                return 100-sum(w*m*REF.anchored('linear_tanh',z,b)
                               for w,m,b in zip(weights,mask,(-0.2,0.,0.3)))
            self.assertEqual(score(0),100)
            self.assertGreaterEqual(score(0.1),score(0.2))
            expected += probability*score(0.2)
        full = 100-sum(w*REF.anchored('linear_tanh',0.2,b) for w,b in zip(weights,(-0.2,0.,0.3)))
        self.assertAlmostEqual(expected,full,places=12)

    def test_monotonicity_does_not_require_convexity(self):
        slopes = [REF.activation('linear_tanh',x)[1] for x in (-1.,0.,1.)]
        self.assertTrue(all(s>=1 for s in slopes))
        self.assertGreater(slopes[1],slopes[0])
        self.assertLess(slopes[2],slopes[1])

    def test_stable_reference_extremes(self):
        for name in ('relu','smooth_hinge','leaky','softplus','gaussian','rational','linear_tanh'):
            for b in (-1000.,-1.,-0.,0.,1.,1000.):
                self.assertEqual(REF.anchored(name,0.,b),0.)
                values = [REF.anchored(name,z,b) for z in (0.,1e-12,1e-8,0.1,1.,1000.)]
                self.assertTrue(all(math.isfinite(v) for v in values))
                self.assertTrue(all(a<=b for a,b in zip(values,values[1:])))

    def test_anchoring_does_not_repair_silu_or_gelu_order(self):
        for name in ('silu_negative_control','gelu_negative_control'):
            self.assertEqual(REF.anchored(name,0.,-2.),0.)
            self.assertLess(REF.anchored(name,0.1,-2.),0.)


if __name__ == '__main__':
    unittest.main()
