import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_probe_trace import f32_layers

class ProbeTraceTests(unittest.TestCase):
    def test_softplus_stable_extremes_and_monotone_values(self):
        values=np.array([[-1000.],[-80.],[-1.],[0.],[1.],[80.],[1000.]])
        model=dict(scaler_scale=[1.],scaler_mean=[0.],layers=[dict(in_dim=1,out_dim=1,weights=[1.],biases=[0.],activation='softplus')])
        actual=f32_layers(values,model)[0][:,0]
        expected=np.logaddexp(0.,values[:,0])
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=0.)
        self.assertTrue((np.diff(actual)>=0).all())
        self.assertEqual(actual[-1],np.float32(1000.))

    def test_unknown_activation_is_not_silently_approximated(self):
        model=dict(scaler_scale=[1.],scaler_mean=[0.],layers=[dict(in_dim=1,out_dim=1,weights=[1.],biases=[0.],activation='unknown')])
        with self.assertRaisesRegex(ValueError,'unsupported scalar'):f32_layers(np.zeros((1,1)),model)
