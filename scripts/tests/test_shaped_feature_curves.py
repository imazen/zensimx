"""Check normalization, inactive terms, and physical-domain scale handling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaped_linear_report import feature_curve_stability

class FeatureCurveTests(unittest.TestCase):
    def test_strength_and_shape_are_separate(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            root=Path(temporary)
            for outer in range(3):
                for label,gain in [('full',1),('f0.75-s1',2),('f0.75-s2',4)]:
                    path=root/'fits'/f's3-source-positive_shaped_products-o{outer}-{label}'
                    (path/'lambda-0.0001').mkdir(parents=True)
                    # First term is y=gain*x: normalized shape identical. Second
                    # is inactive; third changes from x to max(x-1,0).
                    spec=dict(terms=[[0,None],[1,None],[2,None]],scales=[2,1,1],
                              knots=[[.5,1],[1,2],[1,2]],basis_scales=[1]*9)
                    (path/'SPEC.json').write_text(json.dumps(spec));(path/'TASK.json').write_text('{}')
                    last=[1,1,1] if label=='full' else [0,1,1]
                    np.savez(path/'lambda-0.0001/predictions.npz',coefficients=np.array([2*gain]*3+[0]*3+last))
            feature_curve_stability(root,root/'result')
            result=json.loads((root/'result/FEATURE_CURVE_STABILITY.json').read_text())
            first,zero,changing=result['terms']
            self.assertEqual(first['normalized_curve_max_spread'],0)
            self.assertEqual(first['endpoint_strength_ratio'],4)
            self.assertEqual(first['endpoint_strength'][:3],[2,4,8])
            self.assertEqual(zero['active_fits'],0)
            self.assertIsNone(zero['normalized_curve_max_spread'])
            self.assertEqual(changing['normalized_curve_max_spread'],.5)
            self.assertEqual(result['active_in_all'],2)
            with self.assertRaises(FileExistsError):feature_curve_stability(root,root/'result')

if __name__=='__main__':unittest.main()
