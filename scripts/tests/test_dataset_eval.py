"""Synthetic ties exercise the reporting owner's actual Rust statistic."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("dataset_eval", Path(__file__).resolve().parents[1]/"byconstruction/dataset_eval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class DatasetEval(unittest.TestCase):
    def test_mos_midrank_is_tie_aware_and_order_independent(self):
        x, y = [1.,1.,2.,3.], [1.,2.,2.,4.]
        self.assertAlmostEqual(evaluation.spearman(x,y), 5/6, places=14)
        self.assertAlmostEqual(evaluation.spearman(x[::-1],y[::-1]), 5/6, places=14)

    def test_mos_interval_keeps_sources_and_native_direction(self):
        rows = [dict(a=str(i),group=str(i//2),target=v) for i,v in enumerate([1.,2.,2.,4.])]
        result,n = evaluation.mos_interval(rows,dict(zip(map(str,range(4)),[1.,1.,2.,3.])),resamples=128)
        self.assertEqual(n,4)
        self.assertAlmostEqual(result["signed"]["estimate"],5/6,places=14)
        self.assertEqual(result["srocc_ci_method"]["units"],2)
        self.assertEqual(result["srocc_ci_method"]["owner"],"zenstats::resample::spearman_interval")


if __name__ == "__main__":
    unittest.main()
