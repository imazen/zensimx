"""Synthetic ties exercise the reporting owner's actual Rust statistic."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("dataset_eval", Path(__file__).resolve().parents[1]/"byconstruction/dataset_eval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class DatasetEval(unittest.TestCase):
    def test_bapps_rebind_preserves_original_and_refuses_training_paths(self):
        for split in ("val","train"):
            with self.subTest(split=split),tempfile.TemporaryDirectory() as d:
                p = Path(d)
                original = f"ref_path\tdist_path\tpair_key\n/original/{split}/cnn/ref/1.png\t/original/{split}/cnn/p0/1.png\tk\n"
                pairs = p/"judge_pairs.tsv"
                pairs.write_text(original)
                args = SimpleNamespace(out=str(p),scratch=str(p/"scratch"))
                if split == "train":
                    with self.assertRaisesRegex(ValueError,"VAL"):
                        evaluation.cmd_bapps_keys(args)
                    self.assertEqual(pairs.read_text(),original)
                else:
                    evaluation.cmd_bapps_keys(args)
                    self.assertEqual((p/"judge_pairs.original-paths.bak.tsv").read_text(),original)
                    self.assertIn(str(p/"scratch/val/cnn/ref/1.png"),pairs.read_text())
                    with self.assertRaisesRegex(ValueError,"already exists"):
                        evaluation.cmd_bapps_keys(args)

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
