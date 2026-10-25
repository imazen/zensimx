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
    def test_train_fold_mos_uses_signed_midranks_and_reports_undefined_refs(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
        from byc_cells import dataset_train_statistic
        pair = dataset_train_statistic(["r","r"],[0.,1.],[1.,2.],False)
        self.assertEqual(pair["value"],1.)
        tied = dataset_train_statistic(["r","r","r","s","s"],
            [1.,1.,2.,0.,1.],[3.,3.,1.,1.,1.],False)
        self.assertEqual(tied["value"],-1.)
        self.assertEqual(tied["defined_groups"],1)
        self.assertEqual(tied["excluded_groups"],1)

    def test_mix_eval_refuses_changed_selection_before_eval_access(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"TRAIN_SELECTION.json").write_text('{"finalists":[]}')
            (p/"FINALISTS.json").write_text(json.dumps(dict(schema="dataset-mix-finalists-v1",
                selection_uses_EVAL=False, TRAIN_selection_sha256="0"*64)))
            with self.assertRaisesRegex(ValueError,"frozen TRAIN-only"):
                evaluation.cmd_mix_eval(SimpleNamespace(study=str(p),source=str(p/"missing-eval"),predict="unused"))
            self.assertFalse((p/"eval").exists())

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

    def _eval_dir(self, d, keys, preds, judge_rows=()):
        import pyarrow as pa, pyarrow.parquet as pq
        p = Path(d)
        pq.write_table(pa.table({"ref_basename": keys}), p/"features.parquet")
        for name, v in preds.items():
            (p/f"pred_{name}.tsv").write_text("ref_basename\tpred\n" + "".join(f"{k}\t{x!r}\n" for k,x in zip(keys,v)))
        (p/"judge_pairs.tsv").write_text("ref_path\tdist_path\tpair_key\n" + "".join(f"{r}\t{t}\t{k}\n" for r,t,k,_ in judge_rows))
        if judge_rows:
            (p/"judge_ssim2.tsv").write_text("ref_path\tdist_path\tpair_key\tssim2\n"
                                              + "".join(f"{r}\t{t}\t{k}\t{s}\n" for r,t,k,s in judge_rows))
        return p

    def test_judge_scores_reach_every_row_that_shares_a_distorted_image(self):
        shared = [("r.png","d.png","k1",70.0), ("r.png","d.png","k2",70.0), ("r.png","e.png","k3",50.0)]
        with tempfile.TemporaryDirectory() as d:
            p = self._eval_dir(d, ["k1","k2","k3"], {"M":[1.,2.,3.]}, shared)
            self.assertEqual(evaluation.load_scores(p)["ssim2"], {"k1":70.0,"k2":70.0,"k3":50.0})
            (p/"judge_ssim2.tsv").write_text("ref_path\tdist_path\tpair_key\tssim2\nr.png\te.png\tk1\t70\n")
            with self.assertRaisesRegex(ValueError,"does not match"):
                evaluation.load_scores(p)

    def test_paired_comparison_runs_the_panel_owner_over_references(self):
        import json
        rows, keys, a, b = [], [], [], []
        for ref in range(8):
            for level in range(4):
                k = f"k{ref}_{level}"
                rows.append(dict(collection="c", row=len(rows), kind="mos", group=f"ref{ref}", a=k, b=None,
                                 target=float(level)))
                keys.append(k); a.append(float(level)); b.append(float(level if level % 2 == 0 or ref % 3 else 3 - level))
        with tempfile.TemporaryDirectory() as d:
            p = self._eval_dir(d, keys, {"A":a, "B":b})
            (p/"eval_rows.json").write_text(json.dumps(rows))
            evaluation.cmd_paired(SimpleNamespace(out=str(p), dest=str(p/"paired"), a="A", vs=["B"], resamples=64))
            report = json.loads((p/"paired/report/report.json").read_text())
            (cmp,) = report["comparisons"]
            self.assertEqual(cmp["report"]["state"], "measured")
            m = cmp["report"]["measurements"]
            self.assertEqual(m["bootstrap"]["references"], 8)
            self.assertGreater(m["delta"][m["delta_order"].index("srocc")]["estimate"], 0.0)
            self.assertIn("| c | A − B |", (p/"paired/paired.md").read_text())

    def test_pair_agreement_is_the_soft_formula_through_the_owner(self):
        rows, s = [], {}
        for i, (pa, da, db) in enumerate([(1.0, 3, 1), (0.0, 3, 1), (0.3, 3, 1), (0.5, 2, 5), (0.8, 4, 4),
                                          (0.9, 1, 2), (0.6, 7, 3), (0.2, 2, 6)]):
            s[f"a{i}"], s[f"b{i}"] = float(da), float(db)
            rows.append(dict(group=f"ref{i % 3}", a=f"a{i}", b=f"b{i}", target=pa))
        def soft(r):
            d = s[r["a"]] - s[r["b"]]
            return r["target"] if d > 0 else (1 - r["target"]) if d < 0 else 0.5
        want = sum(map(soft, rows)) / len(rows)
        acc, lo, hi, n = evaluation.pair_agreement(rows, s, resamples=256)
        self.assertEqual(n, 8)
        self.assertAlmostEqual(acc, want, places=9)
        self.assertLessEqual(lo, acc)
        self.assertLessEqual(acc, hi)

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
