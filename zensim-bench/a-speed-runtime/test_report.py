"""Admission controls with invented timings; these are not benchmark results."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("a_speed_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportAdmission(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="a-speed-report-test-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "fixtures").mkdir()
        (self.root / "results").mkdir()
        (self.root / "COMPLETE").touch()
        fixtures = []
        for size in report.SIZES:
            fixtures.extend({"id": f"{size}_{i}", "size": size, "width": size, "height": size} for i in range(20))
            for threads in (1, 16):
                groups = []
                for i in range(20):
                    samples = [{"gate_clean": True, "iterations": 1, "execution_order": [j % 3, (j + 1) % 3, (j + 2) % 3],
                                "elapsed_ns": [int(100000 + size * size * k / threads + j) for k in (1, 2, 3)]} for j in range(32)]
                    groups.append({"group_name": f"{size}/{size * size}/{size}_{i}", "benchmarks": [{"name": a} for a in report.ARMS], "samples": samples})
                p = self.root / f"results/{threads}t_{size}_latency.json"
                p.write_text(json.dumps({"unreliable": False, "comparisons": groups}))
                p.with_suffix(".log").write_text("invented admission fixture\n")
                for arm in report.ARMS:
                    stem = self.root / f"results/{threads}t_{size}_rss-{arm}"
                    stem.with_suffix(".log").write_text("run-heavy: done rc=0 1s | peak-RSS 0.01GiB | min-avail 20000MiB | peak-load 0.10\n")
                    stem.with_suffix(".time.txt").write_text("Maximum resident set size (kbytes): 10000\n")
        (self.root / "fixtures/pairs.json").write_text(json.dumps(fixtures))
        self.path = self.root / "results/1t_64_latency.json"
        self.original = json.loads(self.path.read_text())

    def write(self, run):
        self.path.write_text(json.dumps(run))

    def test_complete_matrix_and_units(self):
        result = report.summarize(self.root)
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(len(result["fits"]), 6)
        self.assertTrue(all(r["samples_per_arm"] == 640 for r in result["rows"]))
        self.assertEqual(result["rows"][0]["zensim_a"]["rss_kib"], 10000)

    def test_foreign_compiler_refuses_even_with_clean_round_flags(self):
        self.path.with_suffix(".log").write_text('[zenbench gate] foreign pid=123 name="rustc" cpu_pct=100\n')
        with self.assertRaisesRegex(AssertionError, "foreign build/training"):
            report.summarize(self.root)

    def test_duplicate_image_refuses(self):
        run = copy.deepcopy(self.original)
        run["comparisons"][1]["group_name"] = run["comparisons"][0]["group_name"]
        self.write(run)
        with self.assertRaisesRegex(AssertionError, "duplicate image"):
            report.summarize(self.root)

    def test_batched_calls_refuse(self):
        run = copy.deepcopy(self.original)
        run["comparisons"][0]["samples"][0]["iterations"] = 2
        self.write(run)
        with self.assertRaises(AssertionError):
            report.summarize(self.root)

    def test_insufficient_clean_pairs_refuse(self):
        run = copy.deepcopy(self.original)
        run["comparisons"][0]["samples"][0]["gate_clean"] = False
        self.write(run)
        with self.assertRaises(AssertionError):
            report.summarize(self.root)


if __name__ == "__main__":
    unittest.main()
