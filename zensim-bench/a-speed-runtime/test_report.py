"""Admission controls with invented timings; these are not benchmark results."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
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
        (self.root / "measurement-host.txt").write_text("i270\n")
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
                    score_lines = "".join(f"{size}_{i}\t{arm}\t{i + 1}.00000000000000000\n" for i in range(20))
                    stem.with_suffix(".log").write_text(score_lines + "run-heavy: done rc=0 1s | peak-RSS 0.01GiB | min-avail 20000MiB | peak-load 0.10\n")
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

    def test_host_mismatch_refuses(self):
        (self.root / "measurement-host.txt").write_text("another-host\n")
        with self.assertRaisesRegex(AssertionError, "measurement host"):
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

    def test_incomplete_rss_image_coverage_refuses(self):
        path = self.root / "results/1t_64_rss-margarine.log"
        path.write_text(path.read_text().split("\n", 1)[1])
        with self.assertRaisesRegex(AssertionError, "RSS image coverage"):
            report.summarize(self.root)

    def test_insufficient_clean_pairs_refuse(self):
        run = copy.deepcopy(self.original)
        run["comparisons"][0]["samples"][0]["gate_clean"] = False
        self.write(run)
        with self.assertRaises(AssertionError):
            report.summarize(self.root)

    def test_command_startup_requires_single_calls(self):
        for threads in (1, 16):
            run = {"unreliable": False, "comparisons": [{
                "group_name": "margarine/command-help-startup",
                "benchmarks": [{"name": "margarine_help"}],
                "samples": [{"gate_clean": True, "iterations": 1,
                             "execution_order": [0], "elapsed_ns": [1000000 + i]}
                            for i in range(64)]}]}
            path = self.root / f"startup-{threads}t.json"
            path.write_text(json.dumps(run))
            path.with_suffix(".log").write_text("invented startup fixture\n")
        self.assertEqual(report.command_startup(self.root)[0]["calls"], 32)
        run["comparisons"][0]["samples"][0]["iterations"] = 2
        path.write_text(json.dumps(run))
        with self.assertRaises(AssertionError):
            report.command_startup(self.root)


class ParallelGateAdmission(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="a-parallel-report-test-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "fixtures").mkdir()
        fixtures = [{"id": f"{size}_{i}", "size": size, "width": size, "height": size}
                    for size in (*report.GATE_SIZES, 4096) for i in range(20)]
        (self.root / "fixtures/pairs.json").write_text(json.dumps(fixtures))
        for stage in ("before", "after"):
            base = self.root / stage
            (base / "results").mkdir(parents=True)
            (base / "measurement-host.txt").write_text("i270\n")
            for threads in (16, 4):
                for size in (*report.GATE_SIZES, 4096) if threads == 16 else report.GATE_SIZES:
                    # Invented control: the 4T crossover is deliberately higher.
                    crossover = 192 if threads == 4 else 160
                    groups = [{"group_name": f"{size}/{size*size}/{size}_{i}",
                               "benchmarks": [{"name": a} for a in ("serial", "parallel")],
                               "samples": [{"gate_clean": True, "iterations": 1,
                                            "execution_order": [j % 2, 1 - j % 2],
                                            "elapsed_ns": [1000000+j, (500000 if size >= crossover else 2000000)+j]}
                                           for j in range(32)]} for i in range(20)]
                    path = base / f"results/{threads}t_{size}_latency.json"
                    path.write_text(json.dumps({"unreliable": False, "comparisons": groups}))
                    runner = "run-heavy: done rc=0 1s | peak-RSS 0.01GiB | min-avail 20000MiB | peak-load 0.10\n"
                    path.with_suffix(".log").write_text(runner)
                    (base / f"results/{threads}t_{size}_scores.log").write_text(
                        "".join(f"{size}_{i}\t1.00000000000000000\t1.00000000000000000\t3ff0000000000000\t3ff0000000000000\n" for i in range(20)) + runner)

    def test_higher_pool_crossover_and_complete_scores(self):
        result = report.parallel_gate(self.root, "after")
        self.assertEqual(result["crossovers"], {"16": 160, "4": 192})
        self.assertEqual(result["parallel_min_pixels"], 192**2)
        self.assertEqual(result["score_comparisons"], 1000)

    def test_score_bit_movement_refuses(self):
        path = self.root / "after/results/4t_64_scores.log"
        path.write_text(path.read_text().replace("3ff0000000000000\t3ff0000000000000", "3ff0000000000000\t3ff0000000000001", 1))
        with self.assertRaisesRegex(AssertionError, "score moved"):
            report.parallel_gate(self.root, "after")

    def test_foreign_compiler_refuses(self):
        path = self.root / "before/results/16t_64_latency.log"
        with path.open("a") as stream:
            stream.write('[zenbench gate] foreign pid=123 name="rustc" cpu_pct=100\n')
        with self.assertRaisesRegex(AssertionError, "foreign build/training"):
            report.parallel_gate(self.root, "before")

    def test_shared_clean_round_coverage_refuses(self):
        path = self.root / "before/results/16t_64_latency.json"
        run = json.loads(path.read_text())
        run["comparisons"][0]["samples"][0]["gate_clean"] = False
        path.write_text(json.dumps(run))
        with self.assertRaisesRegex(AssertionError, "insufficient clean rounds"):
            report.parallel_gate(self.root, "before")


class CollectorCompletion(unittest.TestCase):
    def test_targeted_collection_cannot_complete_an_incomplete_matrix(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        for gate in (False, True):
            with self.subTest(parallel_gate=gate), tempfile.TemporaryDirectory(dir=scratch) as folder:
                root = Path(folder)
                results = root / "results"
                results.mkdir()
                threads = (16, 4) if gate else (1, 16)
                sizes = (*report.GATE_SIZES, 4096) if gate else report.SIZES
                for t in threads:
                    for size in sizes:
                        if gate and t == 4 and size == 4096:
                            continue
                        (results / f"{t}t_{size}_latency.json").write_text("invented completion fixture")
                        arms = ("scores.log",) if gate else tuple(f"rss-{a}.time.txt" for a in report.ARMS)
                        for arm in arms:
                            (results / f"{t}t_{size}_{arm}").write_text("invented completion fixture")
                missing = results / "16t_4096_latency.json"
                missing.unlink()
                env = dict(os.environ, BENCH_CPUS="0-15", BENCH_THREADS=" ", BENCH_SIZES=" ")
                if gate:
                    env["A_SPEED_PARALLEL_GATE"] = "1"
                else:
                    env.pop("A_SPEED_PARALLEL_GATE", None)
                command = ["bash", str(Path(__file__).with_name("run.sh")), str(root), "/usr/bin/true"]
                subprocess.run(command, env=env, check=True, capture_output=True)
                self.assertFalse((root / "COMPLETE").exists())
                missing.write_text("invented completion fixture")
                subprocess.run(command, env=env, check=True, capture_output=True)
                self.assertTrue((root / "COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
