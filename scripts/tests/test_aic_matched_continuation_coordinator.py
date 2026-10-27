"""Matched-continuation coordinator: registered route, priority wait and exact control. Invented files only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.byconstruction import aic_matched_continuation as owner


class RegisteredRoute(unittest.TestCase):
    def test_models_are_initial_plus_fixed_epochs_of_both_arms(self):
        self.assertEqual(owner.labels(), ["Initial", "MixedAIC16@e7", "MixedAIC16@e59", "AICOnly@e7", "AICOnly@e59"])

    def test_fit_command_is_the_diagnostic_route_without_packing(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"), initial=dict(directory="/invented/Mlic/full_s0"),
                        registered_inputs=dict(aic_table=dict(path="/invented/aic_full_fit.parquet")))
            for arm, population in owner.ARMS.items():
                command = owner.fit_command(plan, run, arm)
                options, rest = {}, command[3:]
                while rest:
                    flag, *rest = rest
                    options[flag] = rest.pop(0) if rest and not rest[0].startswith("--") else True
                self.assertEqual(command[2], str(run / "source/scripts/rev4_featpot/v2_confirm_fit.py"))
                self.assertEqual(options["--diagnostic-population"], population)
                self.assertEqual((float(options["--fine-tune-epochs"]), float(options["--aic-weight"]),
                                  float(options["--fine-tune-lr"])), (60, 16, 0.00001))
                self.assertEqual(options["--init-from"], "/invented/Mlic/full_s0/refit/last.bin")
                self.assertEqual(options["--dest"], str(run / "arms" / arm / "full_s0"))
                self.assertEqual(options["--columns"], "13,14")
                self.assertNotIn("--pack-production", options)
                self.assertIs(options["--strict-admission"], True)
                self.assertIs(options["--train-only"], True)


class PriorityWait(unittest.TestCase):
    def state(self, root, **value):
        (root / "STATE.json").write_text(json.dumps(dict(failures=[], **value)))

    def test_followthrough_terminal_states_and_dead_workers(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            inputs = dict(owner.INPUTS, followthrough=(str(root / "CONFIG.json"), "invented"))
            live = owner.process_snapshot(os.getpid())
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            with patch.object(owner, "INPUTS", inputs):
                self.state(root, stage=owner.FOLLOWTHROUGH_DONE, worker=live)
                self.assertEqual(owner.followthrough_state(), "complete")
                self.state(root, stage="full-panels", worker=live)
                self.assertEqual(owner.followthrough_state(), "running")
                self.state(root, stage="failed", worker=live)
                self.assertEqual(owner.followthrough_state(), "failed")
                self.state(root, stage="full-panels", worker=dict(pid=child.pid, start_ticks=1))
                self.assertEqual(owner.followthrough_state(), "failed")


class ExactControl(unittest.TestCase):
    def test_strip_compare_uses_copies_and_reports_differences(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            tool = root / "strip-tool"
            # Invented stand-in for the native owner: copies --in to --out.
            tool.write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do case "$1" in --in) i=$2;; --out) o=$2;; esac; shift; done\ncp "$i" "$o"\n')
            tool.chmod(0o755)
            first, second = root / "a.bin", root / "b.bin"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            self.assertTrue(owner.nonrepro_identical(tool, first, second, root / "equal")["identical"])
            second.write_bytes(b"different")
            result = owner.nonrepro_identical(tool, first, second, root / "unequal")
            self.assertFalse(result["identical"])
            self.assertEqual((first.read_bytes(), second.read_bytes()), (b"same", b"different"))
            with self.assertRaises(FileExistsError):
                owner.nonrepro_identical(tool, first, second, root / "unequal")


class PlanFreeze(unittest.TestCase):
    def test_existing_run_and_changed_registered_inputs_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "preserve"):
                owner.plan(root, "HEAD", root / "t", root / "m")
            changed = root / "input.json"
            changed.write_text("{}")
            with patch.object(owner, "INPUTS", dict(fold_plan=(str(changed), "0" * 64))):
                with self.assertRaisesRegex(ValueError, "registered input changed"):
                    owner.plan(root / "fresh", "HEAD", root / "t", root / "m")
            self.assertFalse((root / "fresh").exists())


class SourceFoldStudy(unittest.TestCase):
    def test_registered_inventory_is_two_populations_three_seeds_seven_folds(self):
        runs = owner.fold_runs()
        self.assertEqual(len(runs), 42)
        self.assertEqual(len(set(runs)), 42)
        self.assertEqual({r[0] for r in runs}, set(owner.ARMS))
        self.assertEqual({r[1] for r in runs}, {0, 1, 2})
        self.assertEqual({r[2] for r in runs}, set(range(7)))

    def test_fold_fit_binds_its_seed_initial_model_and_fold_table(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"))
            command = owner.fit_command(plan, run, "AICOnly", 2, "/invented/Mlic/full_s2",
                                        "/invented/aic_k5_fit.parquet", run / "arms/AICOnly/s2_k5")
            self.assertEqual(command[command.index("--seed-index") + 1], "2")
            self.assertEqual(command[command.index("--init-from") + 1], "/invented/Mlic/full_s2/refit/last.bin")
            self.assertEqual(command[command.index("--aic-public-train") + 1], "/invented/aic_k5_fit.parquet")
            self.assertEqual(command[command.index("--dest") + 1], str(run / "arms/AICOnly/s2_k5"))
            self.assertEqual(command[command.index("--diagnostic-population") + 1], "aic-only")

    def test_out_of_fold_uses_only_each_models_held_out_sources(self):
        import numpy as np
        folds = np.repeat(np.arange(7), 3)
        values = {f"A|s0|k{k}": np.where(folds == k, 100.0 + k, -1.0) for k in range(7)}
        oof = owner.out_of_fold(values, folds, "A|s0")
        self.assertTrue(np.array_equal(oof, 100.0 + folds))
        values["A|s0|k3"] = np.where(folds == 3, np.nan, -1.0)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            owner.out_of_fold(values, folds, "A|s0")

    def test_gates_queue_end_requires_a_terminal_line(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            log = Path(tmp) / "queue.log"
            with patch.object(owner, "GATES_LOG", log):
                self.assertFalse(owner.gates_finished())
                log.write_text("2026-10-27T09:24:58Z waiting for the matched-continuation diagnostic\n")
                self.assertFalse(owner.gates_finished())
                log.write_text(log.read_text() + "2026-10-27T14:00:00Z GATES-DONE rc=0\n")
                self.assertTrue(owner.gates_finished())
                log.write_text("2026-10-27T14:00:00Z GATES-EXIT rc=1\n")
                self.assertTrue(owner.gates_finished())

    def test_fold_plan_refuses_an_existing_run(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            with self.assertRaisesRegex(ValueError, "preserve"):
                owner.fold_plan(Path(tmp), "HEAD", Path(tmp) / "t", Path(tmp) / "m")


if __name__ == "__main__":
    unittest.main()
