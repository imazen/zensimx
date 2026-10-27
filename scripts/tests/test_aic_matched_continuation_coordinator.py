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


if __name__ == "__main__":
    unittest.main()
