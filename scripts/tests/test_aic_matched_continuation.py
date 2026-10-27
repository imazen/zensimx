"""Registered matched-continuation diagnostic: authority derivation and CLI refusals.

The diagnostic (benchmarks/byconstruction_2026-10-22.md, 2026-10-27) is separate
from the frozen 0/4/8-epoch continuation contract. Invented metadata only.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "rev4_featpot"))
import subjective_training_leg as leg


class DiagnosticAuthority(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.training = dict(name="subjective_rank_invented", path=str(self.root / "t.parquet"), train_weight=0.5,
                             val_weight=0.0, mode="withinref,rank", subjective=True)
        self.human = dict(name="human", path=str(self.root / "h.parquet"), train_weight=32.0, val_weight=0.0,
                          mode="withinref,rank")
        self.development = dict(name="human_development", path=str(self.root / "d.parquet"), train_weight=0.0,
                                val_weight=1.0, mode="withinref,rank")
        base = dict(schema="subjective-training-authority-v2", state="approved", artifact_visibility="private",
                    selection="fixed-final-epoch",
                    cells={"full-s0": dict(route="full", width=1853, groups=[self.human, self.development, self.training]),
                           "full-s1": dict(route="full", width=1853, groups=[])})
        self.base_path = self.root / "dataset-authority.json"
        self.base_path.write_text(json.dumps(base, sort_keys=True, separators=(",", ":")) + "\n")
        self.checkpoint = self.root / "last.bin"
        self.checkpoint.write_bytes(b"invented checkpoint")
        self.binding = (self.base_path, leg.sha(self.base_path), "full-s0", [0, 1])
        self.aic = [("aic_public_train", self.root / "aic.parquet", 16.0, 0, "rank")]

    def derive(self, population):
        binding, retained = leg.diagnostic_authority(self.binding, self.checkpoint, population, 0.00001)
        return binding, retained, leg.document(binding[0], binding[1])

    def test_mixed_arm_keeps_every_base_group(self):
        binding, retained, a = self.derive("mixed-aic16")
        self.assertEqual(a["schema"], "subjective-training-authority-v4")
        self.assertEqual(retained, ["human", "human_development", "subjective_rank_invented"])
        c = a["cells"]["full-s0"]["continuation"]
        self.assertEqual((c["epochs"], c["aic_weight"], c["dump_checkpoints_every"], c["excluded_groups"]),
                         (60, 16.0, 1, []))
        self.assertEqual(c["retained_checkpoint_epochs"], [7, 59])
        self.assertFalse(leg.diagnostic_binding(a, "full-s0", self.aic))

    def test_aic_only_arm_excludes_exactly_the_training_groups(self):
        binding, retained, a = self.derive("aic-only")
        self.assertEqual(retained, ["human_development"])
        self.assertEqual([r["name"] for r in a["cells"]["full-s0"]["continuation"]["excluded_groups"]],
                         ["human", "subjective_rank_invented"])
        self.assertTrue(leg.diagnostic_binding(a, "full-s0", self.aic))

    def test_changed_protocol_partition_base_or_leg_refuses(self):
        _, _, a = self.derive("aic-only")
        for mutate, needle in (
                (lambda v: v["cells"]["full-s0"]["continuation"].update(epochs=8), "protocol"),
                (lambda v: v["cells"]["full-s0"]["continuation"].update(dump_checkpoints_every=59), "protocol"),
                (lambda v: v["cells"]["full-s0"]["continuation"].update(population="aic"), "population"),
                (lambda v: v["cells"]["full-s0"].update(groups=[self.human, self.development]), "partition"),
                (lambda v: v["cells"]["full-s0"].update(width=1825), "base binding"),
                (lambda v: v["cells"]["full-s1"].update(route="d1"), "another cell"),
                (lambda v: v.update(state="draft"), "base field")):
            changed = json.loads(json.dumps(a))
            mutate(changed)
            with self.assertRaisesRegex(ValueError, needle):
                leg.diagnostic_binding(changed, "full-s0", self.aic)
        for legs in ([], [("aic_public_train", self.root / "aic.parquet", 8.0, 0, "rank")], self.aic * 2):
            with self.assertRaisesRegex(ValueError, "weight 16"):
                leg.diagnostic_binding(a, "full-s0", legs)
        self.base_path.write_text(self.base_path.read_text().replace("approved", "approved "))
        with self.assertRaisesRegex(ValueError, "pin"):
            leg.diagnostic_binding(a, "full-s0", self.aic)

    def test_derivation_preserves_prior_authority_and_requires_registered_lr(self):
        with self.assertRaisesRegex(ValueError, "LR"):
            leg.diagnostic_authority(self.binding, self.checkpoint, "mixed-aic16", 0.001)
        self.derive("mixed-aic16")
        with self.assertRaisesRegex(ValueError, "preserve"):
            self.derive("aic-only")
        with self.assertRaisesRegex(ValueError, "population"):
            leg.diagnostic_partition([self.human], "aic")

    def test_frontier_populations_bind_only_their_own_weight_and_checkpoints(self):
        for population, weight in (("mixed-aic64", 64.0), ("mixed-aic256", 256.0), ("mixed-aic1024", 1024.0)):
            with self.subTest(population=population):
                (self.root / "diagnostic-authority.json").unlink(missing_ok=True)
                binding, retained, a = self.derive(population)
                c = a["cells"]["full-s0"]["continuation"]
                self.assertEqual((c["protocol"], c["aic_weight"], c["retained_checkpoint_epochs"], c["excluded_groups"]),
                                 ("aic-share-frontier-v1", weight, [7, 15, 29, 59], []))
                self.assertEqual(retained, ["human", "human_development", "subjective_rank_invented"])
                legs = [("aic_public_train", self.root / "aic.parquet", weight, 0, "rank")]
                self.assertFalse(leg.diagnostic_binding(a, "full-s0", legs))
                with self.assertRaisesRegex(ValueError, f"weight {weight:g}"):
                    leg.diagnostic_binding(a, "full-s0", self.aic)
                for mutate in (lambda w: w.update(protocol="aic-matched-continuation-diagnostic-v1"),
                               lambda w: w.update(aic_weight=16.0),
                               lambda w: w.update(retained_checkpoint_epochs=[7, 59])):
                    changed = json.loads(json.dumps(a))
                    mutate(changed["cells"]["full-s0"]["continuation"])
                    with self.assertRaisesRegex(ValueError, "protocol"):
                        leg.diagnostic_binding(changed, "full-s0", legs)
        # The matched diagnostic's own table is unchanged.
        self.assertEqual(leg.diagnostic_protocol("mixed-aic16"), ("aic-matched-continuation-diagnostic-v1", 16.0, [7, 59]))
        self.assertEqual(leg.diagnostic_protocol("aic-only"), ("aic-matched-continuation-diagnostic-v1", 16.0, [7, 59]))

    def test_frozen_continuation_budgets_are_unchanged(self):
        for epochs in (60, 12):
            with self.assertRaisesRegex(ValueError, "unregistered"):
                leg.continuation_authority(self.binding, self.checkpoint, epochs, 0.00001)


class DiagnosticCommandLine(unittest.TestCase):
    def refuse(self, *extra):
        result = subprocess.run([sys.executable, "-B", str(SCRIPTS / "rev4_featpot/v2_confirm_fit.py"), "--spec", "x",
                                 "--head", "N", "--seed-index", "0", *extra], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 2, result.stderr)
        return result.stderr

    def test_diagnostic_budgets_and_output_stage_are_registered(self):
        base = ["--init-from", "/nonexistent", "--aic-public-train", "x", "--strict-admission", "--train-only",
                "--fine-tune-lr", "0.00001"]
        self.assertIn("registered continuation", self.refuse("--diagnostic-population", "aic-only"))
        for extra in (["--fine-tune-epochs", "60", "--aic-weight", "16"],  # no diagnostic population
                      ["--diagnostic-population", "aic-only", "--fine-tune-epochs", "8", "--aic-weight", "16"],
                      ["--diagnostic-population", "aic-only", "--fine-tune-epochs", "60", "--aic-weight", "8"],
                      ["--diagnostic-population", "mixed-aic16", "--fine-tune-epochs", "60", "--aic-weight", "16",
                       "--pack-production"],
                      ["--diagnostic-population", "mixed-aic256", "--fine-tune-epochs", "60", "--aic-weight", "16"],
                      ["--diagnostic-population", "mixed-aic64", "--fine-tune-epochs", "8", "--aic-weight", "64"]):
            self.assertIn("registered D6 strength", self.refuse(*base, *extra))


if __name__ == "__main__":
    unittest.main()
