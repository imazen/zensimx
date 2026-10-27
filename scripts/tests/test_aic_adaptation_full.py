"""Full-data continuation keeps seeds, initializations and recipe factors explicit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.byconstruction import aic_adaptation_full as owner


class FullExtensionInventory(unittest.TestCase):
    def test_original_and_extension_inventory_cannot_be_interchanged(self):
        original = dict(schema="aic-recipe-full-plan-v1",recipes=list(owner.RECIPES))
        extension = dict(schema="aic-confidence-full-plan-v1",recipes=["Mlic","MlicConfidence"])
        self.assertEqual(6*len(owner.full_recipes(original)),24)
        self.assertEqual(6*len(owner.full_recipes(extension)),12)
        with self.assertRaisesRegex(ValueError,"inventory"):
            owner.full_recipes(dict(original,recipes=extension["recipes"]))
        with self.assertRaisesRegex(ValueError,"inventory"):
            owner.full_recipes(dict(extension,recipes=["MlicConfidence"]))

    def test_reuse_requires_all_seeds_without_duplicates(self):
        rows = [dict(seed=s, value=s+10) for s in range(3)]
        self.assertEqual(owner.unique_seed(list(reversed(rows)),1)["value"],11)
        for bad in (rows[:2], [rows[0],rows[0],rows[2]]):
            with self.assertRaisesRegex(ValueError,"distinct"):
                owner.unique_seed(bad,0)

    def test_reused_continuation_must_match_strength_and_receipts(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            (root / "PLAN.json").write_text("original plan")
            evidence = root / "result.json"; evidence.write_text("original result")
            rows = [dict(arm="MlicFT8x4",recipe="Mlic",seed=s,weight=8,epochs=4,
                         directory=str(root / f"full_s{s}"),input_sha256={str(evidence):owner.sha(evidence)})
                    for s in range(3)]
            complete = dict(plan_sha256=owner.sha(root / "PLAN.json"),selector=None,continued=rows)
            receipt = root / "FULL_FITS_COMPLETE.json";receipt.write_text(json.dumps(complete))
            self.assertEqual(owner.reused_mlic(dict(reuse_mlic_full=str(root)),1,(8,4)),root / "full_s1")
            with self.assertRaisesRegex(ValueError,"distinct"):
                owner.reused_mlic(dict(reuse_mlic_full=str(root)),1,(16,8))
            evidence.write_text("changed result")
            with self.assertRaisesRegex(ValueError,"evidence changed"):
                owner.reused_mlic(dict(reuse_mlic_full=str(root)),1,(8,4))


class FullCommands(unittest.TestCase):
    def test_base_recipe_and_common_continuation_are_distinct(self):
        args = [Path("source"), Path("root"), Path("dest"), "1,2,3", 0]
        schedule = dict(path="matching.json", sha256="pin")
        base = owner.fit_command(*args, matching=schedule)
        self.assertIn("--matched-codec-full", base)
        self.assertNotIn("--init-from", base)
        for strength in owner.STRENGTHS:
            fit = owner.fit_command(*args, initial=Path("initial.bin"), strength=strength, aic=Path("aic_full_fit.parquet"))
            self.assertNotIn("--matched-codec-full", fit)
            self.assertEqual(fit[fit.index("--aic-weight") + 1], str(strength[0]))
            self.assertEqual(fit[fit.index("--fine-tune-epochs") + 1], str(strength[1]))
            with self.assertRaises(ValueError):
                owner.fit_command(*args, initial=Path("initial.bin"), strength=strength,
                                  aic=Path("aic_full_fit.parquet"), matching=schedule)
        with self.assertRaises(ValueError):
            owner.fit_command(*args, strength=(16, 8))
        with self.assertRaises(ValueError):
            owner.fit_command(*args[:-1], 3)


class FullControlParity(unittest.TestCase):
    def test_changed_result_or_model_cannot_pass(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            tool = root / "native-owner"
            tool.write_bytes(b"fixture-owner")
            cells = [root / label for label in ("original", "replay")]
            for cell in cells:
                (cell / "refit").mkdir(parents=True)
                (cell / "result.json").write_text(json.dumps(dict(epochs=120, selected_bake=str(cell))))
                for name in ("last.bin", "production-f16.bin"):
                    (cell / "refit" / name).write_bytes(b"exact numerical and serving bytes")
            def native_copy(argv, **kwargs):
                Path(argv[argv.index("--out") + 1]).write_bytes(Path(argv[argv.index("--in") + 1]).read_bytes())
            with patch.object(owner.subprocess, "run", side_effect=native_copy):
                result = owner.require_full_control_parity(*cells, tool, root / "equal")
                self.assertEqual(len(result["models"]), 2)
                (cells[1] / "refit/production-f16.bin").write_bytes(b"changed calibration")
                with self.assertRaisesRegex(ValueError, "numerical/serving"):
                    owner.require_full_control_parity(*cells, tool, root / "different")
            (cells[1] / "result.json").write_text(json.dumps(dict(epochs=119)))
            with self.assertRaisesRegex(ValueError, "recipe/admission"):
                owner.require_full_control_parity(*cells, tool, root / "bad-recipe")


class LiveTrainingProgress(unittest.TestCase):
    def test_stale_logs_do_not_create_live_trainers(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            (root / "train.log").write_text("subjective-rank epoch=119 exact_target_ties=2\n")
            self.assertEqual(owner.active_training([]), [])
            worker = dict(pid=123,start_ticks=456,state="R",cpu_ticks=789,
                          argv=["/bin/zensim_mlp_train","--epochs","120","--out",str(root/"refit/best.bin")])
            result = owner.active_training([worker])
            self.assertEqual(result[0]["latest_logged_epoch"],119)
            self.assertFalse(result[0]["completion_inferred"])
            self.assertEqual(result[0]["start_ticks"],456)
            (root / "train.log").write_text("  epoch  10 | lr=0.1\nsubjective-rank epoch=13 exact_target_ties=2\n")
            self.assertEqual(owner.active_training([worker])[0]["latest_logged_epoch"],13)

    def test_missing_log_and_unbound_arguments_are_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            worker = dict(pid=123,start_ticks=456,state="R",cpu_ticks=789,
                          argv=["zensim_mlp_train","--epochs","8","--out",str(Path(tmp)/"refit/best.bin")])
            result = owner.active_training([worker])[0]
            self.assertIsNone(result["latest_logged_epoch"])
            self.assertFalse(result["log_present"])
            self.assertEqual(result["epochs"],8)
            worker["argv"] = ["zensim_mlp_train"]
            self.assertIn("error",owner.active_training([worker])[0])


class FullEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "refit").mkdir()
        for name in ("last.bin", "production-f16.bin"):
            (self.root / "refit" / name).write_bytes(name.encode())
        (self.root / "keep_features.txt").write_text("fixture")
        self.initial = self.root / "original.bin"
        self.initial.write_bytes(b"initial model")
        self.result = dict(schema="rev5-qualified-training-cell-v1", training_only=True,
            execution_contract="registered-fit", seed_index=1, spec=owner.SPEC, head="N", epochs=8, pairs_per_epoch=50000,
            selection=dict(epoch_rule="last", selected_epoch=7), init_seed=1103, sample_seed=100000101,
            frozen_sha256="root-pin", width=1853, kept_features=410, output_stage="e33-identity-knot-tail2",
            selected_bake_sha256=owner.sha(self.root / "refit/last.bin"),
            packed_model_sha256=owner.sha(self.root / "refit/production-f16.bin"),
            continuation=dict(initial_checkpoint=str(self.initial), initial_checkpoint_sha256=owner.sha(self.initial),
                              epochs=8, lr=.00001, lr_schedule="constant", optimizer_state="fresh"),
            dev_curve={"7": -1.0})
        self.selection = dict(root_sha256="root-pin", requested_exclusions=[], omitted_collections=[])
        self.authority = dict(schema="subjective-training-authority-v3", cells={"full-s1": dict(route="full",
            continuation=dict(checkpoint=str(self.initial), checkpoint_sha256=owner.sha(self.initial),
                              epochs=8, lr=.00001, lr_schedule="constant", optimizer_state="fresh"))})
        self.command = "$ trainer --group aic_public_train:/frozen/aic_full_fit.parquet:16.0:0:rank\n"
        self.save()

    def save(self):
        for name, value in (("result.json", self.result), ("DATASET_GROUP_SELECTION.json", self.selection),
                            ("continuation-authority.json", self.authority)):
            (self.root / name).write_text(json.dumps(value))
        (self.root / "train.log").write_text(self.command)

    def check(self):
        return owner.check_fit(self.root, 1, "root-pin", initial=self.initial, strength=(16, 8))

    def test_unfavorable_scalar_does_not_reject_recipe(self):
        self.assertEqual(self.check()["seed"], 1)

    def test_seed_final_epoch_and_initialization_are_required(self):
        original = copy.deepcopy(self.result)
        for key, value in (("seed_index", 0), ("sample_seed", 101), ("training_only", False), ("epochs", 120)):
            self.result = dict(original, **{key: value})
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
        self.result = original
        self.save()
        self.initial.write_bytes(b"substituted initial model")
        with self.assertRaisesRegex(ValueError, "initialization"):
            self.check()

    def test_omissions_native_authority_and_actual_aic_mass_cannot_drift(self):
        self.selection["omitted_collections"] = ["pipal"]
        self.save()
        with self.assertRaisesRegex(ValueError, "omission"):
            self.check()
        self.selection["omitted_collections"] = []
        self.authority["cells"]["full-s1"]["continuation"]["optimizer_state"] = "reused"
        self.save()
        with self.assertRaisesRegex(ValueError, "authority"):
            self.check()
        self.authority["cells"]["full-s1"]["continuation"]["optimizer_state"] = "fresh"
        self.command = self.command.replace(":16.0:0:rank", ":8.0:0:rank")
        self.save()
        with self.assertRaisesRegex(ValueError, "actual full continuation"):
            self.check()


if __name__ == "__main__":
    unittest.main()
