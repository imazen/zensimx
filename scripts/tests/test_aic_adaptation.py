"""Refuse changed fit evidence, while allowing unfavorable panel statistics."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.byconstruction import aic_adaptation as owner


class RegisteredInventories(unittest.TestCase):
    def test_confidence_extension_cannot_shrink_original_study(self):
        original = dict(schema="aic-recipe-adaptation-plan-v1",recipes=list(owner.RECIPES))
        extension = dict(schema="aic-confidence-adaptation-plan-v1",recipes=["Mlic","MlicConfidence"])
        self.assertEqual(24*len(owner.registered_recipes(original)),96)
        self.assertEqual(24*len(owner.registered_recipes(extension)),48)
        for bad in (dict(original,recipes=extension["recipes"]),
                    dict(extension,recipes=original["recipes"]),dict(extension,recipes=["MlicConfidence"])):
            with self.assertRaises(ValueError):
                owner.registered_recipes(bad)

    def test_continuation_refuses_inherited_confidence_weighting(self):
        owner.require_clean_factors({"PATH":"unchanged"})
        for key in ("ZENSIM_RESEARCH_CONFIDENCE_MANIFEST","BYC_CONFIDENCE_MANIFESTS"):
            with self.assertRaisesRegex(ValueError,"inherit"):
                owner.require_clean_factors({key:"stale-parent-binding"})


class ContinuationEvidence(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=Path.home() / "tmp")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.initial, self.cell = self.root / "initial", self.root / "fit"
        for directory in (self.initial, self.cell):
            (directory / "refit").mkdir(parents=True)
            (directory / "refit/last.bin").write_bytes(b"synthetic-checkpoint-identity")
        for name in ("pred_kadid.tsv", "pred_aic_k0.tsv"):
            (self.cell / name).write_text("row_idx\tpred\n0\t4.0\n1\t3.0\n")
        self.plan = dict(common_root_sha256="root-pin", trainer_sha256="trainer-pin", aic_folds=str(self.root / "folds"))
        self.result = dict(training_only=True, epochs=4, pairs_per_epoch=50000, heldout="kadid", seed_index=0,
            spec=owner.SPEC, frozen_sha256="root-pin", selected_bake_sha256=owner.sha(self.cell / "refit/last.bin"),
            selection=dict(epoch_rule="last", selected_epoch=3))
        self.record = dict(aic_fold=0, trainer_sha256="trainer-pin", aic_weight=8,
            last_bin_sha256=owner.sha(self.cell / "refit/last.bin"), pred_sha256=owner.sha(self.cell / "pred_kadid.tsv"),
            aic_pred_sha256=owner.sha(self.cell / "pred_aic_k0.tsv"), curve={str(i): -1.0 for i in range(4)},
            continuation=dict(initial_checkpoint=str(self.initial / "refit/last.bin"), epochs=4, lr=0.00001, lr_schedule="constant"))
        self.authority = dict(schema="subjective-training-authority-v3", cells={"kadid-s0": {"continuation":
            dict(checkpoint=str(self.initial / "refit/last.bin"), checkpoint_sha256=owner.sha(self.initial / "refit/last.bin"),
                 epochs=4, lr=0.00001, lr_schedule="constant", optimizer_state="fresh")}})
        (self.cell / "train.log").write_text(f"trainer --group aic_public_train:{self.plan['aic_folds']}/aic_k0_fit.parquet:8.0:0:rank\n")
        self.save()

    def save(self):
        for name, value in (("result.json", self.result), ("byc_record.json", self.record),
                            ("continuation-authority.json", self.authority)):
            (self.cell / name).write_text(json.dumps(value))

    def validate(self):
        return owner.continuation_cell(self.cell, self.initial, "kadid", 0, 8, 4, self.plan)

    def test_no_srocc_gate_and_native_authority_schema(self):
        # Even deliberately unfavorable development scores remain reportable.
        self.assertEqual(self.validate()["epochs"], 4)

    def test_changed_initialization_or_predictions_refused(self):
        self.validate()
        (self.initial / "refit/last.bin").write_bytes(b"different-source-model")
        with self.assertRaisesRegex(ValueError, "initialization"):
            self.validate()
        (self.initial / "refit/last.bin").write_bytes(b"synthetic-checkpoint-identity")
        (self.cell / "pred_aic_k0.tsv").write_text("changed held-out prediction")
        with self.assertRaisesRegex(ValueError, "output differs"):
            self.validate()

    def test_folds_epochs_and_actual_mass_cannot_drift(self):
        original = copy.deepcopy(self.result)
        for key, value in (("training_only", False), ("epochs", 8), ("heldout", "tid2013"), ("seed_index", 1)):
            self.result = dict(original, **{key: value})
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
        self.result = original
        self.result["selection"]["selected_epoch"] = 2
        self.save()
        with self.assertRaises(ValueError):
            self.validate()
        self.result["selection"]["selected_epoch"] = 3
        self.save()
        p = self.cell / "train.log"
        p.write_text(p.read_text().replace(":8.0:0:rank", ":16.0:0:rank"))
        with self.assertRaisesRegex(ValueError, "mass differs"):
            self.validate()

    def test_queue_refuses_stale_process_and_changed_completion(self):
        path = self.root / "ADMISSION_STATE.json"
        path.write_text(json.dumps(dict(stage="waiting", failure=None,
            queue_process=dict(pid=os.getpid(), start_ticks=-1))))
        with self.assertRaisesRegex(RuntimeError, "exited without completion"):
            owner.admission_finished(self.root)
        (self.root / "screen").mkdir()
        (self.root / "screen/screen-summary.json").write_text("changed")
        path.write_text(json.dumps(dict(stage="screen-complete-requires-membership-freeze", failure=None,
                                       screen_summary_sha256="old-pin")))
        with self.assertRaisesRegex(ValueError, "completion changed"):
            owner.admission_finished(self.root)

    def test_heldout_base_refused_before_payload(self):
        (self.initial / "result.json").write_text(json.dumps(dict(training_only=False)))
        (self.initial / "byc_record.json").write_text("{}")
        with patch.object(owner, "sha", side_effect=AssertionError("payload opened too early")):
            with self.assertRaisesRegex(ValueError, "starting checkpoint contract"):
                owner.initial_cell(self.initial, "kadid", 0)

    def test_zero_step_uses_owner_selected_best_and_preserves_parity(self):
        directory = self.root / "zero"
        (directory / "refit").mkdir(parents=True)
        best = directory / "refit/best.bin"
        best.write_bytes(b"zero-step-selected-model")
        result = dict(epochs=0, selection=dict(selected_epoch=-1), training_only=True, spec=owner.SPEC,
                      selected_bake=str(best), selected_bake_sha256=owner.sha(best))
        (directory / "result.json").write_text(json.dumps(result))
        self.assertFalse((directory / "refit/last.bin").exists())
        self.assertEqual(owner.zero_checkpoint(directory), best)
        actual, reference = directory / "actual.tsv", directory / "reference.tsv"
        actual.write_text("row_idx\tpred\n0\t3.25\n")
        reference.write_bytes(actual.read_bytes())
        owner.require_prediction_parity(actual, reference)
        actual.write_text("row_idx\tpred\n0\t3.2501\n")
        with self.assertRaisesRegex(ValueError, "byte-for-byte"):
            owner.require_prediction_parity(actual, reference)
        best.write_bytes(b"substituted-model")
        with self.assertRaisesRegex(ValueError, "selected checkpoint"):
            owner.zero_checkpoint(directory)

    def test_recipe_label_cannot_be_attached_to_another_valid_model(self):
        row = dict(directory=str(self.initial), fold="kadid", seed=0)
        selection = dict(cells=[dict(arm="MlicVote", fold="kadid", seed=0, model_sha256="vote-model-pin")])
        with self.assertRaisesRegex(ValueError, "declared recipe cell"):
            owner.bind_recipe_cell("MlicVote", row, self.root, selection, "1,2")
        selection["cells"][0]["model_sha256"] = owner.sha(self.initial / "refit/last.bin")
        with self.assertRaisesRegex(ValueError, "declared recipe cell"):
            owner.bind_recipe_cell("MlicNeff", row, self.root, selection, "1,2")


if __name__ == "__main__":
    unittest.main()
