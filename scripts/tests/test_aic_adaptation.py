"""Refuse changed fit evidence, while allowing unfavorable panel statistics."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.byconstruction import aic_adaptation as owner


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


if __name__ == "__main__":
    unittest.main()
