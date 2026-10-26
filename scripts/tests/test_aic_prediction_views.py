"""Prediction repairs preserve source features and fail on changed observations."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "byconstruction"))
from aic_train import restore_prediction_features
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
from aic_finetune_status import status
import tempfile
import json

class PredictionViews(unittest.TestCase):
    def tables(self):
        reference = pa.table({"key": ["a", "b"], "ref_basename": ["r", "r"],
                              "f0": [1.0, 2.0], "f1": [float("nan"), 3.0]})
        table = reference.set_column(3, "f1", pa.array([None, 3.0])).append_column("human_score", pa.array([7., 8.]))
        table = table.set_column(0, "key", pa.array(["a", "b"], type=pa.large_string()))
        return table, reference

    def test_restore_only_original_nan_and_preserve_other_columns(self):
        table, reference = self.tables()
        restored, count = restore_prediction_features(table, reference)
        self.assertEqual(count, 1)
        self.assertEqual(restored["f1"].null_count, 0)
        self.assertTrue(np.isnan(restored["f1"][0].as_py()))
        for name in ("key", "ref_basename", "f0", "human_score"):
            self.assertTrue(restored[name].equals(table[name]))

    def test_refuses_missing_observed_feature(self):
        table, reference = self.tables()
        table = table.set_column(2, "f0", pa.array([None, 2.0]))
        with self.assertRaisesRegex(ValueError, "feature differs"):
            restore_prediction_features(table, reference)

    def test_refuses_changed_value_and_row_order(self):
        table, reference = self.tables()
        with self.assertRaisesRegex(ValueError, "feature differs"):
            restore_prediction_features(table.set_column(2, "f0", pa.array([1.1, 2.0])), reference)
        with self.assertRaisesRegex(ValueError, "key/source"):
            restore_prediction_features(table.take(pa.array([1, 0])), reference)

    def test_failed_dead_run_and_wrapper_are_not_active_cells(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            (root / "RUN_STATE.json").write_text(json.dumps({"stage": "registered-arm", "pid": 99999999}))
            arm = root / "cells/MlicFT8x4"
            arm.mkdir(parents=True)
            (arm / "without_kadid_s0.wrapper.log").write_text("failed")
            self.assertEqual(status(root)["status"], "not-running")
            self.assertEqual(status(root)["arms"]["MlicFT8x4"]["active"], {})
            (root / "FAILED.json").write_text('{"stage":"registered-arm"}')
            self.assertEqual(status(root)["status"], "failed")
            self.assertEqual(status(root)["completed_cells"], 0)

class RecoveryGuards(unittest.TestCase):
    def test_native_change_and_control_tampering_refuse(self):
        from aic_finetune_chain import verify_recovery, digest
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            prior, repo = root / "prior", root / "repo"
            (prior / "bin").mkdir(parents=True)
            repo.mkdir()
            native = repo / "trainer.rs"
            native.write_text("native source")
            binary = prior / "bin/trainer"
            binary.write_bytes(b"frozen tool")
            actual, reference = prior / "actual", prior / "reference"
            actual.write_bytes(b"predictions")
            reference.write_bytes(actual.read_bytes())
            control = dict(byte_identical=True, rows=7869, actual=str(actual),
                           reference=str(reference), prediction_sha256=digest(actual))
            frozen = dict(source_files={"trainer.rs": digest(native)},
                          binaries={"trainer": digest(binary)}, zero_step=control, default_parity=control)
            for name in ("PREFLIGHT_PASSED.json", "BIN_FROZEN.json"):
                (prior / name).write_text(json.dumps(frozen))
            packet = dict(root="root", initial_cells="initial", fold_manifest_sha256="pin")
            verify_recovery(prior, packet, repo)
            native.write_text("changed native source")
            with self.assertRaisesRegex(ValueError, "full preflight required"):
                verify_recovery(prior, packet, repo)
            native.write_text("native source")
            actual.write_bytes(b"changed predictions")
            with self.assertRaisesRegex(ValueError, "passing control changed"):
                verify_recovery(prior, packet, repo)

if __name__ == "__main__":
    unittest.main()
