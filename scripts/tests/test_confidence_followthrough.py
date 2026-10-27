"""Followthrough waits on process identity and preserves the registered recipes."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.byconstruction import confidence_followthrough as owner


class Followthrough(unittest.TestCase):
    def test_live_queue_is_wait_not_completion_and_reused_pid_refuses(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            worker = owner.process_snapshot(os.getpid())
            queued = root / "QUEUED.json"
            queued.write_text(json.dumps(dict(queue_process=worker)))
            self.assertFalse(owner.omission_ready(root))
            worker["start_ticks"] = -1
            queued.write_text(json.dumps(dict(queue_process=worker)))
            with self.assertRaisesRegex(RuntimeError, "exited"):
                owner.omission_ready(root)
            (root / "LAUNCH_FAILED.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                owner.omission_ready(root)

    def test_historical_rejection_is_not_a_followthrough_gate(self):
        from scripts.byconstruction.mlic_confidence_full import FOLDS
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            selection = root / "TRAIN_SELECTION.json"
            selection.write_text(json.dumps(dict(schema="mlic-contrast-confidence-TRAIN-selection-v1",
                rule_commit="d37a01340fa9", eval_labels_used=False, finalist=None,
                cells=[dict(arm=a,fold=f,seed=s) for a in ("Mlic","MlicConfidence")
                       for f in FOLDS for s in range(3)])))
            (root / "TRAIN_SCREEN_COMPLETE.json").write_text(json.dumps(dict(selection_sha256=owner.sha(selection))))
            self.assertTrue(owner.omission_ready(root))
            selection.write_text(selection.read_text()+" ")
            with self.assertRaisesRegex(ValueError, "changed"):
                owner.omission_ready(root)

    def test_fold_plan_preserves_common_recipe_and_original_template(self):
        template = dict(schema="aic-recipe-adaptation-plan-v1",initial_cells={"Mlic":"original-cells"},
            recipe_authority={"Mlic":"original-study"},input_sha256={"original-input":"pin"},
            common_root="original-root",trainer_sha256="original-trainer",selection="full-panel-no-srocc-cutoff",
            target_role="D6-public_train-only", aic_folds="original-folds")
        unchanged = copy.deepcopy(template)
        plan = owner.fold_extension(template,Path("study"),Path("source"),"commit",
                                    {"source/aic_adaptation.py":"owner-pin"})
        self.assertEqual(template,unchanged)
        self.assertEqual(plan["recipes"],["Mlic","MlicConfidence"])
        self.assertEqual((plan["required_new_fold_cells"],plan["required_reused_fold_cells"]),(24,24))
        for key in ("common_root","trainer_sha256","selection","target_role","aic_folds"):
            self.assertEqual(plan[key],template[key])

    def test_full_plan_uses_completed_controls_without_refitting_initials(self):
        template = dict(schema="aic-recipe-full-plan-v1",input_sha256={"original":"pin"},
                        replay_control="preserved-control",original_full="original-full")
        pins={"source/aic_adaptation_full.py":"owner-pin","fold-plan":"fold-pin"}
        plan = owner.full_extension(template,Path("fold-plan"),Path("fold-panel"),Path("confidence-full"),
                                    Path("original-study"),Path("source"),"commit",pins)
        self.assertEqual(plan["required_new_initial_fits"],0)
        self.assertEqual(plan["required_new_continuations"],6)
        self.assertEqual(plan["required_reused_continuations"],6)
        self.assertEqual(plan["replay_control"],"preserved-control")
        self.assertEqual(plan["confidence_full"],"confidence-full")
        self.assertEqual(plan["reuse_mlic_full"],"original-study")


if __name__ == "__main__":
    unittest.main()
