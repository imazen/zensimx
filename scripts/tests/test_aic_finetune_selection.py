"""The full-data continuation must obey the preregistered TRAIN selector."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "rev4_featpot/aic_finetune_full.py"
SPEC = importlib.util.spec_from_file_location("aic_finetune_full", MODULE)
FULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL)


class FrozenSelection(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.e21 = {arm: dict(delta=0., seed_unit_se=0.) for arm in FULL.STRENGTHS}
        self.aic = {arm: dict(mean_cell_global_gain=0., per_cell={str(i): {} for i in range(12)}) for arm in FULL.STRENGTHS}
        self.mixture = {arm: dict(cells=12, per_cell={str(i): dict(inversions=0) for i in range(12)}) for arm in FULL.STRENGTHS}

    def save(self, winner):
        files = {"E21.json": self.e21, "AIC_OOF.json": self.aic, "mixture_global.json": self.mixture}
        for name, value in files.items():
            (self.root / name).write_text(json.dumps(value))
        (self.root / "TRAIN_SELECTION.json").write_text(json.dumps(dict(
            winner=winner, evidence_sha256={name: FULL.sha(self.root / name) for name in files})))

    def test_exact_tie_prefers_the_smaller_registered_strength(self):
        self.save("MlicFT8x4")
        self.assertEqual(FULL.selected_arm(self.root), "MlicFT8x4")

    def test_one_cell_inversion_refuses_an_otherwise_leading_arm(self):
        self.aic["MlicFT8x4"]["mean_cell_global_gain"] = 1.
        self.mixture["MlicFT8x4"]["per_cell"]["0"]["inversions"] = 1
        self.save("MlicFT16x8")
        self.assertEqual(FULL.selected_arm(self.root), "MlicFT16x8")

    def test_e21_bound_can_leave_no_finalist(self):
        for value in self.e21.values():
            value.update(delta=-0.1, seed_unit_se=0.01)
        self.save(None)
        self.assertIsNone(FULL.selected_arm(self.root))

    def test_winner_and_evidence_changes_are_refused(self):
        self.save("MlicFT16x8")
        with self.assertRaisesRegex(ValueError, "frozen TRAIN rule"):
            FULL.selected_arm(self.root)
        self.save("MlicFT8x4")
        (self.root / "E21.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence pin"):
            FULL.selected_arm(self.root)


if __name__ == "__main__":
    unittest.main()
