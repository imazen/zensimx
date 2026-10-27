"""Fitted-source predictions never enter held-out-source aggregates."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.byconstruction import aic_adaptation_panel as owner


class SeparateRoles(unittest.TestCase):
    def setUp(self):
        self.folds = np.repeat(np.arange(7), 4)
        self.rows = []
        for seed in range(3):
            for i, fold in enumerate(owner.FOLDS):
                k = (4 * seed + i) % 7
                self.rows.append(dict(seed=seed, fold=fold, aic_fold=k,
                                      values=np.where(self.folds == k, 42.0, -7.0)))

    def test_source_roles_stay_separate_with_unequal_coverage(self):
        held, held_count = owner.role_average(self.rows, self.folds, "heldout")
        fitted, fit_count = owner.role_average(self.rows, self.folds, "fitted")
        np.testing.assert_array_equal(held, np.full(28, 42.0))
        np.testing.assert_array_equal(fitted, np.full(28, -7.0))
        self.assertEqual(set(held_count), {1, 2})
        self.assertEqual(set(fit_count), {10, 11})
        np.testing.assert_array_equal(held_count + fit_count, np.full(28, 12))

    def test_duplicate_missing_or_wrong_fold_is_not_coverage(self):
        for rows in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            with self.assertRaises(ValueError):
                owner.role_average(rows, self.folds, "heldout")
        rows = copy.deepcopy(self.rows)
        rows[0]["aic_fold"] = 1
        with self.assertRaisesRegex(ValueError, "rotation changed"):
            owner.role_average(rows, self.folds, "heldout")
        rows = copy.deepcopy(self.rows)
        rows[0]["values"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite ordered prediction"):
            owner.role_average(rows, self.folds, "fitted")

    def test_undefined_remains_explicit_not_zero(self):
        self.assertEqual(owner.finite_json(dict(metric=float("nan"), rows=17)), dict(metric=None, rows=17))

    def test_stale_fit_process_and_wrong_completion_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as temporary:
            root = Path(temporary)
            (root / "RUN_STATE.json").write_text(json.dumps(dict(failures=[],
                queue_process=dict(pid=os.getpid(), start_ticks=-1))))
            with self.assertRaisesRegex(RuntimeError, "exited without completion"):
                owner.fits_finished(root, "expected")
            (root / "FOLD_FITS_COMPLETE.json").write_text(json.dumps(dict(plan_sha256="changed", cells=[{}] * 96)))
            with self.assertRaisesRegex(ValueError, "completion differs"):
                owner.fits_finished(root, "expected")


if __name__ == "__main__":
    unittest.main()
