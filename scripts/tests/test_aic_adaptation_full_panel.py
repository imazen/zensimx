"""Full-data seed and completion integrity; invented predictions only."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.byconstruction import aic_adaptation_full_panel as owner


class FullPanelIntegrity(unittest.TestCase):
    def test_seed_mean_has_exact_complete_coverage(self):
        rows = [dict(seed=s, values=[s, s+3.]) for s in (2, 0, 1)]
        np.testing.assert_array_equal(owner.seed_average(rows), [1., 4.])
        for bad in (rows[:-1], rows[:-1]+[rows[0]], rows+[rows[0]],
                    [dict(seed=s, values=[float("nan"), 1]) for s in range(3)],
                    [dict(seed=s, values=[s] if s else [1, 2]) for s in range(3)]):
            with self.assertRaises(ValueError):
                owner.seed_average(bad)

    def test_paired_undefined_seed_remains_explicit(self):
        out = owner.seed_delta({0: 2., 1: float("nan"), 2: 4.}, {0: 1., 1: 2., 2: 3.})
        self.assertIsNone(out["mean"])
        self.assertIsNone(out["seed_unit_se"])
        self.assertEqual(out["seed_counts"], dict(required=3, defined=2, undefined=1))
        out = owner.seed_delta({0: 2., 1: 3., 2: 4.}, {0: 1., 1: 2., 2: 3.})
        self.assertEqual(out["mean"], 1.)
        self.assertEqual(out["seed_unit_se"], 0.)
        with self.assertRaises(ValueError):
            owner.seed_delta({0: 2.}, {0: 1.})

    def test_confidence_population_cannot_replace_original_four_arms(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            recipes = ("Mlic", "MlicConfidence")
            complete = dict(plan_sha256="pin",initial={r:[{}, {}, {}] for r in recipes},continued=[{}]*12)
            path = root / "FULL_FITS_COMPLETE.json";path.write_text(json.dumps(complete))
            self.assertTrue(owner.finished(root,"pin",recipes))
            with self.assertRaisesRegex(ValueError,"population"):
                owner.finished(root,"pin")
            complete["continued"].pop();path.write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError,"population"):
                owner.finished(root,"pin",recipes)

    def test_stale_fitter_or_partial_completion_cannot_start_panels(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as directory:
            root = Path(directory)
            (root / "FULL_STATE.json").write_text(json.dumps(dict(failures=[], worker=dict(pid=os.getpid(), start_ticks=-1))))
            with self.assertRaisesRegex(RuntimeError, "exited without completion"):
                owner.finished(root, "expected")
            result = dict(plan_sha256="expected", initial={recipe: [{}, {}, {}] for recipe in owner.RECIPES}, continued=[{}] * 23)
            (root / "FULL_FITS_COMPLETE.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "population differs"):
                owner.finished(root, "expected")


if __name__ == "__main__":
    unittest.main()
