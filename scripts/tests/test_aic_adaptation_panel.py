"""Fitted-source predictions never enter held-out-source aggregates."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.byconstruction import aic_adaptation_panel as owner
from scripts.byconstruction import aic_adaptation_diagnostics as diagnostics
from scripts.byconstruction import mixture_test


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

    def test_extension_completion_does_not_satisfy_original_population(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            receipt = root / "FOLD_FITS_COMPLETE.json"
            receipt.write_text(json.dumps(dict(plan_sha256="pin", cells=[{}]*48)))
            self.assertTrue(owner.fits_finished(root,"pin",48))
            with self.assertRaisesRegex(ValueError,"completion"):
                owner.fits_finished(root,"pin")
            with self.assertRaisesRegex(ValueError,"population"):
                owner.fits_finished(root,"pin",24)

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


class DiagnosticIntegrity(unittest.TestCase):
    def test_ladder_cohorts_ties_and_disagreement(self):
        # A: three concurring pairs, with one prediction inversion.
        # B: one human-disagreeing pair, one concurring pair, one human tie.
        pairs = pd.DataFrame(dict(group=["PTC_S01_Ref_00"] * 3 + ["PTC_S02_Ref_00"] * 3,
                                  codec=["JPG"] * 6, quality=[1, 2, 3] * 2,
                                  key=[f"PTC_S{s:02d}_JPG_{q:02d}" for s in (1, 2) for q in (1, 2, 3)]))
        x, y = np.array([3., 1., 2., 1., 3., 3.]), np.array([3., 2., 1., 2., 3., 2.])
        mask = np.array([True] * 3 + [False] * 3)
        fitted = diagnostics.ladder_diagnostics(pairs, x, y, mask, lambda x, y: x)
        held = diagnostics.ladder_diagnostics(pairs, x, y, ~mask, lambda x, y: x)
        self.assertEqual((fitted["rows"], fitted["total_pairs"], fitted["concurring"]["eligible_pairs"]), (3, 3, 3))
        self.assertEqual(fitted["concurring"]["inversions"], 1)
        self.assertEqual(held["subjective_only"]["eligible_pairs"], 1)
        self.assertEqual(held["subjective_ties"], 1)
        self.assertEqual(held["concurring"]["prediction_ties"], 1)
        pairs.loc[0, "quality"] = 2
        with self.assertRaisesRegex(ValueError, "filename binding"):
            diagnostics.ladder_diagnostics(pairs, x, y, mask, lambda x, y: x)

    def test_undefined_seed_does_not_change_estimand(self):
        baseline = {(f, s): {"statistics": {"metric": 0.}} for f in owner.FOLDS for s in range(3)}
        candidate = copy.deepcopy(baseline)
        candidate[owner.FOLDS[0], 1]["statistics"]["metric"] = float("nan")
        result = diagnostics.paired_seeds(candidate, baseline, owner.FOLDS, "metric")
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["seed_unit_se"])
        self.assertEqual(result["seed_counts"], dict(required=3, defined=2, undefined=1))
        self.assertEqual(result["per_seed_fold_counts"][1]["undefined"], 1)

    def test_native_result_labels_and_row_coverage_required(self):
        jobs = [("a", [1, 2], [3, 4])]
        for rows in ([], [dict(label="b", n=2, n_dropped=0)], [dict(label="a", n=1, n_dropped=0)]):
            with self.assertRaises(ValueError):
                diagnostics.checked_statistics(jobs, rows)
        diagnostics.checked_statistics(jobs, [dict(label="a", n=1, n_dropped=1)])

    def test_canonical_mixture_reuse_refuses_changed_identity_and_scores(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"model fixture")
            (root / "manifest.json").write_text(json.dumps(dict(rows=[dict(ref="r", base="b", more="m")])))
            valid = dict(bake_sha256=owner.sha(model), post="identity", ring_form="raw", pyramid="xyb",
                         scores=dict(b=1., m=2.))
            score = root / "scores_probe.json"
            args = SimpleNamespace(out=root, bake=[f"probe={model}"], post="identity", bin_dir=root, jobs=1)
            with patch.dict(os.environ, ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb"), patch.object(mixture_test.subprocess, "run") as run:
                score.write_text(json.dumps(valid))
                self.assertEqual(mixture_test.cmd_score(args), 0)
                run.assert_not_called()
                mutations = [dict(bake_sha256="wrong"), dict(post="changed"), dict(ring_form="changed"),
                             dict(pyramid="changed"), dict(scores=dict(b=1.)), dict(scores=dict(b=1., m=float("nan")))]
                for mutation in mutations:
                    score.write_text(json.dumps(dict(valid, **mutation)))
                    with self.assertRaisesRegex(ValueError, "cached bake"):
                        mixture_test.cmd_score(args)
                model.write_bytes(b"substituted dense model")
                score.write_text(json.dumps(valid))
                with self.assertRaisesRegex(ValueError, "cached bake"):
                    mixture_test.cmd_score(args)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
