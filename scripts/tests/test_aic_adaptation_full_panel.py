"""Full-data seed and completion integrity; invented predictions only."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.byconstruction import aic_adaptation_full_panel as owner


class ExposedLiveScope(unittest.TestCase):
    def rows(self):
        return [dict(index=i, case=i < 10, ref=f"source{i % 10}", dist=f"image{i}", human=float(i))
                for i in range(36)]

    def test_only_fixed_exposed_rows_are_retained(self):
        rows = self.rows()
        rows[0]["old_model"] = 5
        scoped = owner.live_rows(dict(rows=rows, rows_read=36))
        self.assertEqual(len(scoped), 36)
        self.assertNotIn("old_model", scoped[0])
        self.assertEqual(scoped[0]["human"], 0)

    def test_population_duplicates_missing_controls_and_nonfinite_targets_refuse(self):
        for change in (lambda r: r.pop(), lambda r: r[0].update(index=1),
                       lambda r: r[0].update(ref="unmatched"), lambda r: r[0].update(human=float("nan"))):
            rows = self.rows()
            change(rows)
            with self.assertRaises(ValueError):
                owner.live_rows(dict(rows=rows, rows_read=36))

    def test_pixel_scoring_requires_compute_lock_before_opening_plan(self):
        with patch.object(owner, "compute_lock", return_value=dict(held_by_worker_ancestor=False)):
            with self.assertRaisesRegex(ValueError, "shared compute lock"):
                owner.live_run(Path("invented-plan"), Path("invented-output"))


class E21SubsetInputs(unittest.TestCase):
    def test_compression_cohorts_reuse_the_established_type_memberships(self):
        np.testing.assert_array_equal(owner.compression_mask("kadid", ["kadid_08", "kadid_09", "kadid_10"]),
                                      [False, True, True])
        np.testing.assert_array_equal(owner.compression_mask("tid2013", ["tid_09", "tid_10", "tid_11", "tid_21"]),
                                      [False, True, True, True])
        np.testing.assert_array_equal(owner.compression_mask("konfig", ["jpeg2000", "other"]), [True, False])
        np.testing.assert_array_equal(owner.compression_mask("cid22_a25", ["a", "b"]), [True, True])

    def test_full_predictions_bind_to_their_completed_model_and_inventory(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            plan = dict(schema="aic-share-full-plan-v1", arms=dict(XrefAIC256="mixed-aic256-xref"), packed_epoch=59)
            (root / "PLAN.json").write_text(json.dumps(plan))
            fits, inventory = [], {}
            for seed in range(3):
                model = root / "full/XrefAIC256" / f"full_s{seed}/pack_e059/refit/production-f16.bin"
                model.parent.mkdir(parents=True)
                model.write_bytes(f"invented seed {seed}".encode())
                fits.append(dict(seed=seed, packed=dict(packed_model_sha256=owner.sha(model))))
                for dataset in owner.FOLDS:
                    p = root / "panels/predictions" / f"XrefAIC256e59|s{seed}|{dataset}.tsv"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("row_idx\tpred\n0\t1\n1\t2\n")
                    inventory[str(p.relative_to(root))] = owner.sha(p)
            (root / "FITS_COMPLETE.json").write_text(json.dumps(dict(plan_sha256=owner.sha(root / "PLAN.json"), cells=fits)))
            inventory["FITS_COMPLETE.json"] = owner.sha(root / "FITS_COMPLETE.json")
            (root / "EVIDENCE_INVENTORY.json").write_text(json.dumps(inventory))
            (root / "COMPLETE.json").write_text(json.dumps(dict(plan_sha256=owner.sha(root / "PLAN.json"),
                evidence_inventory_sha256=owner.sha(root / "EVIDENCE_INVENTORY.json"))))
            group = dict(name="full", kind="full", run=str(root), model="candidate")
            rows, inputs = owner.e21_group(group)
            self.assertEqual(len(rows), 12)
            self.assertEqual({r["scope"] for r in rows}, {"in sample"})
            self.assertEqual({r["seed"] for r in rows}, {0, 1, 2})
            with self.assertRaisesRegex(ValueError, "omission plan"):
                owner.e21_group(dict(group, kind="omission"))
            prediction = Path(rows[0]["prediction"])
            self.assertIn(str(prediction), inputs)
            prediction.write_text("row_idx\tpred\n0\t100\n1\t200\n")
            with self.assertRaisesRegex(ValueError, "source prediction changed"):
                owner.e21_group(group)
            prediction.write_text("row_idx\tpred\n0\t1\n1\t2\n")
            # Updating a model receipt alone cannot silently rebind a completed source.
            changed = json.loads((root / "FITS_COMPLETE.json").read_text())
            changed["cells"][0]["packed"]["packed_model_sha256"] = "0" * 64
            (root / "FITS_COMPLETE.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "source fit records changed"):
                owner.e21_group(group)

    def test_paired_contrasts_cannot_mix_fitted_and_omitted_populations(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps(dict(groups=[dict(name="full"), dict(name="omission")],
                                             contrasts=[["full", "omission"]])))
            with patch.object(owner, "e21_group", side_effect=[
                    ([dict(group="full", scope="in sample")], {}),
                    ([dict(group="omission", scope="out of sample")], {})]):
                with self.assertRaisesRegex(ValueError, "cannot form a paired contrast"):
                    owner.e21_plan(config, root / "report")
            self.assertFalse((root / "report").exists())


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
