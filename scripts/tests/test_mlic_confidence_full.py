"""Full confidence keeps independent factors, exact seeds and complete evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.byconstruction import mlic_confidence_full as owner


class FullConfidence(unittest.TestCase):
    def test_historical_rejection_does_not_exclude_complete_arm(self):
        selection = dict(schema="mlic-contrast-confidence-TRAIN-selection-v1", eval_labels_used=False, rule_commit="d37a01340fa9",
            finalist=None, cells=[dict(arm=a, fold=f, seed=s)
                for a in ("Mlic", "MlicConfidence") for f in owner.FOLDS for s in range(3)])
        owner.completed_omissions(selection)
        selection["cells"].pop()
        with self.assertRaisesRegex(ValueError, "inventory"):
            owner.completed_omissions(selection)
        selection["eval_labels_used"] = True
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            owner.completed_omissions(selection)

    def test_plan_cannot_omit_dependency_pins(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            path = root / "PLAN.json"
            path.write_text(json.dumps(dict(schema="mlic-confidence-full-plan-v1",
                runner_sha256=owner.sha(owner.__file__),study=str(root / "study"),
                registry=str(root / "registry.json"),input_sha256={})))
            with self.assertRaisesRegex(ValueError,"prerequisite"):
                owner.checked_plan(path)

    def test_only_child_fit_has_confidence_factors(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            parent = {"PATH": "unchanged"}
            binding = dict(path="manifest", sha256="a" * 64)
            child = owner.child_environment(parent, binding, root)
            self.assertEqual(parent, {"PATH": "unchanged"})
            self.assertEqual(owner.child_environment(parent), parent)
            self.assertEqual(child[owner.PREFIX + "SHA256"], binding["sha256"])
            with self.assertRaisesRegex(ValueError, "inherited"):
                owner.child_environment(child)
            (root / "CONFIDENCE_NORMALIZERS.json").write_text("preserved")
            with self.assertRaisesRegex(ValueError, "preserve"):
                owner.child_environment(parent, binding, root)

    def test_full_registry_cannot_be_omission_or_swapped_seed(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            cells = {}
            for seed in range(3):
                p = root / f"full-s{seed}.json"
                p.write_text(json.dumps(dict(candidate="MlicConfidence", fold="full",
                                            seed_index=seed, n_features=2263)))
                cells[f"full-s{seed}"] = dict(path=str(p), sha256=owner.sha(p))
            index = dict(schema="mlic-full-contrast-manifests-v1", cells=cells)
            path = root / "MANIFESTS.json"
            path.write_text(json.dumps(index))
            self.assertEqual(owner.full_binding(path, owner.sha(path), 2), cells["full-s2"])
            cells["full-s2"] = cells["full-s0"]
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "identity"):
                owner.full_binding(path, owner.sha(path), 2)
            index["schema"] = "mlic-contrast-manifests-v1"
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "registry"):
                owner.full_binding(path, owner.sha(path), 0)


if __name__ == "__main__":
    unittest.main()
