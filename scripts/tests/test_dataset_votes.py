"""Invented native counts; metadata admission never reads vote responses."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
import subjective_training_leg as leg
import test_subjective_training_leg as legacy


class NativeVotes(unittest.TestCase):
    def setUp(self):
        f = self.f = legacy.SubjectiveAdmission()
        f.setUp()
        self.addCleanup(f.doCleanups)
        guard = f.path.parent/"guard.json"
        guard.write_text(json.dumps(dict(schema="dataset-family-guard-v1", protected_families=["reserved"], inherited_leaks=0)))
        f.cell.update(formula_revision=7, family_guard_path=str(guard), family_guard_sha256=leg.sha(guard))
        f.authority["schema"] = "subjective-training-authority-v2"
        f.record.update(name="subjective_vote_bapps", train_weight=1.)
        f.groups = [("subjective_vote_bapps", f.path, 1., 0., "withinref,rank")]
        for i, k in enumerate(f.rows):
            k.update(reference=k["source"], observation=k["source"]+":"+"ab"[i])
        f.save()
        self.votes = f.path.parent/"votes.json"
        f.decl.update(schema="subjective-training-leg-v3", loss="binomial-two-vote",
                      nominal_weight=0.5, formula_revision=7, native_role="TRAIN", collection="bapps",
                      target_transform="native-vote-probability", native_votes_per_pair=2,
                      native_votes_path=str(self.votes), native_votes_sha256="a"*64,
                      keys_path=str(f.keys), keys_sha256=f.record["keys_sha256"])
        self.counts = dict(source=f.rows[0]["source"], n_a=1, n_b=1, n_tie=0, n=2, p_a=0.5)

    def payload(self, targets=(0.5, 0.5)):
        self.votes.write_text(json.dumps(dict(schema="bapps-native-votes-v1", comparisons=[self.counts])))
        self.f.decl["native_votes_sha256"] = leg.sha(self.votes)
        self.f.save()
        return leg.validate_native_votes(self.f.decl, targets)

    def test_metadata_does_not_open_counts_or_targets(self):
        original = Path.read_bytes
        def read(path):
            if path in (self.votes, self.f.path):
                self.fail("target-bearing data opened before all metadata admission")
            return original(path)
        with mock.patch.object(Path, "read_bytes", read):
            self.assertEqual(set(self.f.admit()), {"subjective_vote_bapps"})
        self.assertFalse(self.votes.exists())

    def test_native_endpoints_and_disagreement_bind_both_targets(self):
        for na in (0, 1, 2):
            self.counts.update(n_a=na, n_b=2-na, p_a=na/2)
            self.payload((na/2, 1-na/2))
        with self.assertRaisesRegex(ValueError, "target binding"):
            self.payload((0.5, 0.5))

    def test_count_and_identity_corruption_refused(self):
        original = copy.deepcopy(self.counts)
        for field, value in (("n_a", 0.5), ("n_a", True), ("n_b", 0), ("n_tie", 1),
                             ("n", 3), ("p_a", 1.), ("source", "different")):
            self.counts = {**original, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.payload()
        self.counts = original
        del self.counts["n_a"]
        with self.assertRaisesRegex(ValueError, "count fields"):
            self.payload()

    def test_v3_cannot_enter_as_hard_rank_or_change_mass_or_role(self):
        for field, value in (("loss", "hard-pair-preference"), ("native_role", "EVAL"),
                             ("nominal_weight", 1.), ("collection", "pieapp"),
                             ("target_transform", "orient-native-quality"), ("keys_sha256", "b"*64)):
            old = self.f.decl[field]
            self.f.decl[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "binding"):
                self.f.admit()
            self.f.decl[field] = old
        self.f.rows[1]["family"] = "reserved"
        self.f.save()
        self.f.decl["keys_sha256"] = self.f.record["keys_sha256"]
        with self.assertRaisesRegex(ValueError, "family"):
            self.f.admit()

    def test_vote_builder_refuses_mixed_trials_and_missing_policy_before_labels(self):
        from test_dataset_legs import root
        args = SimpleNamespace(root=str(self.f.path.parent/"missing-source"),
            out=str(self.f.path.parent/"fresh-root"), admission="missing-admission",
            features="missing-features", library="missing-library", columns="missing-columns",
            nominal_weight=0.5, collections="bapps", bapps_votes=True, additional_subsets=False,
            confidence=None)
        with mock.patch.object(root, "dataset_targets", side_effect=AssertionError("labels read")):
            with self.assertRaisesRegex(ValueError, "frozen confidence policy"):
                root.cmd_datasets(args)
            args.additional_subsets = True
            with self.assertRaisesRegex(ValueError, "separate studies"):
                root.cmd_datasets(args)

    def test_vote_screen_reuses_mass_rule_and_requires_omission_identity(self):
        from byc_cells import neff_decision, verify_omitted_vote_predictions
        arms = {a:{m:dict(per_seed=[0.8]*3) for m in ("old", "new", "balanced")}
                for a in ("Mlic", "MlicNeff")}
        arms["MlicNeff"]["balanced"]["per_seed"] = [0.81]*3
        mass = neff_decision(arms, [0]*21)
        votes = neff_decision({"Mlic":arms["Mlic"], "MlicVote":arms["MlicNeff"]}, [0]*21, "MlicVote")
        self.assertEqual(votes, {**mass, "finalist":"MlicVote"})
        self.assertIsNone(neff_decision({"Mlic":arms["Mlic"], "MlicVote":arms["MlicNeff"]},
                                       [0]*20+[1], "MlicVote")["finalist"])
        control, candidate = (self.f.path.parent/n for n in ("control", "candidate"))
        for directory in (control, candidate):
            directory.mkdir()
            for collection in ("bapps", "pieapp", "fg-iqa"):
                (directory/f"pred_dataset_{collection}.tsv").write_text("key\tprediction\ninvented\t1.0\n")
        self.assertEqual(len(verify_omitted_vote_predictions(control, candidate)), 3)
        (candidate/"pred_dataset_bapps.tsv").write_text("key\tprediction\ninvented\t1.1\n")
        with self.assertRaisesRegex(ValueError, "omission predictions differ"):
            verify_omitted_vote_predictions(control, candidate)


if __name__ == "__main__":
    unittest.main()
