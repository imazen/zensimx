"""Invented native counts; metadata admission never reads vote responses."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
