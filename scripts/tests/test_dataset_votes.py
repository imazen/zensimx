"""Invented native counts; metadata admission never reads vote responses."""
import copy
import io
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

    def test_actual_training_wrapper_admits_v3_with_external_authority(self):
        import v2_lodo_mlp as wrapper
        import pyarrow as pa
        import pyarrow.parquet as pq
        f = self.f
        pq.write_table(pa.table({"human_score": [0.5, 0.5]}), f.path)
        self.payload()
        f.decl["table_sha256"] = f.record["table_sha256"] = leg.sha(f.path)
        f.decl["immutable_input_roots"] = [str(f.path.parent)]
        f.cell["width"] = 1825
        pin = f.save()
        output = f.path.parent.parent/(f.path.parent.name+"-vote-output")
        argv = wrapper.train_command(f.groups, 1101, 101, 1825, output/"keep.txt", "N", output/"last.bin",
            recipe={"hidden":128}, strict_admission=True,
            subjective_authority=(f.authority_path,pin,"invented-cell",[0,1]))
        self.assertIn("--subjective-authority", argv)
        self.assertIn("subjective_vote_bapps:"+str(f.path)+":1.0:0.0:withinref,rank", argv)
        with self.assertRaisesRegex(ValueError, "independently pinned authority"):
            wrapper.strict_training_groups(f.groups)

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

    def test_released_two_judge_aggregate_recovers_only_exact_sufficient_statistics(self):
        import numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"byconstruction"))
        from bapps_vote_audit import recover, judge_path
        def raw(value):
            stream = io.BytesIO()
            np.save(stream,np.array([value],dtype=np.float64),allow_pickle=False)
            return stream.getvalue()
        for p1 in (0.,0.5,1.):
            label = dict(split="train",n="2",p_a=str(1-p1),n_a="",n_b="",n_tie="")
            self.assertEqual(recover(raw(p1),label),dict(n_a=int(2*(1-p1)),n_b=int(2*p1),n_tie=0,n=2,p_a=1-p1))
        for change in (dict(split="val"),dict(n=""),dict(n="5"),dict(p_a="0.75"),dict(n_a="2")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                recover(raw(0.5),{**label,"p_a":"0.5",**change})
        with self.assertRaises(ValueError):
            recover(raw(0.25),{**label,"p_a":"0.75"})
        member=dict(reference_path="train/cnn/ref/000001.png",a_path="train/cnn/p0/000001.png",b_path="train/cnn/p1/000001.png")
        self.assertEqual(judge_path(member),"train/cnn/judge/000001.npy")
        with self.assertRaises(ValueError):
            judge_path({**member,"reference_path":"val/cnn/ref/000001.png"})

    def test_recovery_admission_is_metadata_only_and_payload_binds_native_response(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"byconstruction"))
        import bapps_vote_audit as audit
        directory = self.f.path.parent
        count_path, receipt_path, ledger = (directory/n for n in ("counts.json","audit.json","reads.json"))
        receipt = dict(schema="bapps-native-vote-source-audit-v1", role="TRAIN", state="verified", rows=146003,
            original_root_sha256=audit.ROOT_SHA, membership_sha256=audit.MEMBERSHIP_SHA,
            confidence_policy_sha256=audit.POLICY_SHA, archive_sha256=audit.ARCHIVE_SHA,
            source_script_sha256=audit.AUDITOR_SHA, excluded_train_judgments_parsed=0, eval_judgments_parsed=0,
            count_path=str(count_path),count_sha256="a"*64)
        receipt_path.write_text(json.dumps(receipt))
        self.assertEqual(audit.admit_recovery(receipt_path,audit.sha(receipt_path)),receipt)
        self.assertFalse(count_path.exists())
        for field,value in (("role","EVAL"),("rows",146002),("source_script_sha256","b"*64),
                            ("excluded_train_judgments_parsed",1),("eval_judgments_parsed",1)):
            receipt_path.write_text(json.dumps({**receipt,field:value}))
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.admit_recovery(receipt_path,audit.sha(receipt_path))
        member = dict(row_index=7,reference_path="train/cnn/ref/000001.png",
            a_path="train/cnn/p0/000001.png",b_path="train/cnn/p1/000001.png")
        row = dict(row=7,source="bapps:pairs.tsv:7",judge_member="train/cnn/judge/000001.npy",
            judge_sha256="c"*64,n_a=1,n_b=1,n_tie=0,n=2,p_a=0.5)
        label = dict(n="2",p_a="0.5",n_a="",n_b="",n_tie="")
        def apply(changes=None, label_changes=None):
            count_path.write_text(json.dumps(dict(schema="bapps-released-two-vote-counts-v1",rows=[{**row,**(changes or {})}])))
            ledger.write_text(json.dumps(dict(reads=[])))
            return audit.apply_recovery({**receipt,"count_sha256":audit.sha(count_path)},[member],
                {("pairs.tsv",7):{**label,**(label_changes or {})}},ledger)
        self.assertEqual(apply()[("pairs.tsv",7)],{**label,"n_a":"1","n_b":"1","n_tie":"0"})
        self.assertEqual(json.loads(ledger.read_text())["reads"][-1]["state"],"verified")
        for field,value in (("row",8),("source","bapps:pairs.tsv:8"),("judge_member","val/cnn/judge/000001.npy"),
                            ("n_a",True),("n_b",0),("n_tie",1),("n",3),("p_a",1.)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                apply({field:value})
        for change in (dict(n_a="2"),dict(p_a="0.25"),dict(n="5")):
            with self.subTest(change=change),self.assertRaises(ValueError):
                apply(label_changes=change)
        with self.assertRaisesRegex(ValueError,"payload changed"):
            audit.apply_recovery(receipt,[member],{("pairs.tsv",7):label},ledger)
        with self.assertRaisesRegex(ValueError,"protected subjective"):
            audit.admit_recovery(directory/"_sealed"/"audit.json","a"*64)


if __name__ == "__main__":
    unittest.main()
