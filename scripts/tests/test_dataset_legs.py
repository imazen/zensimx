"""TRAIN-only ingress, native direction and pair-local hard-label semantics."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
import test_subjective_training_leg as legacy

spec = importlib.util.spec_from_file_location("dataset_root", Path(__file__).resolve().parents[1]/"byconstruction/ring_root.py")
root = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root)


class DatasetAuthority(unittest.TestCase):
    def test_dataset_valid_and_reserved_family_refusal(self):
        f = legacy.SubjectiveAdmission()
        f.setUp()
        self.addCleanup(f.doCleanups)
        guard = f.path.parent/"guard.json"
        guard.write_text(json.dumps(dict(schema="dataset-family-guard-v1",protected_families=["reserved"],inherited_leaks=0)))
        f.cell.update(formula_revision=7,family_guard_path=str(guard),family_guard_sha256=root.sha(guard))
        f.authority["schema"] = "subjective-training-authority-v2"
        f.record.update(name="subjective_rank_fixture", train_weight=1.)
        f.groups = [("subjective_rank_fixture",f.path,1.,0.,"withinref,rank")]
        f.decl.update(schema="subjective-training-leg-v2",nominal_weight=0.5,formula_revision=7,
                      target_transform="orient-native-quality",native_role="TRAIN",loss="hard-pair-preference")
        for k in f.rows:
            k["reference"] = k["source"]
        self.assertEqual(set(f.admit()), {"subjective_rank_fixture"})
        f.rows[1]["family"] = "reserved"
        with self.assertRaisesRegex(ValueError,"family"):
            f.admit()


class DatasetTargets(unittest.TestCase):
    def test_only_selected_train_lines_are_parsed_and_ledgered(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"labels.tsv"
            p.write_text('item\tscore\ntrain\t3.5\n"malformed non-TRAIN payload\n')
            ledger = Path(d)/"LABEL_READS.json"
            self.assertEqual(root.dataset_targets(p,{0},["score"],ledger,"fixture"), {0:{"score":"3.5"}})
            read = json.loads(ledger.read_text())["reads"][0]
            self.assertEqual((read["rows"],read["columns"],read["state"]),([0],["score"],"read"))

    def test_soft_probabilities_become_hard_pair_local_rows_and_ties_drop(self):
        members, mapping, labels = [], [], {}
        for i,p in enumerate((0.75,0.25,0.5)):
            members.append(dict(input_table="pairs.tsv",row_index=i,reference_path="ref",a_path="a",b_path="b",family_id=7))
            labels[("pairs.tsv",i)] = dict(ref="ref",a="a",b="b",p_a=str(p))
            for side in ("a","b"):
                mapping.append(dict(input_table="pairs.tsv",membership_row_index=i,role="TRAIN",skip_reason=None,
                                    side=side,pair_key=f"{i}{side}"))
        obs, dropped = root.dataset_observations("bapps", members, mapping, labels)
        self.assertEqual([r["human_score"] for r in obs],[1.,0.,0.,1.])
        self.assertEqual(obs[0]["ref_basename"],obs[1]["ref_basename"])
        self.assertNotEqual(obs[0]["ref_basename"],obs[2]["ref_basename"])
        self.assertEqual(dropped["exact_probability_tie"],1)

    def test_native_dmos_is_negated_and_mos_is_preserved(self):
        members, mapping, labels = [], [], {}
        for i,(value,direction) in enumerate(((2,"false"),(1,"false"),(2,"true"),(1,"true"))):
            ref = f"ref{i//2}"
            members.append(dict(input_table="labels.tsv",row_index=i,reference_path=ref,item_path=str(i),family_id=i//2))
            mapping.append(dict(input_table="labels.tsv",membership_row_index=i,role="TRAIN",skip_reason=None,
                                side="item",pair_key=str(i)))
            labels[("labels.tsv",i)] = dict(ref=ref,item=str(i),score=str(value),higher_is_better=direction)
        obs,_ = root.dataset_observations("live-md",members,mapping,labels)
        self.assertEqual([r["human_score"] for r in obs],[-2.,-1.,2.,1.])


if __name__ == "__main__":
    unittest.main()
