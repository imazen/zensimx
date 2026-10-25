"""TRAIN-only ingress, native direction and pair-local hard-label semantics."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import pyarrow as pa
import pyarrow.parquet as pq

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
        f.cell["width"] = 1853
        f.authority["schema"] = "subjective-training-authority-v2"
        f.record.update(name="subjective_rank_fixture", train_weight=1.)
        f.groups = [("subjective_rank_fixture",f.path,1.,0.,"withinref,rank")]
        f.decl.update(schema="subjective-training-leg-v2",nominal_weight=0.5,formula_revision=7,
                      target_transform="orient-native-quality",native_role="TRAIN",loss="hard-pair-preference")
        for k in f.rows:
            k["reference"] = k["source"]
        self.assertEqual(set(f.admit()), {"subjective_rank_fixture"})
        from subjective_training_leg import command_binding
        command_binding(f.authority_path, f.save(), "invented-cell", 1101, 101, 1853, "N", {"hidden":128})
        with self.assertRaisesRegex(ValueError,"binding"):
            command_binding(f.authority_path, f.save(), "invented-cell", 1101, 101, 1825, "N", {"hidden":128})
        for nominal in (0.25, 1.0):
            f.decl["nominal_weight"] = nominal
            f.record["train_weight"] = nominal * 2
            f.groups = [("subjective_rank_fixture", f.path, nominal * 2, 0., "withinref,rank")]
            self.assertEqual(set(f.admit()), {"subjective_rank_fixture"})
        f.decl["nominal_weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "nominal-weight"):
            f.admit()
        f.record["train_weight"] = 1.
        f.groups = [("subjective_rank_fixture", f.path, 1., 0., "withinref,rank")]
        f.rows[1]["family"] = "reserved"
        with self.assertRaisesRegex(ValueError,"family"):
            f.admit()


class DatasetHoldouts(unittest.TestCase):
    def test_all_excluded_never_opens_added_targets(self):
        from subjective_training_leg import dataset_groups
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"wide").mkdir()
            (p/"datasets").mkdir()
            dest = p/"cell"
            dest.mkdir()
            guard = p/"datasets/FAMILY_GUARD.json"
            guard.write_text(json.dumps(dict(schema="dataset-family-guard-v1", inherited_leaks=0)))
            packet = p/"datasets/receipt.json"
            packet.write_text(json.dumps(dict(schema="rev7-dataset-legs-v1", feature_ids=[0],
                family_guard_path=str(guard), family_guard_sha256=root.sha(guard),
                groups=[dict(collection="bapps", path=str(p/"missing-target.parquet"))])))
            (p/"wide/frozen.json").write_text(json.dumps(dict(dataset_legs_sha256=root.sha(packet))))
            groups, binding = dataset_groups(p, [], [0], "full", 0, 1, 2, dest, exclude=("bapps",))
            self.assertEqual(groups, [])
            self.assertIsNone(binding)
            selection = json.loads((dest/"DATASET_GROUP_SELECTION.json").read_text())
            self.assertEqual(selection["omitted_collections"], ["bapps"])
            self.assertEqual(selection["retained_collections"], [])

    def test_dataset_fold_inventory_and_subset_noop_refusal(self):
        from byc_cells import dataset_holdout_collections
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"datasets").mkdir()
            (p/"datasets/receipt.json").write_text(json.dumps(dict(groups=[dict(collection="pipal")])))
            self.assertEqual(dataset_holdout_collections(p, "pipal"), (("pipal",), ("pipal",)))
            with self.assertRaisesRegex(ValueError, "no-op"):
                dataset_holdout_collections(p, "pairs")
            scored, excluded = dataset_holdout_collections(p, "full")
            self.assertEqual(len(scored), 10)
            self.assertEqual(excluded, ())
            from subjective_training_leg import DATASET_FOLDS
            (p/"datasets/receipt.json").write_text(json.dumps(dict(groups=[
                dict(collection=c) for cs in DATASET_FOLDS.values() for c in cs])))
            with self.assertRaisesRegex(ValueError,"subset arm"):
                dataset_holdout_collections(p,"full")

    def test_hard_pair_tie_credit(self):
        from byc_cells import dataset_train_statistic
        got = dataset_train_statistic(["a","a","b","b","c","c"], [1.,0.]*3,
                                      [4.,3.,2.,2.,0.,1.], True)
        self.assertEqual(got["value"], 0.5)
        self.assertEqual(got["predicted_ties"], 1)
        with self.assertRaisesRegex(ValueError, "exactly"):
            dataset_train_statistic(["a","a"], [0.3,0.7], [1.,2.], True)

    def test_diagnostic_root_binds_frozen_inventory_and_actual_command(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"byconstruction"))
        from aic_panel import cell_root
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            r=p/"fresh-root"
            (r/"wide").mkdir(parents=True)
            (r/"wide/frozen.json").write_text('{}')
            cell=p/"cell"
            cell.mkdir()
            (cell/"train.log").write_text(f'--group teacher:{r}/wide/main/real/table.parquet\n')
            (cell/"DATASET_GROUP_SELECTION.json").write_text(json.dumps(dict(root=str(r),
                root_sha256=root.sha(r/"wide/frozen.json"))))
            self.assertEqual(cell_root(cell,"new-mix",""),r)
            (cell/"train.log").write_text('--group teacher:/another/root/wide/table.parquet\n')
            with self.assertRaisesRegex(ValueError,"pin/command"):
                cell_root(cell,"new-mix","")
            (cell/"train.log").write_text(f'--group teacher:{r}/wide/table.parquet\n')
            (r/"wide/frozen.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,"pin/command"):
                cell_root(cell,"new-mix","")

    def test_mix_rule_freezes_balanced_and_old_winners(self):
        import byc_cells
        from subjective_training_leg import DATASET_FOLDS
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as d:
            study = Path(d)
            (study/"scorecard").mkdir()
            e21 = {arm:dict(seeds=[0,1,2], per_dataset_per_seed_srocc={f:[rho]*3 for f in byc_cells.FOLDS})
                   for arm,rho in (("A",0.9),("B",0.8))}
            (study/"scorecard/e21.json").write_text(json.dumps(e21))
            for arm,value in (("A",0.),("B",1.)):
                for seed in range(3):
                    dest=study/"cells"/arm/f"without_full_s{seed}"
                    dest.mkdir(parents=True)
                    (dest/"byc_record.json").write_text(json.dumps(dict(dataset_train_statistics={
                        c:dict(value=value) for cs in DATASET_FOLDS.values() for c in cs})))
            args=SimpleNamespace(study=str(study), arms="A,B")
            with patch.object(byc_cells,"CELLS",study/"cells"), redirect_stdout(io.StringIO()):
                byc_cells.cmd_mix_select(args)
                with self.assertRaisesRegex(ValueError,"preserve"):
                    byc_cells.cmd_mix_select(args)
            frozen=json.loads((study/"TRAIN_SELECTION.json").read_text())
            self.assertEqual(frozen["finalists"],["B","A"])
            self.assertFalse(frozen["eval_labels_used"])

    def test_invalid_builder_parameters_refuse_before_inputs(self):
        for weight, subset in [(0., None), (float("nan"), None), (0.5,"pipal,pipal"), (0.5,"unknown")]:
            with self.assertRaisesRegex(ValueError, "positive finite"):
                root.cmd_datasets(SimpleNamespace(nominal_weight=weight, collections=subset))


class DatasetTargets(unittest.TestCase):
    def test_report_counts_informative_pairs_and_ledgers_derived_targets(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"datasets").mkdir()
            table = p/"mos.parquet"
            pq.write_table(pa.table(dict(ref_basename=["r"]*3,human_score=[1.,1.,2.])),table)
            keys = p/"keys.json"
            keys.write_text(json.dumps([dict(family="f")]*3))
            Path(str(table)+".manifest.json").write_text('{"nominal_weight":0.5}')
            ledger = p/"LABEL_READS.json"
            ledger.write_text('{"role":"TRAIN","reads":[]}')
            group = dict(collection="fixture",path=str(table),keys_path=str(keys),rows=3,membership_rows=3,
                         pairs=3,train_weight=0.75,dropped={})
            (p/"datasets/receipt.json").write_text(json.dumps(dict(groups=[group],label_reads_path=str(ledger))))
            output = p/"counts.json"
            root.cmd_dataset_report(SimpleNamespace(root=str(p),out=str(output)))
            count = json.loads(output.read_text())["fixture"]
            self.assertEqual((count["candidate_pairs"],count["informative_pairs"],count["exact_mos_tied_pairs"]),(3,2,1))
            read = json.loads(ledger.read_text())["reads"][0]
            self.assertEqual((read["rows"],read["state"]),([0,1,2],"read"))

    def test_only_selected_train_lines_are_parsed_and_ledgered(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"labels.tsv"
            p.write_text('item\tscore\ntrain\t3.5\n"malformed non-TRAIN payload\n')
            ledger = Path(d)/"LABEL_READS.json"
            self.assertEqual(root.dataset_targets(p,{0},["score"],ledger,"fixture"), {0:{"score":"3.5"}})
            read = json.loads(ledger.read_text())["reads"][0]
            self.assertEqual((read["rows"],read["columns"],read["state"]),([0],["score"],"read"))
            self.assertEqual(read["parsed_columns"],["item","score"])

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
