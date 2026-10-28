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



class SelectionRebind(unittest.TestCase):
    """`ring_root.py selection-rebind`: metadata-only binding of a dataset-leg root to another declaration."""

    def fixture(self, directory, nan_column=None):
        import numpy as np
        from v2_common import derived_declaration
        fx1 = derived_declaration("fx1")[2]["direct"]
        p = Path(directory)/"parent"
        (p/"wide/main/real").mkdir(parents=True)
        (p/"wide/main/real/receipt.json").write_text(json.dumps(dict(width=1853)))
        (p/"datasets").mkdir()
        rows = 4
        cols = {f"f{i}": pa.array(np.full(rows, 0.25 + i * 1e-4, dtype=np.float32)) for i in range(1853)}
        if nan_column is not None:
            cols[f"f{nan_column}"] = pa.array(np.full(rows, np.nan, dtype=np.float32))
        cols["ref_basename"] = pa.array(["a", "a", "b", "b"])
        cols["human_score"] = pa.array([1.0, 2.0, 3.0, 4.0])
        table = p/"datasets/pipal.parquet"
        pq.write_table(pa.table(cols), table)
        keys = p/"datasets/pipal.keys.json"
        keys.write_text(json.dumps([dict(row=i) for i in range(rows)]))
        manifest = Path(str(table)+".manifest.json")
        manifest.write_text(json.dumps(dict(schema="subjective-training-leg-v2", feature_ids=fx1,
                                            admission_root=str(p), table_sha256=root.sha(table))))
        guard = p/"datasets/FAMILY_GUARD.json"
        guard.write_text(json.dumps(dict(schema="dataset-family-guard-v1", inherited_leaks=0)))
        packet = p/"datasets/receipt.json"
        packet.write_text(json.dumps(dict(schema="rev7-dataset-legs-v1", feature_ids=fx1, producer_commit="x",
            family_guard_path=str(guard), family_guard_sha256=root.sha(guard),
            groups=[dict(collection="pipal", name="subjective_rank_pipal", train_weight=0.5, val_weight=0.0,
                         mode="withinref,rank", subjective=True, path=str(table), table_sha256=root.sha(table),
                         manifest_sha256=root.sha(manifest), keys_path=str(keys), keys_sha256=root.sha(keys),
                         rows=rows)])))
        aux = {f"datasets/{f.name}": root.sha(f) for f in (table, keys, manifest, guard)}
        (p/"wide/frozen.json").write_text(json.dumps(dict(dataset_legs_sha256=root.sha(packet), auxiliary_files=aux)))
        return p

    def rebind(self, parent, out, pin=None, declaration="fx2"):
        args = SimpleNamespace(root=str(parent), out=str(out), declaration=declaration,
                               parent_sha256=pin or root.sha(parent/"wide/frozen.json"))
        return root.cmd_selection_rebind(args)

    def test_rebind_rewrites_only_declarations_and_pins(self):
        from v2_common import derived_declaration
        fx2 = derived_declaration("fx2")[2]["direct"]
        with tempfile.TemporaryDirectory() as d:
            parent = self.fixture(d)
            before = {f: root.sha(f) for f in parent.rglob("*") if f.is_file()}
            out = Path(d)/"fx2"
            self.assertEqual(self.rebind(parent, out), 0)
            self.assertEqual(before, {f: root.sha(f) for f in parent.rglob("*") if f.is_file()})
            self.assertEqual(root.sha(out/"datasets/pipal.parquet"), root.sha(parent/"datasets/pipal.parquet"))
            packet = json.loads((out/"datasets/receipt.json").read_text())
            self.assertEqual(packet["feature_ids"], fx2)
            group = packet["groups"][0]
            self.assertEqual(group["path"], str(out/"datasets/pipal.parquet"))
            manifest = json.loads(Path(group["path"]+".manifest.json").read_text())
            self.assertEqual((manifest["feature_ids"], manifest["admission_root"]), (fx2, str(out)))
            self.assertEqual(group["manifest_sha256"], root.sha(group["path"]+".manifest.json"))
            frozen = json.loads((out/"wide/frozen.json").read_text())
            self.assertEqual(frozen["dataset_legs_sha256"], root.sha(out/"datasets/receipt.json"))
            for rel, pin in frozen["auxiliary_files"].items():
                self.assertEqual(pin, root.sha(out/rel), rel)
            self.assertFalse(packet["selection_rebind"]["labels_read"])

    def test_rebind_refuses_changed_parent_and_unmeasured_columns(self):
        with tempfile.TemporaryDirectory() as d:
            parent = self.fixture(d)
            with self.assertRaisesRegex(ValueError, "registered parent"):
                self.rebind(parent, Path(d)/"a", pin="0"*64)
            self.assertFalse((Path(d)/"a").exists())
        with tempfile.TemporaryDirectory() as d:
            parent = self.fixture(d, nan_column=0)  # scale-0 X SSIM mean: in fx2, not in fx1
            with self.assertRaisesRegex(ValueError, "not measured"):
                self.rebind(parent, Path(d)/"b")
            self.assertFalse((Path(d)/"b").exists())

    def test_fitter_declaration_must_match_the_rebound_root(self):
        from subjective_training_leg import dataset_groups
        from v2_common import derived_declaration
        fx1 = derived_declaration("fx1")[2]["direct"]
        fx2 = derived_declaration("fx2")[2]["direct"]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"fx2"
            self.rebind(self.fixture(d), out)
            dest = Path(d)/"cell"
            dest.mkdir()
            with self.assertRaisesRegex(ValueError, "leg recipe differs"):
                dataset_groups(out, [], fx1, "full", 0, 1101, 101, dest)
            # An inherited group (its manifest is read before the check) must not displace the token.
            inherited = Path(d)/"inherited.parquet"
            pq.write_table(pa.table({"ref_basename": ["a"], "human_score": [1.0]}), inherited)
            Path(str(inherited)+".manifest.json").write_text(json.dumps(dict(rows_kept=1)))
            groups = [("human", inherited, 1.0, 0.0, "withinref,rank")]
            with self.assertRaisesRegex(ValueError, "pinned direct columns"):
                dataset_groups(out, groups, fx2, "full", 0, 1101, 101, dest, derived_inputs="fx1")
            with self.assertRaisesRegex(ValueError, "leg recipe differs"):
                dataset_groups(out, groups, fx1, "full", 0, 1101, 101, dest, derived_inputs="fx2")

class EffectiveSampleMasses(unittest.TestCase):
    def test_selection_requires_gain_noninferiority_and_every_global_cell(self):
        from byc_cells import neff_decision
        arms = {a:{m:dict(per_seed=[0.8,0.8,0.8]) for m in ("old","new","balanced")}
                for a in ("Mlic","MlicNeff")}
        # A tie cannot replace the control.
        self.assertIsNone(neff_decision(arms,[0]*21)["finalist"])
        arms["MlicNeff"]["balanced"]["per_seed"] = [0.81]*3
        self.assertEqual(neff_decision(arms,[0]*21)["finalist"],"MlicNeff")
        self.assertIsNone(neff_decision(arms,[0]*20+[1])["finalist"])
        arms["MlicNeff"]["old"]["per_seed"] = [0.79]*3
        self.assertFalse(neff_decision(arms,[0]*21)["checks"]["old_noninferiority"])
        with self.assertRaisesRegex(ValueError,"twenty-one"):
            neff_decision(arms,[0]*20)
        arms["MlicNeff"]["old"]["per_seed"][0] = float("nan")
        with self.assertRaisesRegex(ValueError,"finite"):
            neff_decision(arms,[0]*21)

    def populations(self):
        return dict(pipal=(23200,200), mdid=(800,10), **{
            "live-md":(210,7), "cid-iq":(330,11), "cid-oa":(143,11),
            "irccyn-ivc":(105,5), "colourlab-rating-variability-2022":(320,5),
            "bapps":(223068,111363)})

    def test_proposed_design_effect_preserves_total_and_unknown_parent_mass(self):
        masses = root.effective_sample_masses(self.populations())
        self.assertAlmostEqual(sum(v["nominal_weight"] for v in masses.values()), 4.)
        self.assertEqual(masses["bapps"]["nominal_weight"], 0.5)
        self.assertIsNone(masses["bapps"]["effective_samples"])
        self.assertAlmostEqual(masses["live-md"]["effective_samples"], 210/10.57)
        self.assertAlmostEqual(masses["irccyn-ivc"]["effective_samples"], 105/7.6)
        self.assertGreater(masses["pipal"]["nominal_weight"], masses["mdid"]["nominal_weight"])
        invalid = self.populations()
        invalid["vcl-fer"] = (10,2)
        with self.assertRaisesRegex(ValueError, "eight Mlic"):
            root.effective_sample_masses(invalid)

    def test_root_clone_preserves_payloads_and_coalesces_ranking_groups_by_family(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory)/"source", Path(directory)/"out"
            (source/"datasets").mkdir(parents=True)
            (source/"wide/main/real").mkdir(parents=True)
            receipt = source/"wide/main/real/receipt.json"
            receipt.write_text(json.dumps(dict(admission_view=dict(immutable_input_roots=[str(source)]))))
            groups = []
            for collection in self.populations():
                path = source/f"datasets/{collection}.parquet"
                path.write_bytes(b"opaque-frozen-payload-no-label-parser")
                keys = path.with_suffix(".keys.json")
                rows = [dict(row=i,source=f"group-{i//2}",family="one-image",role="fit",tier="T2")
                        for i in range(4)]
                keys.write_text(json.dumps(rows))
                dp = Path(str(path)+".manifest.json")
                dp.write_text(json.dumps(dict(nominal_weight=.5,native_role="TRAIN",admission_root=str(source))))
                groups.append(dict(collection=collection,path=str(path),keys_path=str(keys),rows=4,
                                   source_counts={"group-0":2,"group-1":2},train_weight=1.,
                                   table_sha256=root.sha(path),keys_sha256=root.sha(keys),manifest_sha256=root.sha(dp)))
            packet = source/"datasets/receipt.json"
            packet.write_text(json.dumps(dict(groups=groups)))
            auxiliary = {str(p.relative_to(source)):root.sha(p) for p in source.rglob("*") if p.is_file()}
            frozen = source/"wide/frozen.json"
            frozen.write_text(json.dumps(dict(dataset_legs_sha256=root.sha(packet),auxiliary_files=auxiliary)))
            original = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}
            policy = root.reweight_dataset_root(source,out)
            self.assertEqual(policy["collections"]["live-md"]["families"],1)
            for rel, payload in original.items():
                self.assertEqual((source/rel).read_bytes(),payload)
            for group in json.loads((out/"datasets/receipt.json").read_text())["groups"]:
                self.assertEqual(root.sha(group["path"]),group["table_sha256"])
                self.assertEqual(root.sha(group["keys_path"]),group["keys_sha256"])
                self.assertTrue(Path(group["path"]).is_relative_to(out))
            # Drift must refuse before creating a destination; never repair an input pin.
            Path(groups[0]["path"]).write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "changed"):
                root.reweight_dataset_root(source,Path(directory)/"refused")
            self.assertFalse((Path(directory)/"refused").exists())


class DatasetHoldouts(unittest.TestCase):
    def test_e21_family_lookup_binds_full_keys_and_native_reference_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            real, bank = p/"real", p/"bank"
            real.mkdir()
            (bank/"native").mkdir(parents=True)
            source = bank/"native/keys.parquet"
            pq.write_table(pa.table(dict(pair_key=["a","b"], ref_pixels_sha256=["reference-a","reference-b"])),source)
            (bank/"native/_MANIFEST.json").write_text(json.dumps(dict(keys_sha256=root.sha(source))))
            keys = real/"kadid.keys.parquet"
            pq.write_table(pa.table(dict(pair_key=["b"], member_set=["native"])), keys)
            receipt = dict(legs=dict(kadid=dict(full=dict(keys_sha256=root.sha(keys)))))
            self.assertEqual(root.dataset_e21_reference_hashes(real,receipt,bank,"kadid",lambda p:p), {"reference-b"})
            receipt["legs"]["kadid"]["full"]["keys_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError,"reference-key"):
                root.dataset_e21_reference_hashes(real,receipt,bank,"kadid",lambda p:p)

    def linked_fixture(self, directory):
        from subjective_training_leg import DATASET_FOLDS
        p = Path(directory)
        (p/"wide").mkdir()
        (p/"datasets").mkdir()
        child = "colourlab-cognitive-science-2022"
        link = dict(schema="dataset-linked-subset-omissions-v1", labels_read=False,
                    parent_frozen_sha256="parent", added_collections=[child],
                    folds={f:dict(excluded_subset_collections=[child] if f in ("kadid", "mos7") else [])
                           for f in ("kadid", "tid2013", "konfig", "cid22_a25", *DATASET_FOLDS)})
        lp = p/"datasets/linked-subset-omissions.json"
        lp.write_text(json.dumps(link))
        packet = dict(subset_policy_sha256="registered", groups=[dict(collection="pipal"),
            dict(collection="colourlab-rating-variability-2022"), dict(collection=child, protocol="labels_experiment2.tsv")],
            linked_subset_omissions=dict(path="datasets/linked-subset-omissions.json", sha256=root.sha(lp),
                                         parent_frozen_sha256="parent"))
        pp = p/"datasets/receipt.json"
        pp.write_text(json.dumps(packet))
        (p/"wide/frozen.json").write_text(json.dumps(dict(dataset_legs_sha256=root.sha(pp),
            auxiliary_files={"datasets/linked-subset-omissions.json":root.sha(lp)})))
        return p, child, packet

    def test_linked_subset_omits_entire_child_in_new_and_e21_folds(self):
        from byc_cells import dataset_holdout_collections
        from subjective_training_leg import DATASET_FOLDS
        with tempfile.TemporaryDirectory() as d:
            p, child, _ = self.linked_fixture(d)
            scored, omitted = dataset_holdout_collections(p, "mos7")
            self.assertEqual(scored, DATASET_FOLDS["mos7"])
            self.assertEqual(omitted, (*DATASET_FOLDS["mos7"], child))
            self.assertEqual(dataset_holdout_collections(p, "kadid"), ((), (child,)))
            self.assertEqual(dataset_holdout_collections(p, "tid2013"), ((), ()))
            self.assertEqual(dataset_holdout_collections(p, "pipal"), (("pipal",), ("pipal",)))

    def test_linked_subset_refuses_changed_or_absent_authority(self):
        from subjective_training_leg import linked_subset_exclusions
        with tempfile.TemporaryDirectory() as d:
            p, child, packet = self.linked_fixture(d)
            lp = p/"datasets/linked-subset-omissions.json"
            lp.write_text(lp.read_text().replace(child, "another-collection"))
            with self.assertRaisesRegex(ValueError, "pin changed"):
                linked_subset_exclusions(p, "kadid")
            del packet["linked_subset_omissions"]
            (p/"datasets/receipt.json").write_text(json.dumps(packet))
            with self.assertRaisesRegex(ValueError, "require.*authority"):
                linked_subset_exclusions(p, "kadid")

    def test_linked_admission_refuses_retained_child_before_target_reads(self):
        from subjective_training_leg import dataset_groups, DATASET_FOLDS
        with tempfile.TemporaryDirectory() as d:
            p, child, packet = self.linked_fixture(d)
            guard = p/"datasets/guard.json"
            guard.write_text(json.dumps(dict(schema="dataset-family-guard-v1", inherited_leaks=0)))
            packet.update(schema="rev7-dataset-legs-v1",feature_ids=[0],family_guard_path=str(guard),
                          family_guard_sha256=root.sha(guard))
            pp = p/"datasets/receipt.json"
            pp.write_text(json.dumps(packet))
            frozen = json.loads((p/"wide/frozen.json").read_text())
            frozen["dataset_legs_sha256"] = root.sha(pp)
            (p/"wide/frozen.json").write_text(json.dumps(frozen))
            for route, excluded in [("kadid",()), ("full",DATASET_FOLDS["mos7"])]:
                with self.assertRaisesRegex(ValueError,"required linked subset"):
                    dataset_groups(p, [], [0], route, 0, 1101, 101, p, exclude=excluded)
            self.assertFalse((p/"LABEL_READS.json").exists())

    def test_linked_e21_argument_reaches_the_existing_lodo_owner(self):
        import byc_cells
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            recipe = p/"root"
            recipe.mkdir()
            recipe, child, _ = self.linked_fixture(recipe)
            cols = p/"columns.txt"
            cols.write_text("1,2")
            def inspect_command(command, **kwargs):
                self.assertEqual(command[1], str(byc_cells.HERE/"v2_lodo_mlp.py"))
                self.assertEqual(command[command.index("--heldout")+1], "kadid")
                self.assertEqual(command[command.index("--exclude-datasets")+1], child)
                raise RuntimeError("command captured before trainer startup")
            with patch.object(byc_cells,"ROOT",recipe), patch.object(byc_cells,"CELLS",p/"cells"), \
                 patch.object(byc_cells,"COLS",cols), patch.dict(byc_cells.os.environ,{"BYC_SCRATCH":str(p/"scratch")}), \
                 patch.object(byc_cells.subprocess,"run",inspect_command):
                with self.assertRaisesRegex(RuntimeError,"command captured"):
                    byc_cells.run_cell("mono:thr:mp32", "kadid", 0)

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

    def test_licence_subset_scores_entire_omitted_groups(self):
        from byc_cells import dataset_holdout_collections, VALIDATE_ONLY_NEW_LEGS
        from subjective_training_leg import DATASET_FOLDS
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"datasets").mkdir()
            trained = {c for cs in DATASET_FOLDS.values() for c in cs} - set(VALIDATE_ONLY_NEW_LEGS)
            self.assertEqual(len(trained), 8)
            (p/"datasets/receipt.json").write_text(json.dumps(dict(groups=[
                dict(collection=c) for c in sorted(trained)])))
            scored = set()
            for fold in ("pipal", "pairs", "mos7"):
                collections, excluded = dataset_holdout_collections(p, fold)
                self.assertTrue(trained.intersection(excluded))
                self.assertEqual(set(collections), set(DATASET_FOLDS[fold]))
                scored.update(collections)
            self.assertTrue(set(VALIDATE_ONLY_NEW_LEGS) <= scored)
            self.assertEqual(len(scored), 11)

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
            settings = {arm:(0.7,0.1) for arm in byc_cells.MIX_ARMS}
            settings.update(Mdata=(0.9,0.), Mlic=(0.8,1.))
            e21 = {arm:dict(seeds=[0,1,2], per_dataset_per_seed_srocc={f:[rho]*3 for f in byc_cells.FOLDS})
                   for arm,(rho,_) in settings.items()}
            (study/"scorecard/e21.json").write_text(json.dumps(e21))
            policy_manifest = study/"policy-fixture.tsv"
            policy_manifest.write_text("dataset\tpolicy\n")
            (study/"LICENCE_POLICY.json").write_text(json.dumps(dict(
                excluded_from_Mlic=list(byc_cells.VALIDATE_ONLY_NEW_LEGS), eligible_new_leg_arms=["Mlic"],
                copied_manifest=str(policy_manifest), source_sha256=root.sha(policy_manifest))))
            for arm,(_,value) in settings.items():
                for seed in range(3):
                    dest=study/"cells"/arm/f"without_full_s{seed}"
                    dest.mkdir(parents=True)
                    (dest/"byc_record.json").write_text(json.dumps(dict(dataset_train_statistics={
                        c:dict(value=value) for cs in DATASET_FOLDS.values() for c in cs})))
            args=SimpleNamespace(study=str(study), arms=",".join(byc_cells.MIX_ARMS))
            with patch.object(byc_cells,"CELLS",study/"cells"), redirect_stdout(io.StringIO()):
                byc_cells.cmd_mix_select(args)
                with self.assertRaisesRegex(ValueError,"preserve"):
                    byc_cells.cmd_mix_select(args)
            frozen=json.loads((study/"TRAIN_SELECTION.json").read_text())
            self.assertEqual(frozen["finalists"],["Mlic","Mdata"])
            self.assertFalse(frozen["eval_labels_used"])
            self.assertTrue(frozen["arms"]["Mlic"]["eligible_under_new_leg_licence_policy"])
            self.assertFalse(frozen["arms"]["Mdata"]["eligible_under_new_leg_licence_policy"])
            self.assertEqual(frozen["input_sha256"][str(policy_manifest)], root.sha(policy_manifest))
            policy_manifest.write_text("changed after selection")
            with self.assertRaisesRegex(ValueError, "TRAIN selection input changed"):
                byc_cells.cmd_mix_freeze(args)

    def test_invalid_builder_parameters_refuse_before_inputs(self):
        for weight, subset in [(0., None), (float("nan"), None), (0.5,"pipal,pipal"), (0.5,"unknown")]:
            with self.assertRaisesRegex(ValueError, "positive finite"):
                root.cmd_datasets(SimpleNamespace(nominal_weight=weight, collections=subset))


class DatasetTargets(unittest.TestCase):
    def test_cognitive_parent_join_refuses_native_and_identifier_drift(self):
        member = dict(input_table="labels_experiment2.tsv", row_index=7,
                      item_path="images/003_05_01.jpg", reference_path="images/003.jpg")
        key = (member["input_table"], 7)
        label = dict(item=member["item_path"], ref="", score="3", higher_is_better="true")
        projection = [dict(input_table=key[0], row_index=7, item=label["item"], ref=member["reference_path"])]
        parent = [dict(item_path=label["item"], reference_path=member["reference_path"])]
        joined = root.subset_parent_binding([member], {key: label}, projection, parent)
        self.assertEqual(joined[key]["ref"], "images/003.jpg")
        self.assertEqual(label["ref"], "")
        for labels, projected, parents in [
                ({key: dict(label, ref="another.jpg")}, projection, parent),
                ({key: label}, [dict(projection[0], ref="another.jpg")], parent),
                ({key: label}, projection, [dict(parent[0], reference_path="another.jpg")]),
                ({key: dict(label, item="another.jpg")}, projection, parent)]:
            with self.assertRaisesRegex(ValueError, "reference binding"):
                root.subset_parent_binding([member], labels, projected, parents)

    def test_dibr_pairs_keep_local_direction_and_native_reference_guard(self):
        members, mapping, labels = [], [], {}
        for i, probability in enumerate((0.9, 0.1)):
            members.append(dict(input_table="pairs.tsv", row_index=i, reference_path="view",
                                a_path="a", b_path="b", family_id=1))
            labels[("pairs.tsv", i)] = dict(ref="view", a="a", b="b", p_a=str(probability))
            for side in ("a", "b"):
                mapping.append(dict(input_table="pairs.tsv", membership_row_index=i, role="TRAIN",
                                    skip_reason=None, side=side, pair_key=side))
        obs, _ = root.dataset_observations("irccyn-ivc-dibr-images", members, mapping, labels)
        self.assertEqual([r["human_score"] for r in obs], [1., 0., 0., 1.])
        self.assertNotEqual(obs[0]["ref_basename"], obs[2]["ref_basename"])
        for i, (na, nb) in enumerate(((30, 13), (13, 30))):
            labels[("pairs.tsv", i)].update(p_a="", n_a=str(na), n_b=str(nb), n_tie="0", n="43")
        counted, _ = root.dataset_observations("irccyn-ivc-dibr-images", members, mapping, labels)
        self.assertEqual(counted, obs)
        labels[("pairs.tsv", 1)]["n"] = "44"
        with self.assertRaisesRegex(ValueError, "sum"):
            root.dataset_observations("irccyn-ivc-dibr-images", members, mapping, labels)
        labels[("pairs.tsv", 1)]["n"] = "43"
        labels[("pairs.tsv", 1)]["ref"] = "another-view"
        with self.assertRaisesRegex(ValueError, "reference"):
            root.dataset_observations("irccyn-ivc-dibr-images", members, mapping, labels)

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

    def test_native_two_vote_observations_keep_disagreements_and_reject_invented_counts(self):
        members, mapping, labels = [], [], {}
        for i,na in enumerate((2,0,1)):
            members.append(dict(input_table="pairs.tsv",row_index=i,reference_path="ref",a_path="a",b_path="b",family_id=7))
            labels[("pairs.tsv",i)] = dict(ref="ref",a="a",b="b",p_a=str(na/2),
                                           n_a=str(na),n_b=str(2-na),n_tie="0",n="2")
            for side in ("a","b"):
                mapping.append(dict(input_table="pairs.tsv",membership_row_index=i,role="TRAIN",skip_reason=None,
                                    side=side,pair_key=f"{i}{side}"))
        obs,dropped = root.dataset_observations("bapps",members,mapping,labels,votes=True)
        self.assertEqual([o["human_score"] for o in obs],[1.,0.,0.,1.,0.5,0.5])
        self.assertNotIn("exact_probability_tie",dropped)
        self.assertEqual(len({o["ref_basename"] for o in obs}),3)
        self.assertEqual({o["family"] for o in obs},{"7"})
        for field,value in (("n_a","0"),("n_tie","1"),("n","3"),("n_a","1.5"),("p_a","0.75")):
            original = labels[("pairs.tsv",2)][field]
            labels[("pairs.tsv",2)][field] = value
            with self.assertRaisesRegex(ValueError,"BAPPS"):
                root.dataset_observations("bapps",members,mapping,labels,votes=True)
            labels[("pairs.tsv",2)][field] = original
        del labels[("pairs.tsv",2)]["n"]
        with self.assertRaisesRegex(ValueError,"integer vote counts"):
            root.dataset_observations("bapps",members,mapping,labels,votes=True)

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
