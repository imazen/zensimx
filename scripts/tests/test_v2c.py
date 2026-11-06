#!/usr/bin/env python3
"""v2-canon tooling (CANONTAB lane): builder pieces, the sealed-path guard, the confirmatory read on synthetic labels.

    python3 -m unittest scripts/tests/test_v2c.py

The confirmatory read is tested ONLY on synthetic labels: a planted effect it must find, a null it must not, and the
orientation rule it must apply. Needs the `panel` binary (ZEN_PANEL_BIN, default the fleet v8 panel).
"""
import builtins
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TMP = Path.home() / "tmp"  # never /tmp
TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
os.environ.setdefault("ZEN_PANEL_BIN", "/var/tmp/fitv2/bin-v2/panel")
import v2_common  # noqa: E402
import v2c_wide as w  # noqa: E402


def f64_bank_set(bank: Path, name: str, rows: int, seed: int, identical=(2,), extra_manifest=None):
    """A synthetic Rev4 bank set: keys.parquet, features.parquet (float64, f0..f1824), _MANIFEST.json."""
    rng = np.random.default_rng(seed)
    d = bank / name
    d.mkdir(parents=True)
    keys = pd.DataFrame({
        "pair_key": [hashlib.sha256(f"{name}{i}".encode()).hexdigest() for i in range(rows)],
        "row_id": np.arange(rows, dtype=np.int64),
        "ref_group": [f"ref{i % 3}" for i in range(rows)],
        "pixels_identical": [i in identical for i in range(rows)], "n_stimuli": np.ones(rows, dtype=np.int32)})
    pq.write_table(pa.Table.from_pandas(keys, preserve_index=False), d / "keys.parquet")
    feats = {"pair_key": keys.pair_key, "row_id": keys.row_id}
    values = rng.normal(size=(rows, w.WIDTH)) * 1e3 + rng.normal(size=(rows, w.WIDTH)) * 1e-9  # not f32-exact
    feats.update({f"f{i}": values[:, i] for i in range(w.WIDTH)})
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(feats), preserve_index=False), d / "features.parquet")
    manifest = {"set": name, "schema": "rev4-featbank-r4-v1", "rows": rows, "feature_width": w.WIDTH,
                "build_commit": w.CANON_BUILD, "binary_sha256": w.CANON_BINARY, "formula_revision": "Rev4",
                "era_label": w.CANON_ERA, "feature_set_id": w.CANON_FEATURE_SET_ID, "dtype": "float64",
                "formula_revision_eras": ["tiercanon", "c3negfold"], "keys_sha256": w.sha(d / "keys.parquet"),
                "features_parquet_sha256": w.sha(d / "features.parquet"), **(extra_manifest or {})}
    (d / "_MANIFEST.json").write_text(json.dumps(manifest))
    return values


class Seeds(unittest.TestCase):
    """Design log E23: seed indices 10-19 extend the streams without moving any existing cell's (init, sample) pair."""

    def test_indices_0_to_9_unchanged(self):
        init = (1101, 1103, 1107, 1109, 1117, 1123, 1129, 1151, 1153, 1163)
        sample = tuple(101 + k * 100_000_000 for k in range(10))
        for fold, held in enumerate(v2_common.SOURCE_ORDER):
            for i in range(10):
                self.assertEqual(v2_common.seeds(held, i), (init[i], sample[(i + fold) % 10]))

    def test_indices_10_to_19_disjoint_and_bounded(self):
        old = {v2_common.seeds(h, i) for h in v2_common.SOURCE_ORDER for i in range(10)}
        new = {v2_common.seeds(h, i) for h in v2_common.SOURCE_ORDER for i in range(10, 20)}
        self.assertEqual(len(new), 50)
        self.assertFalse({a for a, _ in old} & {a for a, _ in new})
        self.assertFalse({b for _, b in old} & {b for _, b in new})
        self.assertTrue(all(b < 2**31 for _, b in new))
        for bad in (-1, 20):
            with self.assertRaises(ValueError):
                v2_common.seeds("kadid", bad)


def rev5_bank_set(bank: Path, name: str, rows: int, seed: int, poke=None):
    """A synthetic Rev5 bank set: requested slots (f0..227, f372..719) finite, every other slot NaN; `poke(values)` may break it."""
    import rev5_bank
    values = f64_bank_set(bank, name, rows, seed, identical=())
    mask = np.zeros(w.WIDTH, dtype=bool)
    for lo, hi in rev5_bank.SLOT_RANGES:
        mask[lo:hi] = True
    values[:, ~mask] = np.nan
    if poke:
        poke(values)
    d = bank / name
    keys = pq.read_table(d / "keys.parquet").to_pandas()
    feats = {"pair_key": keys.pair_key, "row_id": keys.row_id, **{f"f{i}": values[:, i] for i in range(w.WIDTH)}}
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(feats), preserve_index=False), d / "features.parquet")
    man = json.loads((d / "_MANIFEST.json").read_text())
    man.update({"schema": rev5_bank.SCHEMA, "formula_revision": "Rev5", "era_label": "rev5test", "feature_set_id": "fsid5",
                "binary_sha256": "bin5", "build_commit": "build5", "formula_revision_eras": ["rev5"],
                "requested_slot_ranges": [list(r) for r in rev5_bank.SLOT_RANGES],
                "features_parquet_sha256": w.sha(d / "features.parquet")})
    (d / "_MANIFEST.json").write_text(json.dumps(man))
    return values, mask


class Rev5Tables(unittest.TestCase):
    """Spec rev5_spec_2026-10-04.md §6: a Rev5 bank's absent slots are NaN, never a value; tables keep the Rev4 width."""

    def setUp(self):
        self.saved = w.PROFILE
        w.PROFILE = w.rev5_profile("rev5test", "fsid5", "bin5", "build5", 1853)

    def tearDown(self):
        w.PROFILE = self.saved

    def test_loads_pads_and_keeps_requested_bits(self):
        with tempfile.TemporaryDirectory() as t:
            values, mask = rev5_bank_set(Path(t), "aic4", 6, 3)
            b = w.load_bank_set(Path(t), "aic4", [])
            self.assertEqual(b.X.shape, (6, 1853))
            self.assertTrue(np.array_equal(b.X[:, :w.WIDTH][:, mask], values[:, mask].astype(np.float32)))
            self.assertTrue(np.isnan(b.X[:, :w.WIDTH][:, ~mask]).all() and np.isnan(b.X[:, w.WIDTH:]).all())
            self.assertEqual(w.total_width([]), 1853)

    def test_refusals(self):
        def value_in_absent(v):
            v[0, 900] = 1.0
        def nan_in_requested(v):
            v[1, 400] = np.nan
        for poke in (value_in_absent, nan_in_requested):
            with tempfile.TemporaryDirectory() as t:
                rev5_bank_set(Path(t), "aic4", 6, 3, poke)
                with self.assertRaises(ValueError):
                    w.load_bank_set(Path(t), "aic4", [])
        with tempfile.TemporaryDirectory() as t:
            rev5_bank_set(Path(t), "aic4", 6, 3)
            w.PROFILE = w.REV4_PROFILE                       # a Rev4 build refuses a Rev5 bank
            with self.assertRaises(ValueError):
                w.load_bank_set(Path(t), "aic4", [])
        w.PROFILE = w.rev5_profile("rev5test", "fsid5", "bin5", "build5", 1853)
        with self.assertRaises(ValueError):                  # Rev4 sidecars would mix revisions
            w.total_width(w.parse_extras(["texgain=1825:3:/x/{set}/t.parquet"]))
        leg = w.Leg("kadid", pd.DataFrame({"ref_basename": ["a"], "pair_key": ["k"]}), np.zeros((1, 1853), np.float32),
                    None, [0.0, 1.0], np.zeros(1), {})
        with self.assertRaises(ValueError):                  # the aux family is a Rev4 layout
            w.family_matrix(leg, "aux", 1853)
        with self.assertRaises(ValueError):
            w.rev5_profile("", "fsid5", "bin5", "build5", None)

    def test_fitter_guard_and_revision_probe(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "leg.parquet"
            x = np.ones((4, 6), np.float32)
            x[:, 4] = np.nan
            pq.write_table(pa.table({f"f{i}": x[:, i] for i in range(6)}), path)
            v2_common.refuse_nonfinite_kept([path], [0, 1, 5])              # NaN outside the keep list is fine
            with self.assertRaises(ValueError):
                v2_common.refuse_nonfinite_kept([path], [1, 4])
            self.assertEqual(v2_common.table_revision(path), 4)              # no manifest: a Rev4-era table
            Path(f"{path}.manifest.json").write_text(json.dumps({"formula_revision": 5}))
            self.assertEqual(v2_common.table_revision(path), 5)


class ExtractorContract(unittest.TestCase):
    """e14.extractor: one owner for the Rev4 and Rev5 extractor contracts used by E14, E15 and the external sets."""

    def test_rev4_unchanged_and_rev5_from_build_record(self):
        import e14_kadis_ordinal as e14
        x4 = e14.extractor(4)
        self.assertEqual((x4["bin"], x4["sha"], x4["era"], x4["args"], x4["env"]), (e14.BIN, e14.BIN_SHA, e14.ERA,
                                                                                   e14.EXTRACT_ARGS, e14.EXTRACT_ENV))
        feats = np.arange(12.0).reshape(2, 6)
        self.assertIs(e14.apply_contract(feats, x4), feats)
        with tempfile.TemporaryDirectory() as t:
            meta = Path(t) / "build_meta.json"
            meta.write_text(json.dumps({"binary": "/x/extract", "binary_sha256": "s5", "era": "rev5era", "build_commit": "b5"}))
            with mock.patch.object(e14, "REV5_META", meta):
                x5 = e14.extractor(5)
            self.assertEqual(x5["env"]["ZENSIM_FORMULA_REV"], "5")
            self.assertIn("basic,peaks,v2", x5["args"])
            wide = np.ones((2, 800))
            out = e14.apply_contract(wide, x5)
            self.assertTrue(np.isnan(out[:, 228:372]).all() and np.isnan(out[:, 720:]).all())
            self.assertTrue((out[:, :228] == 1).all() and (out[:, 372:720] == 1).all())
            wide[0, 400] = np.inf
            with self.assertRaises(ValueError):
                e14.apply_contract(wide, x5)
        with self.assertRaises(ValueError):
            e14.extractor(3)


class Guard(unittest.TestCase):
    def test_sealed_paths_refused(self):
        for bad in ("/var/tmp/rev4-featbank/_sealed", "/x/_sealed/y/labels.parquet", "_sealed_extra/z"):
            with self.assertRaises(PermissionError):
                w.safe_path(bad)
        w.safe_path("/var/tmp/rev4-featbank-r4/aic4/keys.parquet")

    def test_bank_file_names(self):
        with self.assertRaises(ValueError):
            w.bank_file(Path("/b"), "aic4", "labels__human.parquet")
        with self.assertRaises(ValueError):
            w.bank_file(Path("/b"), "_sealed", "keys.parquet")
        self.assertEqual(w.bank_file(Path("/b"), "aic4", "keys.parquet"), Path("/b/aic4/keys.parquet"))

    def test_manifest_pins_refused(self):
        with tempfile.TemporaryDirectory() as t:
            bank = Path(t)
            f64_bank_set(bank, "aic4", 6, 1, extra_manifest={"era_label": "tiercanon"})
            with self.assertRaises(ValueError):
                w.load_bank_set(bank, "aic4", [])


class Builders(unittest.TestCase):
    def test_permutation_matches_restore_data_owner(self):
        import restore_data
        rng = np.random.default_rng(3)
        refs = np.repeat([f"r{i}" for i in range(4)], [9, 7, 5, 6])
        keys = np.array([f"k{i // 2 if i % 5 == 0 else i}" for i in range(len(refs))])
        keys = np.where(np.arange(len(refs)) % 7 == 0, np.roll(keys, 1), keys)  # shared keys inside a reference
        vals = rng.normal(size=(len(refs), 6)).astype(np.float32)
        frame = pd.DataFrame({"ref_basename": refs, "pair_key": keys, **{f"c{j}": vals[:, j] for j in range(6)}})
        owner = frame.copy()
        restore_data._permute_within_reference(owner, [f"c{j}" for j in range(6)], np.ones(len(frame), dtype=bool),
                                               np.random.default_rng(20260931))
        ours = w.permute_within_reference(refs, keys, vals, np.ones(len(refs), dtype=bool), np.random.default_rng(20260931))
        self.assertTrue(np.array_equal(owner[[f"c{j}" for j in range(6)]].to_numpy(), ours))
        self.assertFalse(np.array_equal(vals, ours))

    def test_oracle_columns_are_distance_oriented_and_seeded(self):
        y = np.linspace(0, 100, 400) ** 1.1
        bounds, y01, cols = w.oracle_columns(y, 3)
        self.assertLess(np.corrcoef(cols["oracle_hi"], y01)[0, 1], -0.7)
        self.assertLess(np.corrcoef(cols["oracle_lo"], y01)[0, 1], np.corrcoef(cols["oracle_hi"], y01)[0, 1] + 1)
        rng = np.random.default_rng(v2_common.ORACLE_SEED_BASE + 3)
        sd = float(np.std(y01))
        expect = ((1.0 - y01) + rng.normal(0.0, v2_common.ORACLE_SIGMA["oracle_lo"] * sd, len(y))).astype(np.float32)
        self.assertTrue(np.array_equal(cols["oracle_lo"], expect))

    def test_confirm_tables_exact_cast_zero_labels_and_never_touch_sealed(self):
        seen = []
        real_open, real_file, real_read, real_write = builtins.open, pq.ParquetFile, pq.read_table, pq.write_table
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            bank = t / "bank"
            values = {n: f64_bank_set(bank, n, 9, i, identical=(4,)) for i, n in enumerate(("aic4", "csiq"))}
            sealed = bank / "_sealed"  # a trap beside the bank: any access is a failure
            sealed.mkdir()
            (sealed / "labels.parquet").write_text("trap")

            def read_text_spy(self, *a, **k):
                seen.append(str(self))
                with real_open(self) as stream:
                    return stream.read()

            def spy(real):
                def inner(path, *a, **k):
                    seen.append(str(path))
                    return real(path, *a, **k)
                return inner
            with mock.patch.object(builtins, "open", spy(real_open)), mock.patch.object(pq, "ParquetFile", spy(real_file)), \
                    mock.patch.object(pq, "read_table", spy(real_read)), mock.patch.object(pq, "write_table", spy(real_write)), \
                    mock.patch.object(Path, "read_text", read_text_spy):
                rec = w.build_confirm(bank, t / "out", ["main"], [], None, sets=("aic4", "csiq"))
            self.assertTrue(seen)
            self.assertFalse([p for p in seen if "_sealed" in p], seen)
            for name in ("aic4", "csiq"):
                tab = rec["sets"][name]["tables"]["main"]["real"]
                got = pq.read_table(t / "out" / tab["rel"]).to_pandas()
                keys = pq.read_table((t / "out" / tab["rel"]).parent / f"{name}.keys.parquet").to_pandas()
                self.assertEqual(len(got), 8)                         # the identical key is dropped
                self.assertTrue((got.human_score == 0).all())
                self.assertNotIn("target", keys.columns)
                keep = np.arange(9) != 4
                want = values[name][keep].astype(np.float32)
                have = got[[f"f{i}" for i in range(w.WIDTH)]].to_numpy(np.float32)
                self.assertTrue(np.array_equal(want.view(np.uint32), have.view(np.uint32)))
                self.assertFalse(np.array_equal(values[name][keep], want.astype(np.float64)))  # the cast is real

    def test_extra_families_widen_main_and_aux_zero(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            bank = t / "bank"
            f64_bank_set(bank, "aic4", 7, 5, identical=())
            keys = pq.read_table(bank / "aic4" / "keys.parquet").to_pandas()
            side = pd.DataFrame({"pair_key": keys.pair_key[::-1].to_numpy(),
                                 **{f"f{1825 + j}": np.arange(7, dtype=np.float32) + j for j in range(3)}})
            (t / "aic4").mkdir()
            pq.write_table(pa.Table.from_pandas(side, preserve_index=False), t / "aic4" / "texgain.parquet")
            extras = w.parse_extras([f"texgain=1825:3:{t}/{{set}}/texgain.parquet"])
            self.assertEqual(w.total_width(extras), 1828)
            peers = t / "peers"
            peers.mkdir()
            pq.write_table(pa.Table.from_pandas(pd.DataFrame({"pair_key": keys.pair_key, "gmsd": np.arange(7.0),
                                                              "gmsm": np.arange(7.0) * 2}), preserve_index=False),
                           peers / "aic4.parquet")
            rec = w.build_confirm(bank, t / "out", ["main", "aux"], extras, peers, variants=["real", "p2"], sets=("aic4",))
            main = pq.read_table(t / "out" / rec["sets"]["aic4"]["tables"]["main"]["real"]["rel"]).to_pandas()
            self.assertEqual(main.shape[1], 2 + 1828)
            by_key = dict(zip(keys.pair_key[::-1], range(7)))   # sidecar rows were reversed: join is by key, not position
            self.assertEqual([by_key[k] for k in keys.pair_key], main.f1825.astype(int).tolist())
            aux = pq.read_table(t / "out" / rec["sets"]["aic4"]["tables"]["aux"]["real"]["rel"]).to_pandas()
            self.assertTrue((aux[["f1825", "f1826", "f1827"]] == 0).all().all())
            self.assertEqual(aux.f944.tolist(), list(map(float, range(7))))
            self.assertTrue((aux.f946 == 0).all())                # oracle columns need a label: none here
            perm = pq.read_table(t / "out" / rec["sets"]["aic4"]["tables"]["main"]["p2"]["rel"]).to_pandas()
            self.assertTrue((perm.f5 == main.f5).all())           # bank columns untouched by the control
            self.assertEqual(sorted(perm.f1825), sorted(main.f1825))  # extras permuted jointly within reference
            self.assertNotEqual(perm.f1825.tolist(), main.f1825.tolist())

    def test_extra_arm_keep_lists(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "wide").mkdir()
            (root / "wide" / "extra_arms.json").write_text(json.dumps({
                "schema": "rev4-featpot-v2c-extra-arms-v1", "width": 1830, "arms": {"texgain": [1825, 1826]}}))
            with mock.patch.object(v2_common, "V2", root):
                self.assertEqual(v2_common.parse_spec("texgain~p2"), ("texgain", 2))
                fam, var, keep = v2_common.arm_columns("texgain~p2")
                self.assertEqual((fam, var, keep[-2:]), ("main", "p2", [1825, 1826]))
                self.assertIn("texgain~p3", v2_common.all_specs())
            with mock.patch.object(v2_common, "V2", root / "nope"):
                with self.assertRaises(ValueError):
                    v2_common.parse_spec("texgain")

    def test_root_override_from_argv_and_env(self):
        self.assertEqual(v2_common._root_override(["x", "--root", "/a/b"]), "/a/b")
        self.assertEqual(v2_common._root_override(["x", "--root=/c"]), "/c")
        with mock.patch.dict(os.environ, {"REV4_V2_ROOT": "/e"}):
            self.assertEqual(v2_common._root_override(["x"]), "/e")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REV4_V2_ROOT", None)
            self.assertIsNone(v2_common._root_override(["x"]))


class TrainRecipe(unittest.TestCase):
    def test_shared_hidden_activation_preserves_the_recipe_except_depth(self):
        import v2_lodo_mlp as lodo
        recipe = {"hidden": 128, "mono_all": True, "thresholds": True, "max_pairs": 32}
        args = ([], 1, 2, 3, Path("k"), "N", Path("o"))
        base = lodo.train_command(*args, recipe)
        for activation in ("relu", "smooth-hinge"):
            cmd = lodo.train_command(*args, {**recipe, "shared_hidden_activation": activation})
            index = cmd.index("--n-hidden-layers")
            self.assertEqual(cmd[index:index+4], ["--n-hidden-layers", "2", "--shared-hidden-activation", activation])
            self.assertEqual(cmd[:index] + cmd[index+4:], base)
        for bad in ({"shared_hidden_activation": "relu"}, {**recipe, "shared_hidden_activation": "tanh"}):
            with self.assertRaises(ValueError):
                lodo.train_command(*args, bad)

    def test_train_command_is_the_v2_recipe(self):
        import v2_lodo_mlp as lodo
        cmd = lodo.train_command([("safesyn", Path("/t/s.parquet"), 1.5, 0, "withinref,both"),
                                  ("human_development", Path("/t/h.parquet"), 0, 1.0, "withinref,rank")],
                                 1101, 101, 1825, Path("/k.txt"), "N", Path("/o/best.bin"))
        expect = [str(lodo.TRAINER), "--group", "safesyn:/t/s.parquet:1.5:0:withinref,both",
                  "--group", "human_development:/t/h.parquet:0:1.0:withinref,rank",
                  "--target-column", "human_score", "--target-scale", "1", "--hidden", "32", "--epochs", "120",
                  "--pairs-per-epoch", "50000", "--init-seed", "1101", "--sample-seed", "101", "--pair-sampling",
                  "uniform", "--max-features", "1825", "--keep-features", "/k.txt", "--mse-weight", "1",
                  "--early-stop-patience", "0", "--val-policy", "mean", "--val-aggregate", "geomean3", "--out-dtype",
                  "f32", "--log-every", str(lodo.LOG_EVERY), "--no-auto-eval", "--historical-replay", v2_common.REPLAY,
                  "--out", "/o/best.bin", "--nonneg-distance"]
        self.assertEqual(cmd, expect)
        self.assertNotIn("--nonneg-distance", lodo.train_command([], 1, 2, 3, Path("k"), "F", Path("o")))

    def test_block_specs(self):
        self.assertEqual(v2_common.arm_columns("core")[2], list(range(228)))
        self.assertEqual(v2_common.arm_columns("r0-v2@h32")[2], [c for c in range(944) if not 372 <= c < 720])
        self.assertEqual(v2_common.arm_columns("core+iw@h32:H128")[2], list(range(228)) + list(range(300, 372)))
        fam, _, cols = v2_common.arm_columns("core+csfw")
        self.assertEqual((fam, cols), ("main", list(range(228)) + list(range(944, 956))))
        self.assertEqual(sorted(set(v2_common.arm_columns("r0-basic")[2]) & set(v2_common.arm_columns("r0-peaks")[2])), list(range(228, 944)))
        self.assertEqual(v2_common.arm_columns("core+iw+csfw@h32")[2], list(range(228)) + list(range(300, 372)) + list(range(944, 956)))
        self.assertEqual(v2_common.arm_columns("core+p3+v2")[0], "aux")
        self.assertEqual(v2_common.arm_columns("set:csfw@h32:H64")[2], list(range(944, 956)))
        self.assertEqual(v2_common.arm_columns("set:basic+peaks")[2], v2_common.arm_columns("core")[2])
        self.assertEqual(v2_common.arm_columns("set:v2+c3")[2], list(range(372, 720)) + list(range(1154, 1298)))
        for bad_set in ("set:", "set:rall", "set:v2+v2", "set:p3+c3"):
            with self.assertRaises((ValueError, KeyError)):
                v2_common.parse_spec(bad_set)
                v2_common.arm_columns(bad_set)
        for bad in ("core+rall", "core+all", "r0-nope", "core+nope", "core+c1~p1", "core+iw+iw", "core+p3+c3"):
            with self.assertRaises((ValueError, KeyError)):
                v2_common.parse_spec(bad)
                v2_common.arm_columns(bad)

    def test_block_specs_resolve_without_restore_data(self):
        # The fit program packs no restore_data: set:/core+ specs must resolve candidate arms from the pinned keep lists.
        bank = list(range(944))
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "wide").mkdir()
            (root / "wide" / "keep_lists.json").write_text(json.dumps({"schema": "rev4-featpot-v2-keeplists-v2", "specs": {
                "csfw": {"family": "main", "variant": "real", "keep": bank + list(range(944, 956))},
                "c3": {"family": "main", "variant": "real", "keep": bank + [1200, 1201]}}}))
            with mock.patch.object(v2_common, "V2", root), mock.patch.dict(sys.modules, {"restore_data": None}):
                self.assertEqual(v2_common.arm_columns("set:csfw@h32:H128")[2], list(range(944, 956)))
                self.assertEqual(v2_common.arm_columns("set:v2+c3@h32:H128"), ("main", "real", list(range(372, 720)) + [1200, 1201]))
                self.assertEqual(v2_common.arm_columns("core+iw+csfw")[2], list(range(228)) + list(range(300, 372)) + list(range(944, 956)))
                self.assertEqual(v2_common.arm_columns("set:basic+peaks")[2], list(range(228)))
                self.assertEqual(v2_common.arm_columns("r0-v2")[2], [c for c in bank if not 372 <= c < 720])
                with self.assertRaises(ImportError):  # an arm the pinned lists lack still needs the owner, loudly
                    v2_common.arm_columns("set:c4")

    def test_selection_specs(self):
        import v2_lodo_mlp as lodo
        cols = [3, 228, 400, 1500]
        sid = v2_common.selection_id(cols)
        self.assertEqual(sid, v2_common.selection_id(list(reversed(cols))))
        self.assertTrue(v2_common.block_spec(f"sel:{sid}"))
        self.assertEqual(v2_common.parse_spec(f"sel:{sid}@h32:H128"), (f"sel:{sid}", 0))
        for bad in ("sel:", "sel:XYZ", f"sel:{sid}0", f"sel:{sid.upper()}"):
            self.assertFalse(v2_common.block_spec(bad), bad)
        with self.assertRaises(ValueError):  # the registry cannot name a subset's columns
            v2_common.arm_columns(f"sel:{sid}")
        lists = {"specs": {"c3": {"family": "main", "variant": "real", "keep": [1, 2]}}}
        self.assertEqual(lodo.resolve_keep(f"sel:{sid}", "3,228,400,1500", lists), ("main", "real", cols))
        self.assertEqual(lodo.resolve_keep("c3", None, lists), ("main", "real", [1, 2]))
        for spec, columns in ((f"sel:{sid}", None), (f"sel:{sid}", "3,228,400"), (f"sel:{sid}", "228,3,400,1500"),
                              (f"sel:{sid}", "3,3,228,400,1500"), ("c3", "1,2"), ("nope", None)):
            with self.assertRaises(ValueError, msg=(spec, columns)):
                lodo.resolve_keep(spec, columns, lists)

    def test_e10_reuses_results_under_any_group_order(self):
        import e10_multistart as e10
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            cell = root / "cells" / "set:v2+basic@h32:H128__N" / "without_kadid_s0"
            cell.mkdir(parents=True)
            (cell / "result.json").write_text("{}")
            (root / "cells" / "set:v2+basic@h32:H128:gl1__N").mkdir(parents=True)  # another recipe: never matched
            with mock.patch.object(e10, "V2", root):
                idx = e10.index(":H128")
                self.assertEqual(idx, {frozenset({"v2", "basic"}): ["set:v2+basic@h32:H128"]})
                self.assertEqual(e10.find(idx, ["basic", "v2"], ":H128", "kadid", 0), cell / "result.json")
                self.assertIsNone(e10.find(idx, ["basic", "v2"], ":H128", "kadid", 1))
                self.assertEqual(e10.canonical(["v2", "basic"], ":H128"), "set:basic+v2@h32:H128")

    def test_recipe_tokens(self):
        import v2_lodo_mlp as lodo
        self.assertEqual(v2_common.recipe_of("r0@h32"), {})
        self.assertEqual(v2_common.recipe_of("r0@h32:H128:gl0.0001"), {"hidden": 128, "group_l1": 0.0001})
        self.assertEqual(v2_common.split_weight("screen_main@h32:H64"), ("screen_main", 32.0))
        self.assertEqual(v2_common.recipe_of("r0@h32:H128:gl2.8"), {"hidden": 128, "group_l1": 2.8})   # calibrated E9′ grid
        for bad in ("r0@h32:H4", "r0@h32:x1", "r0@h32:H64:H128", "r0@h32:gl0", "r0@h32:gl200"):
            with self.assertRaises(ValueError):
                v2_common.recipe_of(bad)
        base = lodo.train_command([], 1, 2, 3, Path("k"), "N", Path("o"))
        cmd = lodo.train_command([], 1, 2, 3, Path("k"), "N", Path("o"), {"hidden": 128, "group_l1": 0.0001})
        self.assertEqual(base, lodo.train_command([], 1, 2, 3, Path("k"), "N", Path("o"), {}))   # no tokens: argv unchanged
        self.assertEqual(cmd[cmd.index("--hidden") + 1], "128")
        self.assertEqual(cmd[cmd.index("--group-l1") + 1], "0.0001")
        self.assertEqual([a for a in cmd if a not in ("--group-l1", "0.0001")], [a if a != str(v2_common.HIDDEN) or base[i - 1] != "--hidden" else "128" for i, a in enumerate(base)])

    def test_read_curve_accepts_the_sparse_final_epoch_curve(self):
        import tempfile
        import v2_lodo_mlp as lodo
        self.assertEqual(lodo.LOG_EVERY, 17 if lodo.EPOCH_RULE == "last" else 1)
        epochs = sorted(set(range(0, lodo.EPOCHS, lodo.LOG_EVERY)) | {lodo.EPOCHS - 1})
        line = "  epoch {e:>3} | lr=0.00500 | loss=0.1 | val(geomean3)={v:.4f} (best=0.1) | a: srocc=0.1 | t=1.0s\n"
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "train.log"
            log.write_text("".join(line.format(e=e, v=0.5 + e / 1000) for e in epochs))
            curve = lodo.read_curve(log)
            self.assertEqual(sorted(curve), epochs)
            self.assertEqual(curve[lodo.EPOCHS - 1], 0.5 + (lodo.EPOCHS - 1) / 1000)
            log.write_text("".join(line.format(e=e, v=0.5) for e in epochs[:-1]))
            with self.assertRaises(ValueError):
                lodo.read_curve(log)

    def test_lodo_grids(self):
        import v2c_grid
        cal = v2c_grid.grid_calibration_screen(32.0, "/r")
        self.assertEqual(len(cal), 900)
        self.assertEqual(len({c["name"] for c in cal}), 900)
        names = {c["name"] for c in cal}
        self.assertIn("screen_main@h32__N/without_aic3_s9", names)
        self.assertIn("oracle_lo~p3@h32__F/without_kadid_s0", names)
        self.assertEqual(cal[0]["argv"], ["v2_lodo_mlp.py", "--spec", "r0@h32", "--head", "N", "--heldout", "kadid", "--seed-index", "0",
                                          "--root", "/r"])
        arms = v2c_grid.grid_full_arms(["c1", "csfw"], 32.0, "/r")
        self.assertEqual(len(arms), 800)
        self.assertIn("csfw~p2@h32__N/without_konfig_s5", {c["name"] for c in arms})
        self.assertFalse(any(c["name"].startswith("r0") for c in arms))

    def test_grid_shape(self):
        import v2c_grid
        cells = v2c_grid.cells(["c1", "c3"], 2.0, "/r")
        self.assertEqual(len(cells), (1 + 2 * 4) * 2 * 10)
        self.assertEqual(len({c["name"] for c in cells}), len(cells))
        self.assertEqual(cells[0], {"name": "r0@h2__N/full_s0", "argv": [
            "v2_confirm_fit.py", "--spec", "r0@h2", "--head", "N", "--seed-index", "0", "--root", "/r"]})
        self.assertIn("c3~p3@h2__F/full_s9", {c["name"] for c in cells})


class Screen(unittest.TestCase):
    BANK = Path("/var/tmp/rev4-featbank-r4")

    def test_worst_types_match_filenames_in_the_real_bank(self):
        import v2c_screen as sc
        expect = {"kadid": (r"I\d+_(\d+)_\d+\.png$", {"20", "08", "07", "03", "21"}),
                  "tid2013": (r"I\d+_(\d+)_\d+\.png$", {"17", "18", "14", "12", "23"}),
                  "konfig": (r"SRC\d+_([a-z]+)_\d+\.png$", {"highsharpen", "multinoise", "colordiffusion"})}
        import re
        for source, (rx, types) in expect.items():
            for member in v2_common.SOURCES[source]:
                k = pq.read_table(self.BANK / member / "keys.parquet", columns=["pair_key", "dist_path"]).to_pandas()
                want = np.array([bool(re.search(rx, p, re.I)) and re.search(rx, p, re.I).group(1) in types for p in k.dist_path])
                got = sc.worst_mask(source, sc.codecs_for(source, k.pair_key.to_numpy(), self.BANK))
                self.assertTrue(np.array_equal(want, got), (source, member))
                self.assertGreater(got.sum(), 0, (source, member))
        k = pq.read_table(self.BANK / "aic3" / "keys.parquet", columns=["pair_key"]).to_pandas()
        self.assertEqual(sc.worst_mask("aic3", sc.codecs_for("aic3", k.pair_key.to_numpy()[:5], self.BANK)).sum(), 0)  # no worst types

    def test_union_arms_never_take_a_slot(self):
        import v2c_screen as sc
        imp = {"N": {"all": 0.20, "rall": 0.19, "csfw": 0.03, "c7": 0.02, "c3": 0.05, "b2": 0.01, "a1": 0.04, "p1": 0.0, "c8n": 0.006},
               "F": {}}
        out = sc.select(imp, {"N": {"all": 0.3, "rall": 0.3, "c3": 0.01}})
        self.assertNotIn("all", out["selected"]); self.assertNotIn("rall", out["selected"])
        self.assertNotIn("all", out["rank_by_head_N_importance"])
        self.assertEqual(out["unions_upper_bound"]["all"]["N"], 0.20)
        self.assertEqual(out["selected"][:3], ["c3", "a1", "csfw"])

    def test_selection_rule(self):
        import v2c_screen as sc
        n = {f"f{i}": 1.0 - 0.1 * i for i in range(10)}                      # f0 .. f9 by head-N importance
        t = {f: 0.0 for f in n}
        t.update({"f7": 0.9, "f8": 0.8, "f2": 0.7, "f9": 0.6})               # targeted top 3: f7, f8, f2 (f2 already chosen)
        fi = {f: 0.0 for f in n}
        got = sc.select({"N": n, "F": fi}, {"N": t, "F": fi}, {"f9": (6, 12), "f4": (1, 12)})
        self.assertEqual(got["selected"][:6], [f"f{i}" for i in range(6)])
        self.assertEqual(got["selected"][6:], ["f7", "f8"])                   # cap 8: the E4 family f9 does not fit
        self.assertEqual(len(got["selected"]), 8)
        self.assertIn("f9", got["not_tested_in_full"])
        t2 = {f: 0.0 for f in n}
        t2.update({"f1": 0.9, "f2": 0.8, "f3": 0.7})                         # targeted top 3 all already chosen
        got = sc.select({"N": n, "F": fi}, {"N": t2, "F": fi}, {"f9": (6, 12), "f4": (1, 12)})
        self.assertEqual(got["selected"][6:], ["f9"])                         # then the >= 50% sign-consistent family
        self.assertEqual(got["reasons"]["f9"], "design log E4 sign-consistent slots 6/12")
        tie = {f: 0.5 for f in n}                                             # head-N ties are broken by head-F importance
        got = sc.select({"N": tie, "F": {f: float(i) for i, f in enumerate(n)}}, {"N": t2, "F": fi}, {})
        self.assertEqual(got["selected"][:2], ["f9", "f8"])

    def test_importance_drops_by_family_with_chunking_and_targeting(self):
        import v2c_screen as sc
        from scipy.stats import spearmanr
        rng = np.random.default_rng(4)
        refs = np.repeat([f"r{i}" for i in range(6)], 20)
        keys = np.array([f"k{i}" for i in range(len(refs))])
        y = rng.normal(size=len(refs))
        x = rng.normal(size=(len(refs), 6)).astype(np.float32)
        x[:, 2] = y + 0.1 * rng.normal(size=len(refs))                       # column 2 carries the signal, inside family A
        targeted = np.arange(len(refs)) % 2 == 0
        store = {}

        def writer(path, ref, score, big):
            store[path] = big

        def predict(bake, path, out):
            return store[path][:, 2].astype(np.float64) * bake                # a "bake" is a scale on column 2

        def srocc(jobs):
            return [float(spearmanr(p, t)[0]) for _, p, t in jobs]

        fams = [("A", [2]), ("B", [4])]
        bakes = {("N", 0): 1.0, ("N", 1): 2.0}
        out = {}
        for chunk in (1, 3, 99):
            with tempfile.TemporaryDirectory() as t:
                out[chunk] = sc.screen_source(x, refs, keys, y, targeted, fams, bakes, 0, Path(t), writer, predict, srocc, draws=3,
                                              chunk_blocks=chunk)
        for chunk in (1, 3):
            self.assertEqual(out[chunk]["drop"], out[99]["drop"])               # the packing never changes a number
        res = sc.summarise({"toy": out[99]}, ["N"])
        self.assertGreater(res["importance"]["N"]["A"], 0.3)
        self.assertLess(abs(res["importance"]["N"]["B"]), 1e-9)               # an uninformative family drops nothing
        self.assertGreater(res["targeted_importance"]["N"]["A"], 0.2)

    def test_screen_specs(self):
        for spec, fam in (("screen_main", "main"), ("screen_aux", "aux")):
            f, v, cols = v2_common.arm_columns(spec)
            self.assertEqual((f, v), (fam, "real"))
            self.assertEqual(cols[:944], list(range(944)))
        _, _, main = v2_common.arm_columns("screen_main")
        self.assertEqual(sorted(set(main)), main)
        self.assertEqual(main[944], 944)
        _, _, aux = v2_common.arm_columns("screen_aux")
        self.assertEqual(aux[944:946], [944, 945])
        self.assertEqual(aux[946:], list(range(1322, 1502)))
        with self.assertRaises(ValueError):
            v2_common.parse_spec("screen_main~p1")


class PinBuilder(unittest.TestCase):
    def test_shortlist_rule(self):
        import v2c_pin
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)

            def rec(arm, head, excess, v1sources, regress=()):
                src = {f"s{i}": {"excess": e, "v1_pass": i < v1sources} for i, e in enumerate(excess)}
                (d / f"{arm}_{head}.json").write_text(json.dumps({"status": "OK", "sources": src, "regressions": list(regress)}))
            rec("c1", "N", [0.02] * 5, 2)             # eligible, mean 0.02
            rec("c1", "F", [0.02] * 5, 2)             # ties c1/N: N first
            rec("c2", "N", [0.05] * 5, 1)             # only 1 V1 source: ineligible
            rec("c3", "N", [0.09] * 5, 3, ["s0"])     # regression: ineligible
            rec("c4", "N", [0.03] * 5, 4)             # eligible, largest
            got, prov = v2c_pin.shortlist(d, ["c1", "c2", "c3", "c4"])
            self.assertEqual([(e["arm"], e["head"]) for e in got], [("c4", "N"), ("c1", "N"), ("c1", "F")])
            self.assertEqual(len(prov), 5)
            many = ["all", "csfw", "c7", "p1", "b1", "b1s", "c8n", "rall"]
            for k, arm in enumerate(many):            # more than 6 eligible: capped at six
                rec(arm, "N", [0.001 * k] * 5, 2)
            self.assertEqual(len(v2c_pin.shortlist(d, [*many, "c1", "c4"])[0]), 6)
            for f in d.glob("*.json"):                 # v2_compare --human-weight 32 writes <arm>_<head>_h32.json
                f.rename(f.with_name(f.stem + "_h32.json"))
            self.assertEqual(v2c_pin.shortlist(d, ["c1", "c2", "c3", "c4"])[0], [])
            got, prov = v2c_pin.shortlist(d, ["c1", "c2", "c3", "c4"], "_h32")
            self.assertEqual([(e["arm"], e["head"]) for e in got], [("c4", "N"), ("c1", "N"), ("c1", "F")])


class Freeze(unittest.TestCase):
    def test_freeze_pins_receipts_and_blocks_rebuilds(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            wide = root / "wide"
            for fam in w.FAMILIES:
                for var in w.VARIANTS:
                    (wide / fam / var).mkdir(parents=True)
                    (wide / fam / var / "receipt.json").write_text(json.dumps({"complete": True, "width": 1825, "feature_set_id": "x"}))
            sets = {n: {"tables": {f: {v: {} for v in w.VARIANTS} for f in w.FAMILIES}} for n in w.CONFIRM_SETS}
            (wide / "confirm").mkdir()
            (wide / "confirm" / "receipt.json").write_text(json.dumps({"width": 1825, "feature_set_id": "x", "sets": sets}))
            (wide / "keep_lists.json").write_text("{}")
            (wide / "verify.json").write_text(json.dumps({"all_ok": False}))
            with self.assertRaises(ValueError):                              # an unclean verify cannot be frozen
                w.freeze(root)
            (wide / "verify.json").write_text(json.dumps({"all_ok": True}))
            with self.assertRaises(ValueError):                              # no frozen.json yet
                v2_common.load_frozen(root)
            w.freeze(root)
            record, digest = v2_common.load_frozen(root)
            self.assertEqual(len(digest), 64)
            with self.assertRaises(ValueError):                              # no rebuild into a frozen root
                w.build_confirm(Path("/nonexistent"), root, ["main"], [], None)
            (wide / "main" / "p1" / "receipt.json").write_text(json.dumps({"complete": True, "width": 1825, "feature_set_id": "y"}))
            with self.assertRaises(ValueError):                              # any receipt change after the freeze is caught
                v2_common.load_frozen(root)

    def test_rev5_freeze_main_real_and_refuse_mixed_confirm_identity(self):
        old_profile = w.PROFILE
        try:
            w.PROFILE = w.rev5_profile("rev5test", "fsid5", "bin5", "build5", 1853)
            with tempfile.TemporaryDirectory() as t:
                root = Path(t)
                wide = root / "wide"
                (wide / "main" / "real").mkdir(parents=True)
                receipt = {"complete": True, "width": 1853, "feature_set_id": "fsid5", "formula_revision": 5}
                rp = wide / "main" / "real" / "receipt.json"
                rp.write_text(json.dumps(receipt))
                (wide / "confirm").mkdir()
                cp = wide / "confirm" / "receipt.json"
                confirm = {"width": 1853, "feature_set_id": "wrong",
                           "sets": {n: {"tables": {"main": {"real": {}}}} for n in w.CONFIRM_SETS}}
                cp.write_text(json.dumps(confirm))
                (wide / "keep_lists.json").write_text("{}")
                (wide / "verify.json").write_text(json.dumps({"all_ok": True}))
                with self.assertRaises(ValueError):
                    w.freeze(root)
                confirm["feature_set_id"] = "fsid5"
                cp.write_text(json.dumps(confirm))
                receipt["formula_revision"] = 4
                rp.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    w.freeze(root)
                receipt["formula_revision"] = 5
                rp.write_text(json.dumps(receipt))
                w.freeze(root)
                frozen, _ = v2_common.load_frozen(root)
                self.assertEqual(set(frozen["wide_receipts"]), {"main/real"})
        finally:
            w.PROFILE = old_profile


class Labels(unittest.TestCase):
    """The label adapter (v2c_labels): by (ref_path, dist_path), never by row_id; strict accounting."""

    def keys(self, n=5):
        return pd.DataFrame({"pair_key": [f"k{i}" for i in range(n)], "ref_path": [f"/r/{i // 2}.png" for i in range(n)],
                             "dist_path": [f"/d/{i}.png" for i in range(n)], "ref_pixels_sha256": [f"r{i // 2}" for i in range(n)],
                             "dist_pixels_sha256": [f"d{i}" for i in range(n)], "n_stimuli": 1, "pixels_identical": False})

    def rows(self, keys, extra=()):
        base = pd.DataFrame({"ref_path": keys.ref_path, "dist_path": keys.dist_path, "label": np.arange(len(keys), dtype=float)})
        base = pd.concat([base, pd.DataFrame(extra, columns=["ref_path", "dist_path", "label"])], ignore_index=True)
        base["file_row"] = np.arange(len(base))
        return base

    def test_maps_by_paths_not_row_order(self):
        import v2c_labels as L
        keys = self.keys()
        rows = self.rows(keys).iloc[::-1].reset_index(drop=True)          # source order is not bank order
        got, acct = L.adapt(rows, keys)
        self.assertEqual(dict(zip(got.pair_key, got.label)), {f"k{i}": float(i) for i in range(5)})
        self.assertEqual(acct["rows_used"], 5)

    def test_unmatched_rows_must_be_identical_stimuli(self):
        import v2c_labels as L
        keys = self.keys()
        with self.assertRaises(ValueError):                               # a stray row that is no identical stimulus
            L.adapt(self.rows(keys, [("/r/9.png", "/d/x.png", 1.0)]), keys)
        keys.loc[1, "pixels_identical"] = True
        keys.loc[1, "n_stimuli"] = 3                                      # 2 more stimuli on the identical key, not matched by path
        got, acct = L.adapt(self.rows(keys, [("/r/0.png", "/d/a.png", 7.0), ("/r/0.png", "/d/b.png", 8.0)]), keys)
        self.assertEqual(sorted(got.pair_key), ["k0", "k2", "k3", "k4"])
        self.assertEqual(acct["rows_on_identical_keys_or_unmatched_identical"], 3)

    def test_collapsed_stimuli_need_the_pixel_table(self):
        import v2c_labels as L
        keys = self.keys()
        keys.loc[2, "n_stimuli"] = 2
        rows = self.rows(keys, [("/r/1.png", "/d/alias.png", 9.0)])
        with self.assertRaises(ValueError):
            L.adapt(rows, keys)
        got, acct = L.adapt(rows, keys, None, {"/r/1.png": "r1", "/d/alias.png": "d2"})
        self.assertEqual(sorted(got.label[got.pair_key == "k2"]), [2.0, 9.0])
        self.assertEqual(acct["matched_by_pixel_hash"], 1)

    def test_select_rule_restricts_keys_too(self):
        import v2c_labels as L
        keys = self.keys(6)                                               # refs /r/0, /r/1, /r/2 (two keys each)
        rows = self.rows(keys)
        rule = {"ref_stem_in": ["0", "2"]}                                # e.g. CID22-B(23): one bank reference is excluded
        got, acct = L.adapt(rows, keys, rule)
        self.assertEqual(sorted(got.pair_key), ["k0", "k1", "k4", "k5"])
        self.assertEqual((acct["unselected_rows"], acct["keys_outside_select"], acct["rows_used"]), (2, 2, 4))
        with self.assertRaises(ValueError):                               # a selected key that lacks its row still refuses
            L.adapt(rows.iloc[1:].reset_index(drop=True), keys, rule)

    def test_select_rule_and_via_pairs(self):
        import v2c_labels as L
        keys = self.keys()
        rows = self.rows(keys, [("/r/other.png", "/d/o.png", 5.0)])
        got, acct = L.adapt(rows, keys, {"ref_path_in": sorted(set(keys.ref_path))})
        self.assertEqual((len(got), acct["unselected_rows"]), (5, 1))
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            pairs = pd.DataFrame({"ref_path": keys.ref_path, "dist_path": keys.dist_path, "row_id": range(5)})
            pairs.to_csv(t / "pairs.tsv", sep="\t", index=False)
            pd.DataFrame({"row_id": range(5), "score": [10, 11, 12, 13, 14]}).to_csv(t / "labels.csv", index=False)
            spec = {"path": str(t / "labels.csv"), "sha256": v2_common.sha(t / "labels.csv"), "format": "csv", "label_col": "score",
                    "via_pairs": {"path": str(t / "pairs.tsv"), "sha256": v2_common.sha(t / "pairs.tsv"), "on": [["row_id", "row_id"]]}}
            got, _ = L.adapt(L.load_label_rows(spec), keys)
            self.assertEqual(got.label.tolist(), [10.0, 11.0, 12.0, 13.0, 14.0])
            spec["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                L.load_label_rows(spec)

    def test_open_sets_reproduce_admitted_labels_exactly(self):
        import v2c_labels as L
        for name in ("aic3", "kadid_select"):                              # OPEN sets only; never a sealed one
            self.assertTrue(L.validate(name)["exact"], name)
        with self.assertRaises(SystemExit):
            L.validate("cid22_b")


class ConfirmRead(unittest.TestCase):
    PRIMARY = ("cid22_b", "aic4", "csiq", "mcljci")
    SETS = ("cid22_b", "aic4", "konjnd_jpeg_select", "konjnd_jpeg_terminal", "csiq", "mcljci")
    FILES = {"cid22_b": ("cid22val_pairs_ab.tsv", "human_score"), "aic4": ("aic4_pairs.tsv", "human_score"),
             "konjnd_jpeg_select": ("konjnd_jpeg_val_pairs.tsv", "human_score"), "konjnd_jpeg_terminal": ("konjnd_jpeg_val_pairs.tsv", "human_score"),
             "csiq": ("csiq_pairs.tsv", "human_score"), "mcljci": ("mcljci_labels.csv", "jnd_dist")}
    DISTORTION = ("aic4", "konjnd_jpeg_select", "konjnd_jpeg_terminal", "mcljci")  # label-file orientation, fixed independently
    REFS, PER_REF = 30, 8
    REFS_OF: dict = {}
    B = 200

    def ref_name(self, name: str, j: int) -> str:
        return f"ref{j}"

    @classmethod
    def setUpClass(cls):
        import v2_compare
        import v2_confirm_read as cr
        cls.cmp, cls.cr = v2_compare, cr
        cls.saved = (v2_compare.BOOT_B, v2_compare.SEED_DRAWS, dict(v2_compare._ref_draws), dict(v2_compare._rendered))
        v2_compare.BOOT_B = cls.B
        v2_compare.SEED_DRAWS = np.random.default_rng(v2_common.BOOT_SEED + 1).integers(0, 10, size=(cls.B, 10))
        v2_compare._ref_draws.clear()
        v2_compare._rendered.clear()

    @classmethod
    def tearDownClass(cls):
        cmp = cls.cmp
        cmp.BOOT_B, cmp.SEED_DRAWS = cls.saved[0], cls.saved[1]
        cmp._ref_draws.clear(), cmp._ref_draws.update(cls.saved[2])
        cmp._rendered.clear(), cmp._rendered.update(cls.saved[3])
        cmp.REF_SEEDS.clear()

    def build_tree(self, root: Path, planted: dict, weight=None, heads=("N",), shortlist=None):
        """Synthetic canon root + bank + labels + pin. planted: arm -> noise sd, or {set: sd, None: default sd}."""
        cr = self.cr
        rng = np.random.default_rng(11)
        width = 948
        for rel in ("wide/confirm", "bank", "labels"):
            (root / rel).mkdir(parents=True, exist_ok=True)
        receipt = {"schema": "rev4-featpot-v2c-confirm-v1", "width": width, "feature_set_id": "x", "sets": {}}
        truth, labels = {}, {}
        for name in self.SETS:
            n = self.REFS_OF.get(name, self.REFS) * self.PER_REF
            keys = pd.DataFrame({"pair_key": [f"{name}-{i}" for i in range(n)], "row_id": np.arange(n),
                                 "ref_group": [self.ref_name(name, i // self.PER_REF) for i in range(n)],
                                 "ref_path": [f"/{name}/{self.ref_name(name, i // self.PER_REF)}.png" for i in range(n)],
                                 "dist_path": [f"/{name}/d{i}.png" for i in range(n)],
                                 "ref_pixels_sha256": ["r"] * n, "dist_pixels_sha256": [f"p{i}" for i in range(n)],
                                 "n_stimuli": np.ones(n, dtype=np.int32), "pixels_identical": [i == 3 for i in range(n)]})
            (root / "bank" / name).mkdir()
            pq.write_table(pa.Table.from_pandas(keys, preserve_index=False), root / "bank" / name / "keys.parquet")
            nonid = keys.loc[~keys.pixels_identical].reset_index(drop=True)
            ck = pd.DataFrame({"pair_key": nonid.pair_key, "row_id": nonid.row_id, "ref_basename": nonid.ref_group, "member_set": name})
            tables = {}
            for variant in ("real", "p1", "p2", "p3"):
                d = root / "wide" / "confirm" / "main" / ("" if variant == "real" else variant)
                d.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(ck, preserve_index=False), d / f"{name}.keys.parquet")
                added = rng.normal(size=(len(ck), width - 944))
                tab = pd.DataFrame({"ref_basename": ck.ref_basename, "human_score": 0.0,
                                    **{f"f{i}": added[:, i - 944] if i >= 944 else 0.0 for i in range(width)}})
                if name == "konjnd_jpeg_select" and variant != "real":
                    tab = tbl_real                                     # singleton references: the permutation is the identity
                pq.write_table(pa.Table.from_pandas(tab, preserve_index=False), d / f"{name}.parquet")
                if variant == "real":
                    tbl_real = tab
                tables[variant] = {"rel": str((d / f"{name}.parquet").relative_to(root)), "sha256": v2_common.sha(d / f"{name}.parquet"),
                                   "manifest_sha256": "m", "keys_sha256": v2_common.sha(d / f"{name}.keys.parquet")}
            receipt["sets"][name] = {"rows": len(nonid), "tables": {"main": tables}}
            q = rng.normal(size=len(nonid))
            truth[name] = q
            sign = -1.0 if name in self.DISTORTION else 1.0
            fname, col = self.FILES[name]
            (root / "labels" / name).mkdir()
            lab = pd.DataFrame({"ref_path": keys.ref_path, "dist_path": keys.dist_path, col: 0.0})
            lab.loc[~keys.pixels_identical.to_numpy(), col] = sign * (2.0 * q + 3.0)   # the identical key keeps a dummy label row
            lp = root / "labels" / name / fname
            lab.to_csv(lp, sep="," if fname.endswith(".csv") else "\t", index=False)
            labels[name] = {"path": str(lp), "sha256": v2_common.sha(lp), "format": "csv" if fname.endswith(".csv") else "tsv",
                            "ref_col": "ref_path", "dist_col": "dist_path", "label_col": col, "select": None}
        (root / "wide" / "confirm" / "receipt.json").write_text(json.dumps(receipt))
        (root / "wide" / "keep_lists.json").write_text("{}")
        wide = {}
        for v in ("real", "p1", "p2", "p3"):
            (root / "wide" / "main" / v).mkdir(parents=True, exist_ok=True)
            (root / "wide" / "main" / v / "receipt.json").write_text(json.dumps({"v": v}))
            wide[f"main/{v}"] = v2_common.sha(root / "wide" / "main" / v / "receipt.json")
        frozen = {"schema": v2_common.FROZEN_SCHEMA, "wide_receipts": wide, "extra_arms_sha256": None,
                  "confirm_receipt_sha256": v2_common.sha(root / "wide" / "confirm" / "receipt.json"),
                  "keep_lists_sha256": v2_common.sha(root / "wide" / "keep_lists.json")}
        (root / "wide" / "frozen.json").write_text(json.dumps(frozen))
        frozen_sha = v2_common.sha(root / "wide" / "frozen.json")
        panel = os.environ["ZEN_PANEL_BIN"]
        binaries = {"zensim_mlp_train": "t", "bake_dial_refit": "b", "panel": v2_common.sha(Path(panel))}
        prog, data = "a" * 64, "b" * 64
        w = weight
        specs = {"r0": 1.0}
        for a, sd in planted.items():
            specs[a] = sd
            specs.update({f"{a}~p{k}": 1.0 for k in (1, 2, 3)})
        for spec, sd in specs.items():
            variant = f"p{spec.partition('~p')[2]}" if "~p" in spec else "real"
            full = self.cr.full_spec(spec, w)
            for head in heads:
                for seed in range(10):
                    preds = {}
                    for name in self.SETS:
                        s = sd.get(name, sd.get(None, 1.0)) if isinstance(sd, dict) else sd
                        preds[name] = {"rows": len(truth[name]), "pred": (truth[name] + rng.normal(0, s, len(truth[name]))).tolist(),
                                       "table_sha256": receipt["sets"][name]["tables"]["main"][variant]["sha256"],
                                       "keys_sha256": receipt["sets"][name]["tables"]["main"][variant]["keys_sha256"]}
                    d = cr.cell_path(root, full, head, seed)
                    d.mkdir(parents=True)
                    (d / "result.json").write_text(json.dumps({
                        "schema": "rev4-featpot-v2c-confirm-cell-v1", "spec": full, "head": head, "seed_index": seed,
                        "family": "main", "variant": variant, "eval_variant": variant, "wide_receipt_sha256": wide[f"main/{variant}"],
                        "confirm_receipt_sha256": frozen["confirm_receipt_sha256"], "keep_lists_sha256": frozen["keep_lists_sha256"],
                        "frozen_sha256": frozen_sha, "binaries": binaries, "predictions": preds,
                        "human_nominal_weight": 0.5 if w is None else w}))
                    (d / "fleet_receipt.json").write_text(json.dumps({"program_sha": prog, "data_sha": data}))
        prov = root / "prov.json"
        prov.write_text("{}")
        pin = {"schema": cr.PIN_SCHEMA, "reference": "r0", "candidates": sorted(planted), "heads": ["N", "F"],
               "shortlist": shortlist or [{"arm": a, "head": h} for a in sorted(planted) for h in heads], "human_weight": weight,
               "frozen_sha256": frozen_sha, "wide_receipts": wide, "confirm_receipt_sha256": frozen["confirm_receipt_sha256"],
               "keep_lists_sha256": frozen["keep_lists_sha256"], "binaries": binaries, "program_sha": prog, "data_sha": data,
               "code": {k: v2_common.sha(p) for k, p in cr.CODE_FILES.items()},
               "shortlist_provenance": {"calibration": {"path": str(prov), "sha256": v2_common.sha(prov)}, "compare": {}},
               "labels": labels}
        pin_path = root / "pin.json"
        pin_path.write_text(json.dumps(pin))
        return pin_path

    def run_read(self, root: Path, planted: dict, **kw):
        cr = self.cr
        pin_path = self.build_tree(root, planted, **kw)
        out, ledger = root / "read.json", root / "ledger.md"
        with mock.patch.object(v2_common, "V2", root), mock.patch.object(cr, "V2", root):
            cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"), "--out", str(out),
                     "--ledger", str(ledger), *[x for a in planted for x in ("--arm", a)]])
        return json.loads(out.read_text()), ledger

    def test_planted_effect_found_null_not_orientation_and_r22_konjnd(self):
        with tempfile.TemporaryDirectory() as t:
            res, ledger = self.run_read(Path(t), {"planted": 0.55, "nullarm": 1.0})
            by = {e["arm"]: e for e in res["primary"]}
            self.assertTrue(by["planted"]["confirmed"], by["planted"])
            self.assertEqual(by["planted"]["verdict"], "confirmed; V3 dial gates next")
            self.assertFalse(by["nullarm"]["confirmed"])
            self.assertEqual(sorted(by["planted"]["per_set_excess"]), sorted(self.PRIMARY))   # KonJND is not in the mean
            kon = by["planted"]["konjnd_select"]
            self.assertGreater(kon["delta"], 0.03)                                          # KonJND reports delta vs R0
            self.assertFalse(kon["regression"])
            self.assertGreater(kon["r0_mean"], 0.5)                                         # distortion labels negated: signed SROCC > 0
            sec = res["secondary_v1_v2"]["planted_N"]
            self.assertTrue(sec["V1"] and sec["V2"])
            self.assertNotIn("nullarm_F", res["secondary_v1_v2"])                          # shortlisted entries only
            self.assertEqual(res["provenance"]["perm_identity_share"]["konjnd_jpeg_select"], [1.0, 1.0, 1.0])
            self.assertLess(max(res["provenance"]["perm_identity_share"]["cid22_b"]), 0.1)
            self.assertIn("pending, update after read", ledger.read_text())
            self.assertIn("R2.2", ledger.read_text())
            self.assertEqual(res["provenance"]["per_set"]["csiq"]["accounting"]["rows_used"], self.REFS * self.PER_REF - 1)

    def test_konjnd_regression_vetoes_an_otherwise_confirmed_entry(self):
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), {"vetoed": {None: 0.55, "konjnd_jpeg_select": 3.0}})
            e = res["primary"][0]
            self.assertTrue(e["holm_significant"])
            self.assertTrue(e["konjnd_select"]["regression"])
            self.assertFalse(e["confirmed"])
            self.assertIn("regresses", e["verdict"])

    def test_primary_holm_keeps_planted_rejects_nulls(self):
        arms = {"planted": 0.6, "nullA": 1.0, "nullB": 1.0, "nullC": 1.0}
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), arms)
        by = {e["arm"]: e for e in res["primary"]}
        self.assertTrue(by["planted"]["confirmed"], by["planted"])
        self.assertLess(by["planted"]["p_one_sided"], 0.05 / 4)
        for n in ("nullA", "nullB", "nullC"):
            self.assertFalse(by[n]["confirmed"], by[n])

    def test_weight_suffixed_cells_and_free_head_labels(self):
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), {"planted": 0.55}, weight=2.0, heads=("N", "F"))
            by = {e["head"]: e for e in res["primary"]}
            self.assertEqual(by["N"]["verdict"], "confirmed; V3 dial gates next")
            self.assertEqual(by["F"]["verdict"], "confirmed under the free head (the N entry is also confirmed)")
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), {"planted": 0.55}, heads=("N", "F"), shortlist=[{"arm": "planted", "head": "F"}])
            self.assertEqual(res["primary"][0]["verdict"], "confirmed under the free head only: helps a free head")

    def test_independent_bootstrap_streams_per_set(self):
        with tempfile.TemporaryDirectory() as t:
            self.run_read(Path(t), {"planted": 0.55})
        d = self.cmp._ref_draws
        self.assertFalse(np.array_equal(d["cid22_b"][0], d["aic4"][0]))           # same sizes, different streams
        self.assertEqual(self.cmp.REF_SEEDS["aic4"], [v2_common.BOOT_SEED, 1])

    def test_holm_step_down(self):
        cr = self.cr
        self.assertEqual(cr.holm([0.001, 0.03, 0.04, 0.5, 0.6, 0.7]), [True, False, False, False, False, False])
        self.assertEqual(cr.holm([0.03, 0.5, 0.6, 0.7, 0.8, 0.9]), [False] * 6)     # lone nominally significant entry is rejected
        self.assertEqual(cr.holm([0.008, 0.012, 0.9]), [True, True, False])
        self.assertEqual(cr.holm([0.02, 0.5]), [True, False])
        self.assertEqual(cr.holm([0.03, 0.5]), [False, False])

    def test_weak_lucky_entry_fails_holm_end_to_end(self):
        cr = self.cr
        # p just under 0.05 for an entry on a list of six: nominally significant, not Holm-significant
        entries = [{"arm": f"a{i}", "head": "N", "status": "OK", "p_one_sided": p, "regressions": [], "konjnd_select": {"regression": False}}
                   for i, p in enumerate([0.03, 0.4, 0.5, 0.6, 0.7, 0.9])]
        cr.verdicts(entries)
        self.assertFalse(any(e["confirmed"] for e in entries))
        self.assertEqual(entries[0]["verdict"], "not confirmed")

    def exposed(self, ledger: Path) -> bool:
        return ledger.exists()

    def test_refusals_leave_no_ledger_and_open_no_label(self):
        cr = self.cr
        def attempt(tamper):
            with tempfile.TemporaryDirectory() as t:
                root = Path(t)
                pin_path = self.build_tree(root, {"planted": 0.55})
                tamper(root, pin_path)
                opened = []
                real_load = cr.v2c_labels.load_label_rows
                with mock.patch.object(v2_common, "V2", root), mock.patch.object(cr, "V2", root), \
                        mock.patch.object(cr.v2c_labels, "load_label_rows", lambda *a, **k: (opened.append(1), real_load(*a, **k))[1]):
                    with self.assertRaises(cr.Refusal):
                        cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"),
                                 "--out", str(root / "o.json"), "--ledger", str(root / "l.md"), "--arm", "planted"])
                self.assertFalse((root / "l.md").exists())
                self.assertEqual(opened, [])

        def cell(mutator, spec="r0"):
            def f(root, pin_path):
                path = self.cr.cell_path(root, spec, "N", 3) / "result.json"
                rec = json.loads(path.read_text())
                mutator(rec)
                path.write_text(json.dumps(rec))
            return f

        attempt(cell(lambda r: r["binaries"].update(panel="other")))
        attempt(cell(lambda r: r.update(variant="real", eval_variant="real"), "planted~p1"))   # control scored on the real tables
        attempt(cell(lambda r: r["predictions"]["csiq"].update(pred=r["predictions"]["csiq"]["pred"] + [0.0])))   # longer vector
        attempt(cell(lambda r: r["predictions"]["aic4"].update(table_sha256="0" * 64)))
        attempt(cell(lambda r: r["predictions"]["aic4"].update(keys_sha256="0" * 64)))
        attempt(cell(lambda r: r.update(frozen_sha256="0" * 64)))
        attempt(lambda root, pin: __import__("shutil").rmtree(self.cr.cell_path(root, "planted", "N", 4)))     # missing cell
        attempt(lambda root, pin: (self.cr.cell_path(root, "r0", "N", 0) / "fleet_receipt.json").unlink())    # fleet receipt mandatory
        attempt(lambda root, pin: (root / "labels" / "csiq" / "csiq_pairs.tsv").write_text("tampered"))        # label sha256

        def pin_edit(mutator):
            def f(root, pin_path):
                pin = json.loads(pin_path.read_text())
                mutator(pin)
                pin_path.write_text(json.dumps(pin))
            return f
        attempt(pin_edit(lambda p: p.pop("program_sha")))
        attempt(pin_edit(lambda p: p["code"].update({"rev4_featpot/v2_compare.py": "0" * 64})))
        attempt(pin_edit(lambda p: p["labels"]["aic4"].update(label_col="other")))             # orientation key unlisted
        attempt(pin_edit(lambda p: p["labels"]["csiq"].update(path="/x/_sealed/csiq_pairs.tsv")))

    def test_cell_names_must_match_the_weight(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pin_path = self.build_tree(root, {"planted": 0.55}, weight=2.0)
            pin = json.loads(pin_path.read_text())
            pin["human_weight"] = None                                                     # cells are @h2: a bare read finds none
            pin_path.write_text(json.dumps(pin))
            with mock.patch.object(v2_common, "V2", root), mock.patch.object(self.cr, "V2", root):
                with self.assertRaises(self.cr.Refusal):
                    self.cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"),
                                  "--out", str(root / "o.json"), "--ledger", str(root / "l.md"), "--arm", "planted"])

    def test_panel_must_be_pinned(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pin_path = self.build_tree(root, {"planted": 0.55})
            with mock.patch.object(v2_common, "V2", root), mock.patch.object(self.cr, "V2", root), \
                    mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ZEN_PANEL_BIN")
                try:
                    with self.assertRaises(self.cr.Refusal):
                        self.cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"),
                                      "--out", str(root / "o.json"), "--ledger", str(root / "l.md"), "--arm", "planted"])
                finally:
                    os.environ["ZEN_PANEL_BIN"] = "/var/tmp/fitv2/bin-v2/panel"


class SetCompareRead(ConfirmRead):
    """Amendment R7: the set-compare read on synthetic labels (planted better set, equal set, worse set, null, refusals)."""
    REFS_OF = {"mcljci": 50}   # R7a needs MCL-JCI's 50 numbered sources

    def ref_name(self, name: str, j: int) -> str:
        return f"imagejnd_src{j + 1:02d}" if name == "mcljci" else f"ref{j}"

    def sc_pin(self, root: Path, pin_path: Path, entries: dict, superiority, noninferiority) -> Path:
        base = json.loads(pin_path.read_text())
        reg = root / "R7.md"
        reg.write_text("registration")
        r7a = root / "R7a.md"
        r7a.write_text("amendment r7a")
        pin = {k: base[k] for k in ("frozen_sha256", "wide_receipts", "confirm_receipt_sha256", "keep_lists_sha256", "binaries",
                                    "program_sha", "data_sha", "code", "labels")}
        pin.update({"schema": self.cr.SC_PIN_SCHEMA, "entries": entries, "head": "N", "superiority": superiority,
                    "noninferiority": noninferiority, "ensemble_pairs": [{"arm": "B", "reference": "A"}],
                    "registration": {"path": str(reg), "sha256": v2_common.sha(reg)},
                    "amendment_r7a": {"path": str(r7a), "sha256": v2_common.sha(r7a)}})
        out = root / "scpin.json"
        out.write_text(json.dumps(pin))
        return out

    def run_sc(self, root: Path, planted: dict, entries: dict, superiority, noninferiority, tamper=None):
        cr = self.cr
        pin_path = self.sc_pin(root, self.build_tree(root, planted), entries, superiority, noninferiority)
        if tamper:
            tamper(root, pin_path)
        out, ledger = root / "sc.json", root / "ledger.md"
        with mock.patch.object(v2_common, "V2", root), mock.patch.object(cr, "V2", root):
            cr.main(["--confirmatory-read", "--set-compare", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"),
                     "--out", str(out), "--ledger", str(ledger)])
        return json.loads(out.read_text()), ledger

    ENTRIES = {"A": "setA", "B": "setB", "C": "setC", "D": "r0"}
    SUP = [{"arm": "A", "reference": "C"}, {"arm": "A", "reference": "D"}]
    NI = [{"arm": "B", "reference": "A"}]

    def test_planted_superiority_and_noninferiority(self):
        with tempfile.TemporaryDirectory() as t:
            # B planted slightly better than A: 240 synthetic pairs per set cannot resolve a ±0.005 margin for EQUAL sets
            # (bootstrap 5th percentile ≈ −0.012 there); this exercises the acceptance branch with margin.
            res, ledger = self.run_sc(Path(t), {"setA": 0.55, "setB": 0.40, "setC": 1.0}, self.ENTRIES, self.SUP, self.NI)
            sup = {(e["arm"], e["reference"]): e for e in res["superiority"]}
            self.assertTrue(sup[("A", "C")]["confirmed"], sup[("A", "C")])
            self.assertTrue(sup[("A", "D")]["confirmed"])
            self.assertGreater(sup[("A", "C")]["mean_delta"], 0.02)
            self.assertEqual(sorted(sup[("A", "C")]["per_set"]),
                             sorted((*self.PRIMARY, "konjnd_jpeg_select", "konjnd_jpeg_terminal", "mcljci_k40")))
            # R7a: the clean primary swaps MCL-JCI for its 40 non-KonFiG sources; both primaries must agree for a verdict.
            self.assertEqual(sup[("A", "C")]["clean"]["sets"], ["cid22_b", "aic4", "csiq", "mcljci_k40"])
            self.assertTrue(sup[("A", "C")]["confirmed_registered"] and sup[("A", "C")]["confirmed_clean"])
            self.assertEqual(sup[("A", "C")]["verdict"], "confirmed")
            self.assertTrue(ni["as_good_registered"] if (ni := res["noninferiority"][0]) else False)
            self.assertEqual(res["provenance"]["per_set"]["mcljci_k40"]["references"], 40)
            self.assertEqual(res["provenance"]["per_set"]["mcljci"]["references"], 50)
            ni = res["noninferiority"][0]
            self.assertTrue(ni["as_good"], ni)
            self.assertGreater(res["entry_mean_signed_srocc"]["A"]["aic4"], 0.5)    # distortion labels negated: signed SROCC > 0
            self.assertEqual(len(res["ensemble_delta_groups_0_4_and_5_9"]["B-A"]["csiq"]), 2)
            self.assertIn("A-C", res["secondary_pairs"])
            self.assertIn("pending, update after read", ledger.read_text())
            self.assertIn("R7", ledger.read_text())

    def test_worse_set_not_noninferior_and_null_not_confirmed(self):
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_sc(Path(t), {"setA": 1.0, "setB": 1.8, "setC": 1.0}, self.ENTRIES, self.SUP, self.NI)
            sup = {(e["arm"], e["reference"]): e for e in res["superiority"]}
            self.assertFalse(sup[("A", "C")]["confirmed"])
            self.assertFalse(sup[("A", "D")]["confirmed"])
            self.assertFalse(res["noninferiority"][0]["as_good"])

    def test_set_compare_refusals_leave_no_ledger_and_open_no_label(self):
        cr = self.cr
        def attempt(tamper):
            with tempfile.TemporaryDirectory() as t:
                root = Path(t)
                opened = []
                real_load = cr.v2c_labels.load_label_rows
                with mock.patch.object(cr.v2c_labels, "load_label_rows", lambda *a, **k: (opened.append(1), real_load(*a, **k))[1]):
                    with self.assertRaises(cr.Refusal):
                        self.run_sc(root, {"setA": 0.55, "setB": 0.55, "setC": 1.0}, self.ENTRIES, self.SUP, self.NI, tamper=tamper)
                self.assertFalse((root / "ledger.md").exists())
                self.assertEqual(opened, [])
        def pin_edit(mutator):
            def f(root, pin_path):
                pin = json.loads(pin_path.read_text())
                mutator(pin)
                pin_path.write_text(json.dumps(pin))
            return f
        attempt(lambda root, pin: (root / "R7.md").write_text("edited after the pin"))          # registration changed
        attempt(lambda root, pin: (root / "R7a.md").write_text("edited after the pin"))         # R7a amendment changed
        attempt(pin_edit(lambda p: p["superiority"].append({"arm": "A", "reference": "Z"})))    # unknown entry
        attempt(pin_edit(lambda p: p["entries"].update(B="setA")))                                # duplicate spec
        attempt(lambda root, pin: __import__("shutil").rmtree(self.cr.cell_path(root, "setB", "N", 2)))   # missing cell


class R7aGuard(unittest.TestCase):
    """R7a: the clean-primary derivation and the agreement rule, without labels."""

    def test_mcljci_clean_and_guarded(self):
        import v2_confirm_read as cr
        lab = pd.DataFrame({"pred_row": np.arange(100), "ref_basename": [f"imagejnd_src{i // 2 + 1:02d}" for i in range(100)]})
        out = cr.mcljci_clean(lab)
        self.assertEqual(out.ref_basename.nunique(), 40)
        self.assertFalse(out.ref_basename.isin(["imagejnd_src01", "imagejnd_src45", "imagejnd_src50"]).any())
        self.assertEqual(len(out), 80)
        with self.assertRaises(cr.Refusal):
            cr.mcljci_clean(lab.iloc[:98])                 # 49 sources
        with self.assertRaises(cr.Refusal):
            cr.mcljci_clean(lab.assign(ref_basename="ref0"))
        self.assertEqual(cr.guarded(True, True, "confirmed", "not confirmed"), "confirmed")
        self.assertEqual(cr.guarded(False, False, "confirmed", "not confirmed"), "not confirmed")
        self.assertEqual(cr.guarded(True, False, "confirmed", "no"), "contamination-sensitive (registered primary only: confirmed)")
        self.assertEqual(cr.guarded(False, True, "as good", "no"), "contamination-sensitive (clean primary only: as good)")


class TeacherSubsets(unittest.TestCase):
    """Design log E13: `ts<name>` recipe tokens and the curated SafeSyn fit leg."""

    def test_token_parses_and_refuses(self):
        self.assertEqual(v2_common.recipe_of("set:v2+basic@h32:H128:tsmono5"), {"hidden": 128, "teacher_subset": "mono5"})
        self.assertEqual(v2_common.recipe_of("set:v2+basic@h32:H128"), {"hidden": 128})
        for bad in ("set:v2+basic@h32:H128:tsbogus", "set:v2+basic@h32:H128:tsneg:tsq20"):
            with self.assertRaises(ValueError):
                v2_common.recipe_of(bad)

    def test_rules(self):
        import v2_teacher as t
        ref = np.array(["a"] * 4 + ["b"] * 4)
        codec = np.array(["zenwebp-default-m4"] * 4 + ["zenjxl-e7"] * 4)
        q = np.array([5, 50, 80, 100] * 2)
        y = np.array([-700.0, 40.0, 30.0, 90.0, -5.0, 50.0, 70.0, 95.0])  # series a falls by 10 from q50 to q80
        keep, new = t.curate("win", ref, y, codec, q, -60.0)
        self.assertTrue(keep.all())
        self.assertEqual(new[0], -60.0)
        self.assertEqual(int((new != y).sum()), 1)
        self.assertEqual(t.curate("floor0", ref, y, codec, q, 0)[1].min(), 0.0)
        self.assertEqual(t.curate("neg", ref, y, codec, q, 0)[0].tolist(), [False, True, True, True, False, True, True, True])
        self.assertEqual(t.curate("q20", ref, y, codec, q, 0)[0].tolist(), [False, True, True, True] * 2)
        self.assertEqual(t.curate("mono5", ref, y, codec, q, 0)[0].tolist(), [False] * 4 + [True] * 4)
        self.assertEqual(t.curate("xwebp", ref, y, codec, q, 0)[0].tolist(), [False] * 4 + [True] * 4)
        with self.assertRaises(ValueError):
            t.curate("none", ref, y, codec, q, 0)

    def test_curated_table_and_strata_identity(self):
        import v2_teacher as t
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            n = 40000  # spans several 16384-row batches
            rng = np.random.default_rng(1)
            src = d / "fit.parquet"
            pq.write_table(pa.table({"ref_basename": [f"r{i // 7}" for i in range(n)],
                                     "human_score": rng.normal(50, 30, n), "f0": rng.normal(size=n).astype(np.float32)}), src)
            keep = rng.random(n) < 0.7
            new = pq.read_table(src)["human_score"].to_numpy().copy()
            new[::3] = 1.5
            rec = t.write_curated(src, d / "out.parquet", keep, new)
            original = pq.read_table(src)["human_score"].to_numpy()
            self.assertEqual(rec, {"rows_in": n, "rows_kept": int(keep.sum()),
                                   "targets_changed": int(np.count_nonzero(original[keep] != new[keep])),
                                   "row_selection_sha256": hashlib.sha256(
                                       np.flatnonzero(keep).astype("<u8").tobytes()).hexdigest()})
            got = pq.read_table(d / "out.parquet").to_pandas()
            want = pq.read_table(src).to_pandas().assign(human_score=new)[keep].reset_index(drop=True)
            pd.testing.assert_frame_equal(got, want)
            strata = d / "strata.npz"
            np.savez_compressed(strata, schema=np.array(t.STRATA_SCHEMA), keys_sha256=np.array("abc"),
                                codec_names=np.array(["zenjxl-e7"]), codec_idx=np.zeros(3, np.uint8),
                                quality=np.array([5, 50, 90], np.uint8))
            with mock.patch.object(t, "strata_path", lambda: strata):
                codec, quality, _ = t.load_strata("abc")
                self.assertEqual(codec.tolist(), ["zenjxl-e7"] * 3)
                self.assertEqual(quality.tolist(), [5, 50, 90])
                with self.assertRaises(ValueError):
                    t.load_strata("other-rows")


class KadisOrdinal(unittest.TestCase):
    """Design log E14: `ko<w>` tokens, ladder labels, and the NaN-column refusal."""

    def test_token(self):
        self.assertEqual(v2_common.recipe_of("set:v2+basic@h32:H128:ko4"), {"hidden": 128, "kadis_ordinal": 4.0})
        self.assertEqual(v2_common.recipe_of("set:v2+basic@h32:H128:tsfloor0:ko16"),
                         {"hidden": 128, "teacher_subset": "floor0", "kadis_ordinal": 16.0})
        for bad in ("set:v2+basic@h32:H128:ko0", "set:v2+basic@h32:H128:ko100", "set:v2+basic@h32:H128:ko1:ko4"):
            with self.assertRaises(ValueError):
                v2_common.recipe_of(bad)

    def test_ladders_split_signed_types_by_direction(self):
        import e14_kadis_ordinal as e
        sel = pd.DataFrame({"source_filename": ["a.png"] * 5 + ["b.png"] * 5, "dist_type": [25] * 5 + [23] * 5,
                            "dist_param": [0.3, 0.15, 0.0, -0.4, -0.6, 2.0, 4.0, 6.0, 8.0, 10.0]})
        key, target = e.ladders(sel)
        self.assertEqual(key.tolist(), ["kadis:a.png|t25|+"] * 2 + ["kadis:a.png|t25|0"] + ["kadis:a.png|t25|-"] * 2
                         + ["kadis:b.png|t23|+"] * 5)
        self.assertEqual(target.tolist(), [-0.3, -0.15, -0.0, -0.4, -0.6, -2.0, -4.0, -6.0, -8.0, -10.0])

    def test_ordinal_table_pin(self):
        import v2_teacher as t
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "kadis_ordinal.parquet"
            bad.write_bytes(b"not the table")
            with mock.patch.object(t, "REPO", Path(d).parent), mock.patch.object(t, "ORDINAL_NAME", str(bad)):
                with self.assertRaises(ValueError):
                    t.ordinal_leg()


class CoverageLeg(unittest.TestCase):
    """Design log E15: `cv<w>:cf<mask>` tokens, the family filter, and our TID-style generators."""

    def test_tokens(self):
        self.assertEqual(v2_common.recipe_of("set:v2+basic@h32:H128:cv4:cfff"),
                         {"hidden": 128, "coverage_weight": 4.0, "coverage_mask": 255})
        for bad in ("set:v2+basic@h32:H128:cv4", "set:v2+basic@h32:H128:cfff", "set:v2+basic@h32:H128:cv4:cf0",
                    "set:v2+basic@h32:H128:cv4:cf100"):
            with self.assertRaises(ValueError):
                v2_common.recipe_of(bad)

    def test_family_filter(self):
        import v2_teacher as t
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fams = ["blur", "new", "light", "new", "blur"]
            pool, keys = d / "coverage_pool.parquet", d / "coverage_pool.keys.parquet"
            pq.write_table(pa.table({"ref_basename": [f"l{i // 2}" for i in range(5)], "human_score": [-1.0, -2.0, -3.0, -4.0, -5.0],
                                     "f0": np.arange(5, dtype=np.float32)}), pool)
            pq.write_table(pa.table({"family": fams}), keys)
            sha = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()
            with mock.patch.object(t, "_packed_or_root", lambda name: pool if name == t.POOL_NAME else keys), \
                    mock.patch.object(t, "POOL_SHA", sha(pool)), mock.patch.object(t, "POOL_KEYS_SHA", sha(keys)):
                mask = 1 << list(v2_common.COVERAGE_FAMILIES).index("new")
                path, rec = t.coverage_leg(mask, d)
                self.assertEqual(pq.read_table(path)["f0"].to_pylist(), [1.0, 3.0])
                self.assertEqual(rec["families"], ["new"])
            with mock.patch.object(t, "_packed_or_root", lambda name: pool if name == t.POOL_NAME else keys):
                with self.assertRaises(ValueError):
                    t.coverage_leg(1, d)
            # Rev5: the pool's manifest declares revision 5, which selects the Rev5 pin (and refuses the Rev4 one).
            Path(f"{pool}.manifest.json").write_text(json.dumps({"formula_revision": 5}))
            with mock.patch.object(t, "_packed_or_root", lambda name: pool if name == t.POOL_NAME else keys), \
                    mock.patch.object(t, "POOL_SHA", sha(pool)), mock.patch.object(t, "POOL_KEYS_SHA", sha(keys)):
                with self.assertRaises(ValueError):
                    t.coverage_leg(mask, d)
                with mock.patch.object(t, "POOL_SHA_REV5", sha(pool)):
                    path, rec = t.coverage_leg(mask, d)
                    self.assertEqual(rec["pool_formula_revision"], 5)

    def test_generators_are_nested_and_monotone(self):
        import e15_coverage as e
        flat = np.full((96, 128, 3), 128, np.uint8)  # every block value differs from 128 (|delta| >= 32), so coverage is visible
        changed = [e.local_block_wise(flat, lv, 77) != flat for lv in range(1, 6)]
        for lo, hi in zip(changed, changed[1:]):
            self.assertTrue((lo <= hi).all())  # every level keeps the lower level's blocks
        self.assertGreater(changed[4].sum(), changed[0].sum())
        img = np.random.default_rng(3).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(e.local_block_wise(img, 3, 77), e.local_block_wise(img, 3, 77)))
        cab = e.chromatic_aberration(img, 2)
        self.assertTrue(np.array_equal(cab[:, 2:, 0], img[:, :-2, 0]) and np.array_equal(cab[:, :-2, 2], img[:, 2:, 2]))
        self.assertTrue(np.array_equal(cab[:, :, 1], img[:, :, 1]))


if __name__ == "__main__":
    unittest.main()
