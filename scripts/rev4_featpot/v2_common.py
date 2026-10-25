"""Instrument v2 shared definitions.

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md (with its revision R1). POTENTIAL —
ceiling, not a model score. Every value here is fixed by that amendment; change the amendment first.

Wide-table layout (revision R1.1): two table families, both WIDTH = 1825 columns, so a kept column's
first-layer initial weights are identical in every arm of either family (--keep-features keeps full width).
  main: f0..f943 bank Rev3 944 surface (39 structural zeros filled with 0, as admit_bank does);
        f944..f1824 the Rev4 research vector at its canonical IDs.
  aux:  f0..f943 bank; f944 gmsd, f945 gmsm (the reviewed peer pair, packed at registered IDs exactly as the
        registered P2 arm did); f946 oracle_lo, f947 oracle_hi; f1322..f1501 gmsbank (arm p3); all other
        columns 0. Every column a bake reads is a registered feature ID, which the product predictor requires.
Variants: "real" and "p1".."p3" (the family's added columns permuted jointly within reference over pair keys).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/var/tmp/rev4-featpot")


def _root_override(argv: list[str]) -> str | None:
    """`--root DIR` / `--root=DIR` anywhere in argv, else $REV4_V2_ROOT, else None (CANONTAB: the Rev3 v2 instrument and
    the v2-canon instrument share these scripts and coexist under different roots; argv carries the root, so the
    declared cells' argv hashes, hence their job ids, differ)."""
    for i, arg in enumerate(argv):
        if arg == "--root" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--root="):
            return arg.split("=", 1)[1]
    return os.environ.get("REV4_V2_ROOT")


V2 = Path(_root_override(sys.argv) or ROOT / "v2")
# Erratum R1.1 binary set locally (v8 trainer and panel, predictor from main 86fc02bb, admitted by
# benchmarks/rev4_featpot_v2_predictor_parity_2026-10-01.json); in a fit-cell container the executor links the same
# three binaries under target/debug.
_LOCAL = Path("/var/tmp/fitv2/bin-v2")
BIN_DIR = Path(os.environ.get("REV4_V2_BIN_DIR", str(_LOCAL if _LOCAL.is_dir() else ROOT / "target/debug")))
TRAINER = BIN_DIR / "zensim_mlp_train"
FITBIN = BIN_DIR / "bake_dial_refit"
PANEL = BIN_DIR / "panel"

# Evaluated human sources, in fold order. Member sets are bank names admitted by admit_bank.
SOURCES = {
    "kadid": ("kadid_train", "kadid_select"),
    "tid2013": ("tid2013",),
    "konfig": ("konfig_train", "konfig_val"),
    "cid22_a25": ("cid22_a25",),
    "aic3": ("aic3",),
}
SOURCE_ORDER = tuple(SOURCES)
# R915 teacher legs (never evaluated): bank set, R915 dedup-table leg, validation weight.
TEACHERS = {"safesyn": ("safesyn", "safesyn", 0.5), "cid22": ("cid22_train", "cid22", 2.0)}
R915_TABLES = Path("/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables")
TEACHER_PIN = REPO / "benchmarks/rev4_featpot_v2_teacher_pin_2026-09-30.json"
NOMINAL_WEIGHT = {"safesyn": 1.0, "cid22": 1.0, "human": 0.5}
HUMAN_VAL_WEIGHT = 1.0
HUMAN_DEV_MODULUS = 5  # label-free: a human reference is dev when sha256(ref) mod 5 == 0

INIT_SEEDS = (1101, 1103, 1107, 1109, 1117, 1123, 1129, 1151, 1153, 1163)
SAMPLE_SEEDS = tuple(101 + k * 100_000_000 for k in range(10))
# Seed indices 10-19 (design log E23, 2026-10-04): a second, disjoint stream. Indices 0-9 keep their exact (init, sample)
# pairs, including the fold rotation modulo 10, so every existing cell stays reproducible.
INIT_SEEDS_EXT = (1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231)
SAMPLE_SEEDS_EXT = tuple(101 + (10 + k) * 100_000_000 for k in range(10))
N_SEEDS = len(INIT_SEEDS) + len(INIT_SEEDS_EXT)
EPOCHS = 120
# Which epoch's weights a cell keeps (design log E5/E5b, amendment R4): "best_dev" = the trainer's best dev-aggregate
# epoch (R1-R3); "last" = the final-epoch checkpoint (dumped by the trainer, identical to its state after training).
EPOCH_RULE = "last"  # amendment R4 (design log E5/E5b)
PAIRS_PER_EPOCH = 50_000
HIDDEN = 32
HEADS = ("N", "F")
# E33 Arm C (`fx1` recipe token): the registered product declaration (benchmarks/e33_registration_2026-10-09.md
# section 5.1), pinned by content so a cell can never train against an edited copy.
FX1_DECLARATION = Path(__file__).resolve().parent / "e33_fx1_declaration.json"
FX1_SHA256 = "a97a5fa63937ed8fade3c56fd7470765a031d1d309afc36e631f371a1796e648"
FX2_DECLARATION = Path(__file__).resolve().parent / "color_noise_fx2_declaration.json"
FX2_SHA256 = "a3802ddeeede6ec5c9e3d03efdd6cc5af683ab4ea2b0d1143c8e9183c5bd46e5"
# By-construction exploration (benchmarks/byconstruction_2026-10-22.md): fx1 without the v2 ringing slots.
FX1NR_DECLARATION = Path(__file__).resolve().parent / "byc_fx1nr_declaration.json"
FX1NR_SHA256 = "923da70c2ccdfcdf2d34192a8b7f6679dd27d918b55ff1c7827e5da5234cb740"
# By-construction exploration: fx1 plus the byc_globaff global affine-change descriptor (research ids f1867-f1890).
FX1G_DECLARATION = Path(__file__).resolve().parent / "byc_fx1g_declaration.json"
FX1G_SHA256 = "2fa7a8c8a908406e101b705d061cb41cfb6f36a31afd815f8ef1d3529fd2d2a0"
# fx1g restricted to pyramid scales 1-3 (no scale-0 slot read; byconstruction AIC2026 TRAIN compute arm)
FX1GS1_DECLARATION = Path(__file__).resolve().parent / "byc_fx1gs1_declaration.json"
FX1GS1_SHA256 = "a8d224e465ed9014793496926630399bf189ec1e6e28b966feb9538493269b52"
# fx1 plus gated damage (research ids f1891-f2300: each difference x (1 - byc_globaff share of its cell)); every input
# monotone, the descriptor itself not read (byconstruction global-edit inversion fix)
FX1GG_DECLARATION = Path(__file__).resolve().parent / "byc_fx1gg_declaration.json"
FX1GG_SHA256 = "c009fbedd3fbd2678720ff2a2b2bf89845a4f571e0909b36ac68caeb766a273a"
# fx1 plus gated luminance damage (research ids f1891-f2054: each Y-cell difference x (1 - byc_globaff share of its
# cell)); X and B damage ungated (byconstruction luminance-gate arm; `ring_root.py gated-decl --channels Y`)
FX1GY_DECLARATION = Path(__file__).resolve().parent / "byc_fx1gy_declaration.json"
FX1GY_SHA256 = "d850c8a89ff1553a8db543a9adc4c34fe6a3dc33241272db1a9e8125b666f1c2"
# fx1 plus the byc_orient damage (research ids f2301-f2336: Haar H/V/D band-difference RMS per scale x channel), direct
# only (byconstruction orientation-band arm)
FX1O_DECLARATION = Path(__file__).resolve().parent / "byc_fx1o_declaration.json"
FX1O_SHA256 = "155be1f8019396e7fc350d61754947d0002884c314d30e1461e5d97eb0b31b3d"


def fx1_declaration() -> dict:
    """The pinned fx1 declaration; refuses a changed file."""
    data = FX1_DECLARATION.read_bytes()
    if hashlib.sha256(data).hexdigest() != FX1_SHA256:
        raise ValueError(f"{FX1_DECLARATION}: content changed after registration pin")
    return json.loads(data)


def derived_declaration(token):
    """Pinned experiment declaration; existing fx1 bytes remain unchanged."""
    if token == 'fx1':return FX1_DECLARATION, FX1_SHA256, fx1_declaration()
    if token == 'fx1nr':
        data=FX1NR_DECLARATION.read_bytes()
        if hashlib.sha256(data).hexdigest()!=FX1NR_SHA256:raise ValueError('fx1nr declaration changed')
        return FX1NR_DECLARATION, FX1NR_SHA256, json.loads(data)
    if token == 'fx1gg':
        data=FX1GG_DECLARATION.read_bytes()
        if hashlib.sha256(data).hexdigest()!=FX1GG_SHA256:raise ValueError('fx1gg declaration changed')
        return FX1GG_DECLARATION, FX1GG_SHA256, json.loads(data)
    if token == 'fx1gy':
        data=FX1GY_DECLARATION.read_bytes()
        if hashlib.sha256(data).hexdigest()!=FX1GY_SHA256:raise ValueError('fx1gy declaration changed')
        return FX1GY_DECLARATION, FX1GY_SHA256, json.loads(data)
    if token == 'fx1gs1':
        data=FX1GS1_DECLARATION.read_bytes()
        if hashlib.sha256(data).hexdigest()!=FX1GS1_SHA256:raise ValueError('fx1gs1 declaration changed')
        return FX1GS1_DECLARATION, FX1GS1_SHA256, json.loads(data)
    if token == 'fx1o':
        data=FX1O_DECLARATION.read_bytes()
        if hashlib.sha256(data).hexdigest()!=FX1O_SHA256:raise ValueError('fx1o declaration changed')
        return FX1O_DECLARATION, FX1O_SHA256, json.loads(data)
    if token == 'fx1g':
        data=FX1G_DECLARATION.read_bytes()
        if hashlib.sha256(data).hexdigest()!=FX1G_SHA256:raise ValueError('fx1g declaration changed')
        return FX1G_DECLARATION, FX1G_SHA256, json.loads(data)
    if token != 'fx2':raise ValueError('unknown derived-input recipe: '+str(token))
    data=FX2_DECLARATION.read_bytes()
    if hashlib.sha256(data).hexdigest()!=FX2_SHA256:raise ValueError('fx2 declaration changed after registration')
    return FX2_DECLARATION, FX2_SHA256, json.loads(data)

CANDIDATES = ("c1", "c2", "c3", "c4", "all", "csfw", "p1", "p3",
              "c8n", "a1", "a1m", "b2", "b2m")
CALIBRATION = ("oracle_lo", "oracle_hi", "minus_basic")
# Amendment R2.3: one all-columns model per family table (no permuted controls): screen_main = bank + every main-family candidate
# column + every appended (extra) column; screen_aux = bank + gmsd + gmsm + gmsbank.
SCREEN = ("screen_main", "screen_aux")
ORACLE_SIGMA = {"oracle_lo": 1.5, "oracle_hi": 0.5}
FAMILIES = ("main", "aux")
WIDTH = 1825
# Design log E9: the seven legacy blocks of the Rev3 944 bank (FEATURE_SET_IDS.md) and the lean core (basic + peaks,
# R915's 228-column regime). Block specs: `r0-<block>` (R0 without the block), `core`, `core+<block or arm>`.
LEGACY_BLOCKS = {"basic": range(0, 156), "peaks": range(156, 228), "masked": range(228, 300), "iw": range(300, 372),
                 "v2": range(372, 720), "append": range(720, 924), "append2": range(924, 944)}
CORE = range(0, 228)
AUX_PEERS = {"gmsd": 944, "gmsm": 945}
AUX_ORACLE = {"oracle_lo": 946, "oracle_hi": 947}
AUX_GMSBANK = tuple(range(1322, 1502))
AUX_ADDED = tuple(sorted({*AUX_PEERS.values(), *AUX_ORACLE.values(), *AUX_GMSBANK}))
N_PERMS = 3
VARIANTS = ("real", *(f"p{k}" for k in range(1, N_PERMS + 1)))
PERM_SEED_BASE = 20260930
ORACLE_SEED_BASE = 20260930
BOOT_B = 2000
BOOT_SEED = 20260930
REPLAY = "Rev4 POTENTIAL Instrument v2 diagnostic: pinned Rev3 944 plus pinned sidecars; never ship"


def table_path(record: dict) -> Path:
    """A receipt record's table file: the root-relative `rel` of a v2-canon receipt (so the tree can be promoted or packed
    anywhere), else the absolute `path` of a Rev3 v2 receipt."""
    return V2 / record["rel"] if "rel" in record else Path(record["path"])


FROZEN_SCHEMA = "rev4-featpot-v2c-frozen-v1"


def load_frozen(root: Path | None = None, *, training_only: bool = False, metadata_only: bool = False) -> tuple[dict, str]:
    """(record, sha256) of `<root>/wide/frozen.json` after re-hashing every file it pins (confirm receipt, wide receipts,
    keep lists, extra arms). Raises if the canon tables changed after the freeze or no freeze exists."""
    root = V2 if root is None else root
    path = Path(root) / "wide" / "frozen.json"
    if not path.is_file():
        raise ValueError(f"{path}: the canon root is not frozen (run `v2c_wide.py freeze`)")
    record = json.loads(path.read_text())
    if training_only and record.get("schema") == "rev5-recipe-admission-freeze-v1":
        pins = {**{f"wide/{k}/receipt.json": v for k, v in record["wide_receipts"].items()},
                "wide/keep_lists.json": record["keep_lists_sha256"], **record["auxiliary_files"]}
        for rel, want in pins.items():
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts or any(p.startswith("_sealed") for p in path.parts):
                raise ValueError("admission freeze contains an unsafe relative path")
            if metadata_only and ((path.suffix == ".parquet" and not path.name.endswith(".keys.parquet")) or path.name.endswith(".pairs.json")):
                continue
            if sha(Path(root) / path) != want:
                raise ValueError(f"{rel}: changed after the admission freeze")
        return record, sha(Path(root) / "wide/frozen.json")
    if record.get("schema") != FROZEN_SCHEMA:
        raise ValueError(f"{path}: unexpected schema")
    pins = {**{f"wide/{k}/receipt.json": v for k, v in record["wide_receipts"].items()},
            "wide/confirm/receipt.json": record["confirm_receipt_sha256"], "wide/keep_lists.json": record["keep_lists_sha256"]}
    if record.get("extra_arms_sha256"):
        pins["wide/extra_arms.json"] = record["extra_arms_sha256"]
    for rel, want in pins.items():
        if sha(Path(root) / rel) != want:
            raise ValueError(f"{rel}: changed after the freeze")
    return record, sha(path)


def refuse_immutable_output(path: Path, roots) -> Path:
    """Refuse resolved equality/ancestry, including symlinks and fresh children."""
    from v2c_wide import safe_path
    path = safe_path(path)
    resolved = path.resolve()
    for root in roots:
        root = safe_path(root).resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(f"{path}: output lies inside immutable input root {root}")
    return path


def admission_input_roots(root: Path) -> tuple[Path, ...]:
    """Read all protected roots from the hash-bound training admission receipt.

    Older admission views without a bank-root binding must be regenerated at
    a fresh location; never amend their frozen receipt in place.
    """
    from v2c_wide import safe_path
    root = safe_path(root)
    frozen, _ = load_frozen(root, training_only=True, metadata_only=True)
    receipt = json.loads((root / "wide/main/real/receipt.json").read_text())
    view = receipt.get("admission_view", {})
    if (frozen.get("schema") != "rev5-recipe-admission-freeze-v1"
            or frozen.get("admission_view") != view or view.get("schema") != "rev5-recipe-admission-v1"):
        raise ValueError("strict outputs require a bound full-recipe admission record")
    roots = [root.resolve()]
    for key in ("source_root", "bank_root"):
        value = view.get(key)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(f"admission record lacks an absolute {key}; regenerate a fresh admission view")
        roots.append(safe_path(value).resolve())
    for value in view.get("immutable_input_roots", []):
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError("admission record contains an invalid immutable input root")
        roots.append(safe_path(value).resolve())
    return tuple(roots)


def table_input_roots(table: Path) -> tuple[Path, ...]:
    """Carry immutable ancestry through selected/curated table declarations."""
    from v2c_wide import safe_path
    declaration = json.loads(safe_path(Path(f"{table}.manifest.json")).read_text())
    if declaration.get("admission_root"):
        return admission_input_roots(Path(declaration["admission_root"]))
    return tuple(safe_path(p).resolve() for p in declaration.get("immutable_input_roots", []))


def split_weight(spec: str) -> tuple[str, float | None]:
    """'oracle_hi~p1@h2' -> ('oracle_hi~p1', 2.0): the instrument-retune sweep's human-weight override (design log E2).
    No suffix -> (spec, None), i.e. NOMINAL_WEIGHT['human']."""
    core, _, w = spec.partition("@h")
    if not w:
        return core, None
    value = float(w.split(":")[0])  # ":H<n>:gl<x>" recipe tokens follow the weight (recipe_of, design log E8)
    if not 0 < value <= 64:
        raise ValueError(f"bad human weight in {spec!r}")
    return core, value


# Design log E13: curated SafeSyn teacher legs, recipe token `ts<name>`. Each rule applies to the SafeSyn fit rows only
# (the dev leg, the CID22 teacher and the human legs are unchanged); the leg keeps its train weight, so a subset concentrates
# the same pair budget on fewer rows. Codec/quality come from the per-row strata file (v2_teacher.STRATA_NAME).
#   none: no SafeSyn fit leg.  win: floor targets at the receipt's lower bound.  floor0: floor targets at 0.
#   neg: drop targets < 0.  q20: drop q <= 15.  mono5: drop every (reference, codec) series whose SSIM2 falls by > 5 between
#   consecutive qualities.  x<codec>: drop one codec.
TEACHER_CODECS = {"moz": "mozjpeg-rs-420-e4", "avif": "zenavif-s5-e6", "zj": "zenjpeg-420-e2", "xyb": "zenjpeg-420-xyb-e2",
                  "jxl": "zenjxl-e7", "webp": "zenwebp-default-m4"}
TEACHER_SUBSETS = ("none", "win", "floor0", "neg", "q20", "mono5", *(f"x{c}" for c in TEACHER_CODECS))
# Design log E15: the coverage pool's families (KADIS rule-compliant types + our TID-style `lbw`/`cab`); a cell picks families with
# `cf<hex mask>` (bit i = the i-th family below) and weights the leg with `cv<w>`.
COVERAGE_FAMILIES = {"blur": (1, 2, 3), "colour": (4, 5, 7), "noise": (11, 12, 13, 14), "light": (16, 17, 18),
                     "spatial": (19, 20, 21, 23), "contrast": (24, 25), "quantize": (22,), "new": ("lbw", "cab")}
# Design log E14: `ko<w>` adds the KADIS ordinal ladder leg (v2_teacher.ORDINAL_NAME) at nominal weight w, rank-only within ladders.


def recipe_of(spec: str) -> dict:
    """Training-recipe tokens after the human weight (design log E8): 'r0@h32:H128:gl0.0001' -> {'hidden': 128,
    'group_l1': 0.0001}; E13 adds 'ts<name>' -> {'teacher_subset': name}. No tokens -> {} (the registered recipe: H = HIDDEN,
    no group lasso, full teachers), so every existing spec, cell name and trainer argv is unchanged."""
    _, _, w = spec.partition("@h")
    out: dict = {}
    for tok in w.split(":")[1:]:
        if tok.startswith("H") and tok[1:].isdigit() and (int(tok[1:]) == 1 or 8 <= int(tok[1:]) <= 512) and "hidden" not in out:
            out["hidden"] = int(tok[1:])
        elif tok.startswith("gl") and "group_l1" not in out and 0 < float(tok[2:]) <= 100:
            out["group_l1"] = float(tok[2:])
        elif tok.startswith("ts") and tok[2:] in TEACHER_SUBSETS and "teacher_subset" not in out:
            out["teacher_subset"] = tok[2:]
        elif tok.startswith("ko") and "kadis_ordinal" not in out and 0 < float(tok[2:]) <= 64:
            out["kadis_ordinal"] = float(tok[2:])
        elif tok in ("s2o", "s2m") and "ssim2_recipe" not in out:
            out["ssim2_recipe"] = tok
        elif tok == "uh4" and "hdr_weight" not in out:
            out.update(hdr_weight=4.0, hdr_mode="rank", upiq380=True)
        elif tok.startswith("hd") and "hdr_weight" not in out and 0 < float(tok[2:]) <= 64:
            out["hdr_weight"] = float(tok[2:])
        elif tok[:2] in ("hp", "ha", "hb", "hc") and "hdr_weight" not in out and tok[2:] == "4":
            out.update(hdr_weight=4.0, hdr_mode={"hp": "rank", "ha": "withinref,both", "hb": "rank", "hc": "rank"}[tok[:2]], **({"hdr_consensus": tok} if tok[:2] in ("hb", "hc") else {}))
        elif tok == "sh1" and "smooth_hinge" not in out:
            out["smooth_hinge"] = True
        elif tok in ("fx1", "fx2", "fx1nr", "fx1g", "fx1gs1", "fx1gg", "fx1gy", "fx1o") and "derived_inputs" not in out:
            out["derived_inputs"] = tok
        elif tok == "mono" and "mono_all" not in out:  # by-construction exploration: W1 >= 0 on every input
            out["mono_all"] = True
        elif tok == "thr" and "thresholds" not in out:  # by-construction exploration: trainable b1 <= 0
            out["thresholds"] = True
        elif tok.startswith("rp") and tok[2:].isdigit() and int(tok[2:]) > 0 and "ramps" not in out:
            out["ramps"] = int(tok[2:])  # by-construction exploration: tied monotone ramp pairs
        elif tok.startswith("fr") and tok[2:].isdigit() and int(tok[2:]) > 0 and "free_ramps" not in out:
            out["free_ramps"] = int(tok[2:])  # by-construction exploration: budgeted sign-free ramp pairs
        elif tok.startswith("fb") and "free_budget" not in out and 0 <= float(tok[2:]) <= 1000:
            out["free_budget"] = float(tok[2:])
        elif tok == "gate" and "gated_products" not in out:  # content-gated monotone over fx1 products
            out["gated_products"] = True
        elif tok == "freering" and "free_mask" not in out:  # partial monotone: v2 ringing inputs sign-free
            out["free_mask"] = "free_ringing"
        elif tok == "freeglob" and "free_mask" not in out:  # partial monotone: byc_globaff descriptor inputs sign-free
            out["free_mask"] = "free_globaff"
        elif tok == "freeglobs1" and "free_mask" not in out:  # the same for fx1gs1 (descriptor scales 1-3)
            out["free_mask"] = "free_globaff_s1"
        elif tok == "dropring" and "free_mask" not in out:  # exactly monotone, v2 ringing inputs dropped
            out.update(free_mask="free_ringing", drop_masked=True)
        elif tok == "dropglobal" and "free_mask" not in out:  # exactly monotone, probe-flagged global-sensitive families dropped
            out.update(free_mask="global_sensitive", drop_masked=True)
        elif tok.startswith("mp") and tok[2:].isdigit() and int(tok[2:]) > 0 and "max_pairs" not in out:
            out["max_pairs"] = int(tok[2:])  # certified monotone pairwise max
        elif tok.startswith("cv") and "coverage_weight" not in out and 0 < float(tok[2:]) <= 64:
            out["coverage_weight"] = float(tok[2:])
        elif (tok.startswith("cf") and "coverage_mask" not in out
              and 0 < int(tok[2:], 16) < 1 << len(COVERAGE_FAMILIES)):
            out["coverage_mask"] = int(tok[2:], 16)
        else:
            raise ValueError(f"bad recipe token {tok!r} in {spec!r}")
    if ("free_ramps" in out) != ("free_budget" in out):
        raise ValueError(f"{spec!r}: fr<N> and fb<budget> go together")
    if ("coverage_weight" in out) != ("coverage_mask" in out):
        raise ValueError(f"{spec!r}: cv<w> and cf<mask> go together")
    return out


def extra_arms() -> dict:
    """Arms over columns appended after f1824 (v2-canon only): `<root>/wide/extra_arms.json`
    {"schema": "rev4-featpot-v2c-extra-arms-v1", "width": W, "arms": {name: [column ids, each 1825 <= id < W]}},
    written by `v2c_wide.py keeplists --extra-arm`. Absent file -> no extra arms (the Rev3 v2 root has none)."""
    path = V2 / "wide" / "extra_arms.json"
    if not path.is_file():
        return {"width": WIDTH, "arms": {}}
    record = json.loads(path.read_text())
    if record.get("schema") != "rev4-featpot-v2c-extra-arms-v1":
        raise ValueError(f"{path}: unexpected schema")
    return record


def parse_spec(spec: str) -> tuple[str, int]:
    """'c1' -> ('c1', 0); 'c1~p2' -> ('c1', 2); an '@h<w>' suffix is accepted and ignored here (split_weight).
    Permutations exist for candidates (and extra arms), oracle_lo and oracle_hi (the sweep's null)."""
    core, _ = split_weight(spec)
    if block_spec(core):  # design log E9: no permuted controls (a direct Δ against r0 or core)
        return core, 0
    base, _, perm = core.partition("~p")
    k = int(perm) if perm else 0
    extras = extra_arms()["arms"]
    if base not in ("r0", *CANDIDATES, *CALIBRATION, *SCREEN, *extras) or not 0 <= k <= N_PERMS:
        raise ValueError(f"bad v2 arm spec {spec!r}")
    if k and base not in (*CANDIDATES, *extras, "oracle_lo", "oracle_hi"):
        raise ValueError(f"{spec!r}: permuted controls exist only for candidates and the oracles")
    return base, k


def all_specs() -> list[str]:
    specs = ["r0", *CALIBRATION, *SCREEN, *(f"oracle_lo~p{k}" for k in range(1, N_PERMS + 1)),
             *(f"oracle_hi~p{k}" for k in range(1, N_PERMS + 1))]
    for arm in (*CANDIDATES, *extra_arms()["arms"]):
        specs += [arm, *(f"{arm}~p{k}" for k in range(1, N_PERMS + 1))]
    return specs


def selection_id(cols) -> str:
    """Name of a column subset in a `sel:<id>` spec (design log E9′ method 2 refits): the first 12 hex digits of the
    SHA-256 of the sorted column ids joined by commas. The ids themselves travel in the cell argv (`--columns`)."""
    return hashlib.sha256(",".join(map(str, sorted(cols))).encode()).hexdigest()[:12]


def block_spec(core: str) -> bool:
    """True for the E9 block specs: 'core', 'r0-<legacy block>', 'core+<legacy block | candidate arm | extra arm>',
    'set:<groups>', and 'sel:<12 hex>' (a column subset whose ids are passed with --columns)."""
    if core == "core":
        return True
    if core.startswith("sel:"):
        return len(core) == 16 and all(c in "0123456789abcdef" for c in core[4:])
    if core.startswith("set:"):  # E9″: any groups, basic and peaks included, nothing exempt
        parts = core[4:].split("+")
        ok = (*LEGACY_BLOCKS, *[a for a in (*CANDIDATES, *extra_arms()["arms"]) if a not in ("all")])
        return bool(core[4:]) and len(parts) == len(set(parts)) and all(x in ok for x in parts)
    if core.startswith("r0-"):
        return core[3:] in LEGACY_BLOCKS
    if core.startswith("core+"):
        parts = core[5:].split("+")
        ok = (*LEGACY_BLOCKS, *[a for a in (*CANDIDATES, *extra_arms()["arms"]) if a not in ("all")])
        return len(parts) == len(set(parts)) and all(x in ok for x in parts)
    return False


def pinned_arm(arm: str) -> tuple[str, str, list[int]]:
    """(table family, variant, kept columns) of one registered arm, read from the root's pinned keep lists
    (`wide/keep_lists.json`: sha-pinned by frozen.json and shipped in the fit-data archive). The fit program does not
    pack restore_data, so E9 `set:`/`core+` specs resolve their candidate arms here (2026-10-02: E9″ round 1 failed
    every cell with `No module named 'restore_data'`). A root without keep lists, or an arm they lack, falls back to
    arm_columns, which needs restore_data and fails loudly where it is absent."""
    path = V2 / "wide" / "keep_lists.json"
    if path.is_file():
        lists = json.loads(path.read_text())
        if lists.get("schema") != "rev4-featpot-v2-keeplists-v2":
            raise ValueError(f"{path}: keep-list schema mismatch")
        entry = lists["specs"].get(arm)
        if entry is not None:
            return entry["family"], entry["variant"], list(entry["keep"])
    return arm_columns(arm)


def arm_columns(spec: str) -> tuple[str, str, list[int]]:
    """(table family, variant, kept wide-column indices) for a spec (any '@h' suffix ignored)."""
    base, k = parse_spec(spec)
    if base.startswith("sel:"):
        raise ValueError(f"{spec}: a sel: spec carries its columns in the cell argv (--columns), not in a registry")
    variant = f"p{k}" if k else "real"
    bank = list(range(944))
    if base == "r0":
        return "main", variant, bank
    if base == "core":
        return "main", variant, list(CORE)
    if base.startswith("r0-"):
        drop = set(LEGACY_BLOCKS[base[3:]])
        return "main", variant, [c for c in bank if c not in drop]
    if base.startswith("core+") or base.startswith("set:"):  # E9′ core + S + X; E9″ any set of groups, no exempt core
        cols, fams = (set(CORE), set()) if base.startswith("core+") else (set(), set())
        for x in base.partition("+")[2].split("+") if base.startswith("core+") else base[4:].split("+"):
            if x in ("all"):
                raise ValueError(f"{spec}: unions are not lean-base arms")
            if x in LEGACY_BLOCKS:
                cols |= set(LEGACY_BLOCKS[x])
                continue
            fam, _, ids = pinned_arm(x)
            fams.add(fam)
            cols |= {c for c in ids if c >= 944}
        if len(fams) > 1:  # the peer pair exists only in the aux table, the main research columns only in main
            raise ValueError(f"{spec}: mixes aux-table and main-table families")
        return (fams.pop() if fams else "main"), variant, sorted(cols)
    if base == "minus_basic":
        return "main", variant, list(range(228, 944))
    if base in AUX_ORACLE:
        return "aux", variant, bank + [AUX_ORACLE[base]]
    extras = extra_arms()
    if base == "screen_aux":
        return "aux", variant, bank + [AUX_PEERS["gmsd"], AUX_PEERS["gmsm"], *AUX_GMSBANK]
    if base == "screen_main":
        cols = set()
        for arm in CANDIDATES:
            fam, _, ids_ = arm_columns(arm)
            if fam == "main":
                cols |= set(ids_[944:])
        for ids_ in extras["arms"].values():
            cols |= set(ids_)
        return "main", variant, bank + sorted(cols)
    if base in extras["arms"]:  # columns appended after f1824 (v2-canon)
        ids = extras["arms"][base]
        if any(not WIDTH <= c < extras["width"] for c in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{spec}: extra-arm columns outside {WIDTH}..{extras['width']}")
        return "main", variant, bank + list(ids)
    import restore_data  # registered arm definitions (pinned JSONs); not in the fit program, so only reached here

    ids = restore_data.arm_ids(base)
    if any(c < 0 for c in ids):  # the peer pair: arm p3 lives in the aux family
        added = [AUX_PEERS["gmsd"] if c == -1 else AUX_PEERS["gmsm"] if c == -2 else c for c in ids]
        if not set(added) <= set(AUX_ADDED):
            raise ValueError(f"{spec}: peer arm columns outside the aux family")
        return "aux", variant, bank + added
    if any(not 944 <= c < WIDTH for c in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{spec}: added columns outside the main layout")
    return "main", variant, bank + list(ids)


def seeds(heldout: str, seed_index: int) -> tuple[int, int]:
    fold = SOURCE_ORDER.index(heldout)
    if not 0 <= seed_index < N_SEEDS:
        raise ValueError(f"seed index {seed_index} outside 0..{N_SEEDS - 1}")
    if seed_index < len(INIT_SEEDS):
        return INIT_SEEDS[seed_index], SAMPLE_SEEDS[(seed_index + fold) % len(SAMPLE_SEEDS)]
    j = seed_index - len(INIT_SEEDS)
    return INIT_SEEDS_EXT[j], SAMPLE_SEEDS_EXT[(j + fold) % len(SAMPLE_SEEDS_EXT)]


def confirm_seeds(seed_index: int) -> tuple[int, int]:
    """Full-data (confirmatory) fits hold out no source, so there is no fold offset: init[i], sample[i]."""
    return INIT_SEEDS[seed_index], SAMPLE_SEEDS[seed_index]


def human_dev(ref: str) -> bool:
    return int(hashlib.sha256(ref.encode()).hexdigest(), 16) % HUMAN_DEV_MODULUS == 0


def acceptance_weight(nominal: float, refs) -> float:
    """R915's within-ref correction: w / mean over refs with n >= 2 of (1 - 1/n)."""
    import collections
    counts = collections.Counter(refs)
    eligible = [n for n in counts.values() if n >= 2]
    return nominal / (sum(1 - 1 / n for n in eligible) / len(eligible))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ Rev5 tables: absent slots are NaN (spec rev5_spec_2026-10-04.md §6)
def table_revision(table: Path) -> int:
    """Formula revision a wide table declares in its sidecar manifest (Rev4 tables predate the field's use: 4)."""
    man = Path(f"{table}.manifest.json")
    return int(json.loads(man.read_text()).get("formula_revision", 4)) if man.is_file() else 4


def refuse_nonfinite_kept(paths, keep) -> None:
    """Every kept column must be finite in every table a cell trains or validates on. A Rev5 table marks absent slots NaN;
    a keep list that reads one is refused here, before the trainer runs."""
    import numpy as np
    import pyarrow.parquet as pq
    cols = [f"f{i}" for i in keep]
    for path in paths:
        selected = cols
        declaration = Path(f"{path}.manifest.json")
        if declaration.is_file() and "research_palette" in json.loads(declaration.read_text()):
            from e32_palette import feature_columns
            selected = feature_columns(path, keep)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=selected):
            for j in range(batch.num_columns):
                if not np.isfinite(batch.column(j).to_numpy(zero_copy_only=False)).all():
                    raise ValueError(f"{path}: kept column {selected[j]} is not finite (an absent slot?); refusing this keep list")


def dense_bake(bake: Path, cache: Path, *, research_palette_cached=False) -> Path:
    """The bake rewritten to the dense contract by the owner (`bake_dial_refit densify`, identity gate: predictions
    bit-identical on its probe rows). Cell bakes are identity-width, so a zero-weight input still multiplies its NaN; on a
    table with NaN absent slots only the dense bake (exactly the inputs it reads) scores finite."""
    import subprocess
    import threading
    out = Path(cache) / "dense" / f"{hashlib.sha256(Path(bake).read_bytes()).hexdigest()[:16]}.bin"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        r = subprocess.run([str(FITBIN), "densify", "--in", str(bake), "--out", str(tmp), *(["--research-palette-cached"] if research_palette_cached else [])], check=True, capture_output=True,
                           text=True)
        if "BIT-IDENTICAL" not in r.stdout + r.stderr:
            raise ValueError(f"densify identity gate did not report bit-identical predictions for {bake}")
        tmp.rename(out)
    return out
