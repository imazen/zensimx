"""v2-canon wide tables from the re-extracted Rev4 bank (CANONTAB lane, 2026-10-01).

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md (R1, R1.1, R2, R3). POTENTIAL — ceiling, not a
model score. The Rev3 builder `v2_wide.py` is untouched, so the Rev3 tables stay reproducible; this one reads
`/var/tmp/rev4-featbank-r4/<set>/{features.parquet,keys.parquet,_MANIFEST.json}` (pair_key, row_id, f0..f1824, float64,
formula Rev4, era tiercanon_c3negfold) and writes the same table layout under `--out`.

  python v2c_wide.py build    [--legs human|teachers|all] [--family F] [--variant V] [--extra FAM=FIRST:WIDTH:TMPL]
  python v2c_wide.py confirm  [--family F]      # features-only tables for the sealed confirmatory sets
  python v2c_wide.py keeplists [--extra-arm NAME=ID,ID,...]
  python v2c_wide.py verify                      # the registered gates (cast equality, row/key order, permutations)
  python v2c_wide.py freeze                      # after a clean verify of the FINAL build: pins every receipt in wide/frozen.json

Layout (same two 1,825+ wide families as R1.1, so kept columns keep identical first-layer initial weights):
  main: f0..f1824 straight from the bank (float64 cast to float32) + appended sidecar families (f1825.. width W)
  aux : bank f0..f943, gmsd f944, gmsm f945, oracle_lo f946, oracle_hi f947 (R3 distance orientation),
        gmsbank f1322..f1501, zeros elsewhere (width W). Needs the REEXTRACT peer output; without it only main builds.
Legs (as v2): five human sources (full table, keys, human_without_<held> fit/dev) + `human_all` fit/dev (union of the
five, same dev rule) + the two R915 teacher legs. pixels_identical keys are dropped everywhere. Human labels come from
the OLD bank's labels__*.parquet through `data.load`, exactly as v2 does; teacher SSIMULACRA2 targets come from the old
bank's labels__ssim2_oracle by pair_key under the committed teacher pin (the R915 dedup tables carry no pair_key; they
define the fit/dev reference split, and `verify` checks their targets against the bank's per reference).

SEALED LABELS: nothing here opens anything under `_sealed`. The confirmatory tables read only features.parquet,
keys.parquet and _MANIFEST.json of the Rev4 bank, through `bank_file`, which refuses any other path.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v2_common import (AUX_GMSBANK, AUX_ORACLE, AUX_PEERS, FAMILIES, N_PERMS, ORACLE_SEED_BASE, ORACLE_SIGMA,
                       PERM_SEED_BASE, REPO, SOURCE_ORDER, SOURCES, TEACHERS, VARIANTS, WIDTH, human_dev, sha)

CANON_BANK = Path("/var/tmp/rev4-featbank-r4")
CANON_PEERS = CANON_BANK / "peer_gmsd"
OUT_DEFAULT = Path("/var/tmp/canontab/v2c")
SCHEMA = "rev4-featpot-v2c-wide-v1"
CONFIRM_SCHEMA = "rev4-featpot-v2c-confirm-v1"
# What the re-extraction (REEXTRACT lane) pinned; every manifest is refused if it differs.
CANON_FEATURE_SET_ID = None  # Historical producer identity is no longer admitted.
CANON_ERA = "tiercanon_c3negfold"
CANON_BINARY = "8c6f4c03695660fbb4fdce46e05859b40d9080497622cd726a7db75fd8323aa8"
CANON_BUILD = "259045b0acc7045cb3d4804217b81a9bfa1cbbad"
CANON_ERAS_TAIL = "c3negfold"
FORMULA_REVISION = 4
HUMAN_SETS = tuple(n for names in SOURCES.values() for n in names)
CONFIRM_SETS = ("cid22_b", "aic4", "konjnd_jpeg_select", "konjnd_jpeg_terminal", "csiq", "mcljci")
BANK_FILES = ("features.parquet", "keys.parquet", "_MANIFEST.json")
LEG_ORDER = (*SOURCE_ORDER, *TEACHERS)  # leg index drives the permutation and oracle seeds, as v2_wide


@dataclass(frozen=True)
class BankProfile:
    """What a bank must be. Rev4: every slot measured and the canon pins above. Rev5 (spec rev5_spec_2026-10-04.md): only
    the requested basic + peaks + v2 slots are measured, every other slot is NaN (absent), the pins are passed on the command
    line once the Rev5 extractor exists, Rev4 sidecar extras and the aux family are refused (they would mix revisions), and
    tables are NaN-padded to `pad_to` so kept columns get the same first-layer initial weights as the Rev4 tables."""
    revision: int
    schema: str
    era: str
    feature_set_id: str
    binary: str
    build: str
    eras_tail: str | None
    requested: tuple | None
    pad_to: int | None = None


REV4_PROFILE = BankProfile(4, "rev4-featbank-r4-v1", CANON_ERA, CANON_FEATURE_SET_ID, CANON_BINARY, CANON_BUILD, CANON_ERAS_TAIL,
                           None)
PROFILE = REV4_PROFILE


def rev5_profile(era: str, fsid: str, binary: str, build: str, pad_to: int | None, revision: int = 5) -> BankProfile:
    """A rev5_bank.py bank: revision 5, or revision 7 (by-construction exploration, raw ringing; the same bank layout)."""
    from rev5_bank import SCHEMA as REV5_SCHEMA, SLOT_RANGES
    if not (era and fsid and binary and build):
        raise ValueError(f"--revision {revision} needs --expect-era, --expect-fsid, --expect-binary and --expect-build")
    return BankProfile(revision, REV5_SCHEMA, era, fsid, binary, build, None, tuple(tuple(r) for r in SLOT_RANGES), pad_to)


def requested_mask(width: int) -> np.ndarray:
    mask = np.zeros(width, dtype=bool)
    for lo, hi in PROFILE.requested or ((0, width),):
        mask[lo:hi] = True
    return mask
META = ["pair_key", "source_row_id", "ref_basename", "member_set", "target"]
KEY_COLUMNS = ["pair_key", "row_id", "ref_group", "pixels_identical"]


# ------------------------------------------------------------------ path guard (sealed labels)
def safe_path(path) -> Path:
    """Refuse any path with a `_sealed` component: the sealed label directory is never opened, listed or copied."""
    p = Path(path)
    if any(part == "_sealed" or part.startswith("_sealed") for part in (*p.parts, *p.resolve().parts)):
        raise PermissionError(f"refusing a path under _sealed: {p}")
    return p


def assessment_path(path) -> Path:
    p = safe_path(path)
    if any(v.startswith("labels__") for v in (*p.parts, *p.resolve().parts)):
        raise PermissionError("protected label path refused")
    return p


def bank_file(bank: Path, name: str, filename: str) -> Path:
    """The only way this module names a bank file: <bank>/<set>/<features|keys|manifest> of a known set."""
    if filename not in BANK_FILES:
        raise ValueError(f"{filename}: not a bank feature/key/manifest file")
    if name not in (*HUMAN_SETS, *CONFIRM_SETS, "cid22_train", "safesyn", "konjnd_bpg_train", "konjnd_bpg_val",
                    "kadid_terminal"):
        raise ValueError(f"{name}: not a Rev4 bank set")
    return safe_path(Path(bank) / name / filename)


# ------------------------------------------------------------------ extras (appended families)
@dataclass(frozen=True)
class Extra:
    name: str
    first: int
    width: int
    template: str  # path template with {set}

    def path(self, set_name: str) -> Path:
        return safe_path(self.template.format(set=set_name))

    @property
    def ids(self) -> list[int]:
        return list(range(self.first, self.first + self.width))


def parse_extras(specs: list[str]) -> list[Extra]:
    """`texgain=1825:12:/dir/{set}/features__signed_texgain.parquet`; contiguous from f1825 in the order given."""
    out, nxt = [], WIDTH
    for spec in specs or []:
        name, _, rest = spec.partition("=")
        first, width, template = rest.split(":", 2)
        extra = Extra(name, int(first), int(width), template)
        if extra.first != nxt or extra.width <= 0 or "{set}" not in template:
            raise ValueError(f"{spec}: extra families must be contiguous from f{nxt} and name {{set}}")
        nxt += extra.width
        out.append(extra)
    return out


def total_width(extras: list[Extra]) -> int:
    if PROFILE.revision != 4 and extras:
        raise ValueError("sidecar extras are Rev4 extractions; refusing to append them to a Rev5 bank")
    base = WIDTH + sum(e.width for e in extras)
    return max(base, PROFILE.pad_to or 0)


# ------------------------------------------------------------------ bank reading
@dataclass
class BankSet:
    name: str
    keys: pd.DataFrame      # KEY_COLUMNS in bank row order
    X: np.ndarray           # (rows, W) float32: f0..f1824 then appended families
    peers: np.ndarray | None  # (rows, 2) float32 gmsd, gmsm, or None
    receipt: dict


def check_manifest(manifest: dict, name: str) -> None:
    """Refuse a sidecar whose era/feature-set identity differs from the pinned canon (REEXTRACT consumer rule)."""
    P = PROFILE
    want = {"set": name, "schema": P.schema, "feature_width": WIDTH, "formula_revision": f"Rev{P.revision}",
            "era_label": P.era, "feature_set_id": P.feature_set_id, "binary_sha256": P.binary,
            "build_commit": P.build, "dtype": "float64"}
    if P.requested is not None:
        want["requested_slot_ranges"] = [list(r) for r in P.requested]
    for key, value in want.items():
        if manifest.get(key) != value:
            raise ValueError(f"{name}: manifest {key}={manifest.get(key)!r} differs from the pinned canon {value!r}")
    eras = manifest.get("formula_revision_eras") or []
    if P.eras_tail is not None and (not eras or eras[-1] != P.eras_tail):
        raise ValueError(f"{name}: formula_revision_eras does not end in {P.eras_tail}")


def read_matrix(path: Path, ids: list[int], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """features.parquet -> (float32 matrix of columns f<ids>, pair_key array) in batches; float64 -> float32 once."""
    parquet = pq.ParquetFile(safe_path(path))
    names = parquet.schema_arrow.names
    cols = [f"f{i}" for i in ids]
    if names[:2] != ["pair_key", "row_id"] or any(c not in names for c in cols):
        raise ValueError(f"{path}: unexpected schema")
    out = np.empty((n_rows, len(cols)), dtype=np.float32)
    keys = []
    at = 0
    for batch in parquet.iter_batches(batch_size=8192, columns=["pair_key", *cols]):
        n = batch.num_rows
        keys.append(batch.column(0).to_numpy(zero_copy_only=False))
        for j in range(len(cols)):
            out[at:at + n, j] = batch.column(j + 1).to_numpy(zero_copy_only=False)
        at += n
    if at != n_rows:
        raise ValueError(f"{path}: {at} rows, keys say {n_rows}")
    return out, np.concatenate(keys)


def gather_by_key(path: Path, columns: list[str], keys: np.ndarray, what: str) -> np.ndarray:
    """Columns of a per-set sidecar parquet (pair_key + columns), rows ordered by `keys`, float32; refuses missing,
    duplicate or non-finite rows."""
    table = pq.read_table(safe_path(path), columns=["pair_key", *columns]).to_pandas()
    if table.pair_key.duplicated().any():
        # a peer scorer emits one row per stimulus; collapsed stimuli repeat a key with equal values
        first = table.drop_duplicates("pair_key", keep="first")
        check = table.merge(first, on="pair_key", suffixes=("", "_first"), validate="many_to_one")  # joinsafety-ok: pair_key-keyed duplicate-consistency check, not a metric join
        for c in columns:
            if not (check[c] == check[f"{c}_first"]).all():
                raise ValueError(f"{what}: duplicate keys with different values in {c}")
        table = first
    index = pd.Index(table.pair_key).get_indexer(keys)
    if (index < 0).any():
        raise ValueError(f"{what}: {(index < 0).sum()} bank keys missing from {path}")
    values = table[columns].to_numpy()[index].astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{what}: non-finite values")
    return values


def load_bank_set(bank: Path, name: str, extras: list[Extra], peer_dir: Path | None = None) -> BankSet:
    manifest_path = bank_file(bank, name, "_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    check_manifest(manifest, name)
    keys_path, feats_path = bank_file(bank, name, "keys.parquet"), bank_file(bank, name, "features.parquet")
    keys_sha, feats_sha = sha(keys_path), sha(feats_path)
    if keys_sha != manifest["keys_sha256"] or feats_sha != manifest["features_parquet_sha256"]:
        raise ValueError(f"{name}: keys or features changed after the manifest")
    keys = pq.read_table(keys_path, columns=KEY_COLUMNS).to_pandas()
    if len(keys) != manifest["rows"] or keys.pair_key.duplicated().any() or not keys.row_id.is_monotonic_increasing:
        raise ValueError(f"{name}: keys disagree with the manifest")
    X, feat_keys = read_matrix(feats_path, list(range(WIDTH)), len(keys))
    if not np.array_equal(feat_keys, keys.pair_key.to_numpy()):
        raise ValueError(f"{name}: features.parquet and keys.parquet rows are not the same keys in the same order")
    mask = requested_mask(WIDTH)
    if not np.isfinite(X[:, mask]).all():
        raise ValueError(f"{name}: non-finite feature cell")
    if not np.isnan(X[:, ~mask]).all():
        raise ValueError(f"{name}: a slot outside the requested ranges carries a value (absent slots must be NaN)")
    receipt = {"rows": len(keys), "manifest_sha256": sha(manifest_path), "keys_sha256": keys_sha,
               "features_sha256": feats_sha, "feature_set_id": manifest["feature_set_id"],
               "extractor_build_commit": manifest["build_commit"], "extractor_binary_sha256": manifest["binary_sha256"],
               "identical_rows": int(keys.pixels_identical.sum())}
    if extras:
        X = np.concatenate([X, np.empty((len(keys), sum(e.width for e in extras)), np.float32)], axis=1)
        for e in extras:
            path = e.path(name)
            cols = [f"f{i}" for i in e.ids]
            X[:, e.first:e.first + e.width] = gather_by_key(path, cols, keys.pair_key.to_numpy(), f"{name}/{e.name}")
            receipt.setdefault("extras", {})[e.name] = {"path": str(path), "sha256": sha(path),
                                                         "first": e.first, "width": e.width}
    if PROFILE.pad_to and X.shape[1] < PROFILE.pad_to:
        X = np.concatenate([X, np.full((len(keys), PROFILE.pad_to - X.shape[1]), np.nan, np.float32)], axis=1)
        receipt["nan_padded_to"] = PROFILE.pad_to
    peers = None
    if peer_dir is not None:
        pp = safe_path(Path(peer_dir) / f"{name}.parquet")
        if pp.is_file():
            peers = gather_by_key(pp, ["gmsd", "gmsm"], keys.pair_key.to_numpy(), f"{name}/peers")
            receipt["peers"] = {"path": str(pp), "sha256": sha(pp)}
            manifest = safe_path(Path(peer_dir) / "_MANIFEST.json")
            if manifest.is_file():
                receipt["peers"]["manifest_sha256"] = sha(manifest)
    return BankSet(name, keys, X, peers, receipt)


# ------------------------------------------------------------------ legs
@dataclass
class Leg:
    name: str
    meta: pd.DataFrame            # META (+ split for teachers)
    X: np.ndarray                 # (n, W) float32
    peers: np.ndarray | None
    bounds: list[float]
    y01: np.ndarray
    oracle: dict                  # oracle_lo / oracle_hi float32 columns


def oracle_columns(target: np.ndarray, leg_index: int) -> tuple[list[float], np.ndarray, dict]:
    """Per-leg q0.001/q0.999 bounds, y01, and the distance-oriented oracle columns of amendment R3 (as v2_wide)."""
    target = np.asarray(target, dtype=np.float64)
    lo, hi = (float(v) for v in np.quantile(target, [0.001, 0.999]))
    y01 = np.clip((target - lo) / (hi - lo), 0.0, 1.0)
    rng = np.random.default_rng(ORACLE_SEED_BASE + leg_index)
    sd = float(np.std(y01))
    cols = {}
    for name in ("oracle_lo", "oracle_hi"):
        cols[name] = ((1.0 - y01) + rng.normal(0.0, ORACLE_SIGMA[name] * sd, len(target))).astype(np.float32)
    return [lo, hi], y01, cols


def rows_from_bank(stim: pd.DataFrame, bank: BankSet) -> tuple[np.ndarray, np.ndarray | None]:
    """Feature (and peer) rows for stimulus rows keyed by pair_key; refuses any key missing from the bank."""
    index = pd.Index(bank.keys.pair_key).get_indexer(stim.pair_key)
    if (index < 0).any():
        raise ValueError(f"{bank.name}: {(index < 0).sum()} stimulus keys missing from the Rev4 bank")
    return bank.X[index], (None if bank.peers is None else bank.peers[index])


def human_member(name: str, bank: BankSet) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """One member set: stimulus rows (old bank labels by key, as data.load), identical keys dropped, Rev4 features."""
    import restore_data
    from data import load as load_old
    frame, meta = load_old(name, features=False)  # pair_key, source_row_id, ref_basename, target; one row per stimulus
    bank.receipt["labels"] = {"file": meta["label_file"], "sha256": sha(Path(meta["label_file"])),
                              "target": meta["target"], "label_scale": meta["label_scale"]}
    flags = restore_data.identical_flags(name)
    old = frame.pair_key.map(flags).astype(bool).to_numpy()
    new = bank.keys.set_index("pair_key").pixels_identical.reindex(frame.pair_key).to_numpy()
    if np.isnan(new.astype(float)).any() or not np.array_equal(old, new.astype(bool)):
        raise ValueError(f"{name}: pixels_identical differs between the admitted table and the Rev4 keys")
    stim = frame.loc[~old].reset_index(drop=True)
    ref_group = bank.keys.set_index("pair_key").ref_group.reindex(stim.pair_key).to_numpy()
    if not np.array_equal(ref_group.astype(str), stim.ref_basename.astype(str).to_numpy()):
        raise ValueError(f"{name}: ref_basename differs from the Rev4 ref_group")
    X, peers = rows_from_bank(stim, bank)
    stim["member_set"] = name
    return stim[META[:2] + ["ref_basename", "member_set", "target"]], X, peers


def make_leg(name: str, meta: pd.DataFrame, X: np.ndarray, peers) -> Leg:
    bounds, y01, oracle = oracle_columns(meta.target.to_numpy(), LEG_ORDER.index(name))
    return Leg(name, meta.reset_index(drop=True), X, peers, bounds, y01, oracle)


def human_leg(source: str, bank: Path, extras: list[Extra], peer_dir: Path | None, receipts: dict) -> Leg:
    metas, xs, ps = [], [], []
    for member in SOURCES[source]:
        b = load_bank_set(bank, member, extras, peer_dir)
        receipts[member] = b.receipt
        m, x, p = human_member(member, b)
        metas.append(m), xs.append(x), ps.append(p)
        del b
    refsets = [set(m.ref_basename.astype(str)) for m in metas]
    if len(metas) > 1 and refsets[0] & refsets[1]:
        raise ValueError(f"{source}: member sets share references")
    peers = None if any(p is None for p in ps) else np.concatenate(ps)
    return make_leg(source, pd.concat(metas, ignore_index=True), np.concatenate(xs), peers)


def teacher_leg(leg: str, bank: Path, extras: list[Extra], peer_dir: Path | None, receipts: dict) -> Leg:
    """R915 teacher leg: Rev4 features, SSIMULACRA2 targets from the pinned old-bank label file by pair_key, split by the
    pinned R915 reference partition (fit / dev / excluded), identical keys dropped."""
    import v2_wide
    bank_name, _, _ = TEACHERS[leg]
    files = v2_wide.pinned(leg)
    b = load_bank_set(bank, bank_name, extras, peer_dir)
    receipts[bank_name] = b.receipt
    b.receipt["labels"] = {"teacher_pin": str(v2_wide.TEACHER_PIN), "teacher_pin_sha256": sha(v2_wide.TEACHER_PIN),
                           **{k: {"file": str(files[k]), "sha256": sha(files[k])} for k in ("labels", "r915_fit", "r915_dev")}}
    old_keys = pq.read_table(files["keys"], columns=["pair_key"]).to_pandas().pair_key.to_numpy()
    if not np.array_equal(old_keys, b.keys.pair_key.to_numpy()):
        raise ValueError(f"{leg}: Rev4 key order differs from the old bank's")
    labels = pq.read_table(files["labels"], columns=["pair_key", "source_row_id", "ssim2_oracle"]).to_pandas()
    index = pd.Index(b.keys.pair_key).get_indexer(labels.pair_key)
    if (index < 0).any() or labels.pair_key.duplicated().any():
        raise ValueError(f"{leg}: label keys do not map one-to-one into the Rev4 bank")
    ident = b.keys.pixels_identical.to_numpy()[index].astype(bool)
    fit = set(pq.read_table(files["r915_fit"], columns=["ref_basename"])["ref_basename"].to_pylist())
    dev = set(pq.read_table(files["r915_dev"], columns=["ref_basename"])["ref_basename"].to_pylist())
    ref_group = b.keys.ref_group.to_numpy()[index].astype(str)
    stripped = pd.Series(ref_group).str.split(":", n=1).str[1]
    split = np.where(stripped.isin(fit), "fit", np.where(stripped.isin(dev), "dev", "excluded"))
    keep = ~ident & (split != "excluded")
    sel = index[keep]
    meta = pd.DataFrame({"pair_key": labels.pair_key.to_numpy()[keep], "source_row_id": labels.source_row_id.to_numpy()[keep],
                         "ref_basename": ref_group[keep], "member_set": bank_name,
                         "target": labels.ssim2_oracle.to_numpy(dtype=np.float64)[keep], "split": split[keep]})
    peers = None if b.peers is None else b.peers[sel]
    X = b.X[sel]
    del b
    return make_leg(leg, meta, X, peers)


# ------------------------------------------------------------------ views, permutations, writing
def permute_within_reference(refs: np.ndarray, keys: np.ndarray, vals: np.ndarray, selector: np.ndarray, rng) -> np.ndarray:
    """Joint key-level permutation of the columns of `vals` within each reference (numpy form of
    restore_data._permute_within_reference, EFFAUDIT D7: same draws, same result; checked against it in the tests)."""
    new = vals.copy()
    for ref in sorted(set(refs.tolist())):
        rows = np.flatnonzero((refs == ref) & selector)
        _, first, inverse = np.unique(keys[rows], return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        source = rows[first[order]]
        new[rows] = vals[source[rng.permutation(len(order))]][rank[inverse.reshape(-1)]]
    return new


def aux_columns(width: int) -> list[int]:
    return sorted({*AUX_PEERS.values(), *AUX_ORACLE.values(), *AUX_GMSBANK})


def family_matrix(leg: Leg, family: str, width: int) -> tuple[np.ndarray, list[int]]:
    """(n, width) float32 matrix for a family and the permutable (added) column indices of that family."""
    if family != "main" and PROFILE.revision != 4:
        raise ValueError(f"family {family!r} is built from the Rev4 bank layout; refusing it at Rev{PROFILE.revision}")
    if family == "main":
        return leg.X, list(range(944, width))
    x = np.zeros((len(leg.X), width), dtype=np.float32)
    x[:, :944] = leg.X[:, :944]
    x[:, AUX_PEERS["gmsd"]] = leg.peers[:, 0]
    x[:, AUX_PEERS["gmsm"]] = leg.peers[:, 1]
    for name, col in AUX_ORACLE.items():
        x[:, col] = leg.oracle[name]
    x[:, AUX_GMSBANK[0]:AUX_GMSBANK[-1] + 1] = leg.X[:, AUX_GMSBANK[0]:AUX_GMSBANK[-1] + 1]
    return x, aux_columns(width)


def variant_matrix(leg: Leg, family: str, variant: str, width: int) -> np.ndarray:
    x, added = family_matrix(leg, family, width)
    k = 0 if variant == "real" else int(variant[1:])
    if not k:
        return x
    salt = 0 if family == "main" else 500
    rng = np.random.default_rng(PERM_SEED_BASE + salt + k + 1000 * LEG_ORDER.index(leg.name))
    out = x.copy()
    out[:, added] = permute_within_reference(leg.meta.ref_basename.to_numpy(), leg.meta.pair_key.to_numpy(),
                                             x[:, added], np.ones(len(x), dtype=bool), rng)
    return out


def write_table(root: Path, path: Path, ref: np.ndarray, score: np.ndarray, X: np.ndarray, family: str, note: str,
                width: int) -> dict:
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"f{i}" for i in range(width)]
    arrays = [pa.array(ref.astype(str)), pa.array(np.asarray(score, dtype=np.float64))]
    arrays += [pa.array(np.ascontiguousarray(X[:, j])) for j in range(width)]
    table = pa.Table.from_arrays(arrays, names=["ref_basename", "human_score", *names])
    # byte-stream-split floats, as v2_wide.write since d029bebc (decoded values identical; EFFAUDIT D6)
    pq.write_table(table, path, compression="zstd", use_dictionary=["ref_basename"],
                   use_byte_stream_split=["human_score", *names])
    layout = ((f"Rev4 bank f0-f1824 (formula Rev4, era {CANON_ERA})" + (f" + appended families to f{width - 1}" if width > WIDTH else "")
               if PROFILE.revision == 4 else
               f"Rev5 bank (era {PROFILE.era}): basic+peaks+v2 measured at f0-f227 and f372-f719, every other slot NaN, width {width}")
              if family == "main" else
              "bank f0-f943, gmsd f944, gmsm f945, oracle_lo f946, oracle_hi f947, gmsbank f1322-f1501, zeros elsewhere")
    Path(f"{path}.manifest.json").write_text(json.dumps({
        "source_bank_feature_set_id": PROFILE.feature_set_id,
        "composite": f"Rev{PROFILE.revision} POTENTIAL Instrument v2-canon {family} wide table ({layout}); diagnostic only; " + note,
        "formula_revision": PROFILE.revision}) + "\n")
    return {"rel": str(path.relative_to(root)), "sha256": sha(path), "manifest_sha256": sha(Path(f"{path}.manifest.json")),
            "rows": len(ref), "references": int(len(set(ref.astype(str).tolist())))}


def write_keys(path: Path, frame: pd.DataFrame, columns: list[str]) -> str:
    pq.write_table(pa.Table.from_pandas(frame[columns].reset_index(drop=True), preserve_index=False), safe_path(path),
                   compression="zstd")
    return sha(path)


def code_identity() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    here = Path(__file__).resolve().parent
    return {"build_commit": run(["jj", "log", "-r", "@-", "--no-graph", "-T", "commit_id"]),
            "working_copy_clean": run(["jj", "diff", "--stat", "scripts/rev4_featpot"]) == "",
            "script_sha256": {n: sha(here / n) for n in ("v2c_wide.py", "v2_common.py")}}


def build(bank: Path, out: Path, legs: str, families: list[str], variants: list[str], extras: list[Extra],
          peer_dir: Path | None) -> None:
    refuse_if_frozen(out)
    width = total_width(extras)
    names = set(legs.split(","))
    if "all" in names:
        names |= {"human", *TEACHERS}
    if "teachers" in names:
        names |= set(TEACHERS)
    if not names <= {"human", "all", "teachers", *TEACHERS}:
        raise ValueError(f"unknown --legs {legs!r}")
    wanted_human = "human" in names
    identity = code_identity()
    all_receipts: dict[str, dict] = {}
    human_parts: dict[tuple, dict] = {}
    recs: dict[tuple, dict] = {}
    todo = (list(SOURCE_ORDER) if wanted_human else []) + [t for t in TEACHERS if t in names]
    skipped_aux = None
    for leg_name in todo:
        bank_receipts: dict = {}
        leg = (human_leg if leg_name in SOURCES else teacher_leg)(leg_name, bank, extras, peer_dir, bank_receipts)
        all_receipts.update(bank_receipts)
        for family in families:
            if family == "aux" and leg.peers is None:
                skipped_aux = "peer columns (gmsd/gmsm) missing from the REEXTRACT peer output; aux not built"
                continue
            for variant in variants:
                x = variant_matrix(leg, family, variant, width)
                dest = out / "wide" / family / variant
                rec = {"bounds": leg.bounds}
                ref = leg.meta.ref_basename.to_numpy()
                if leg_name in SOURCES:
                    score = 100.0 * leg.y01
                    rec["full"] = write_table(out, dest / f"{leg_name}.parquet", ref, score, x, family,
                                              f"human source {leg_name} (evaluation)", width)
                    rec["keys_sha256"] = write_keys(dest / f"{leg_name}.keys.parquet", leg.meta, META)
                    dev = np.array([human_dev(r) for r in ref.astype(str)])
                    human_parts.setdefault((family, variant), {})[leg_name] = (leg.meta[META], x, score, dev)
                else:
                    score = leg.meta.target.to_numpy(dtype=np.float64)  # raw signed SSIMULACRA2, --target-scale 1
                    for split in ("fit", "dev"):
                        m = (leg.meta.split == split).to_numpy()
                        rec[split] = write_table(out, dest / f"{leg_name}_{split}.parquet", ref[m], score[m], x[m], family,
                                                 f"teacher {leg_name} {split}", width)
                        # row-identity sidecar (pair_key per table row) so verify can check every cell against the bank
                        rec[split]["keys_sha256"] = write_keys(dest / f"{leg_name}_{split}.keys.parquet", leg.meta[m], META)
                recs.setdefault((family, variant), {})[leg_name] = rec
                del x
        print(json.dumps({"leg": leg_name, "rows": len(leg.meta)}), flush=True)
        del leg
    for (family, variant), parts in human_parts.items():
        dest = out / "wide" / family / variant
        if set(parts) != set(SOURCE_ORDER):
            continue
        groups = {f"human_without_{held}": [s for s in SOURCE_ORDER if s != held] for held in SOURCE_ORDER}
        groups["human_all"] = list(SOURCE_ORDER)
        for gname, members in groups.items():
            rec = {}
            for split, want_dev in (("fit", False), ("dev", True)):
                chunks = [(parts[t][0][parts[t][3] == want_dev], parts[t][1][parts[t][3] == want_dev],
                           parts[t][2][parts[t][3] == want_dev]) for t in members]
                ref = np.concatenate([c[0].ref_basename.to_numpy() for c in chunks])
                rec[split] = write_table(out, dest / f"{gname}_{split}.parquet", ref, np.concatenate([c[2] for c in chunks]),
                                         np.concatenate([c[1] for c in chunks]), family,
                                         f"human training leg {gname} ({'+'.join(members)}), {split}", width)
            recs.setdefault((family, variant), {})[gname] = rec
    for (family, variant), legs_rec in recs.items():
        path = out / "wide" / family / variant / "receipt.json"
        old = json.loads(path.read_text()) if path.is_file() else {"legs": {}, "bank": {}}
        if old.get("schema") not in (None, SCHEMA) or old.get("width", width) != width:
            raise ValueError(f"{path}: existing receipt has another schema or width; build into a clean --out")
        old["legs"].update(legs_rec)
        old["bank"].update(all_receipts)
        old.update({"schema": SCHEMA, "label": "POTENTIAL — ceiling, not a model score", "family": family,
                    "variant": variant, "width": width, "feature_set_id": PROFILE.feature_set_id, "era": PROFILE.era,
                    "formula_revision": PROFILE.revision, "table_code": identity,
                    "extras": [{"name": e.name, "first": e.first, "width": e.width, "template": e.template} for e in extras],
                    "required_legs": [*SOURCE_ORDER, *(f"human_without_{h}" for h in SOURCE_ORDER), "human_all", *TEACHERS]})
        old["complete"] = all(leg in old["legs"] for leg in old["required_legs"])
        path.write_text(json.dumps(old, indent=1) + "\n")
        print(json.dumps({"receipt": str(path), "sha256": sha(path), "complete": old["complete"]}), flush=True)
    if skipped_aux:
        print(json.dumps({"aux_skipped": skipped_aux}), flush=True)


# ------------------------------------------------------------------ confirmatory tables (features only)
def build_confirm(bank: Path, out: Path, families: list[str], extras: list[Extra], peer_dir: Path | None,
                  variants: list[str] | None = None, sets: tuple[str, ...] = CONFIRM_SETS) -> dict:
    """Features-only tables for the sealed sets: `human_score` constant 0, keys without any target. Reads only
    features.parquet + keys.parquet + _MANIFEST.json of the Rev4 bank (bank_file refuses everything else). All variants by
    default: permuted ones are label-free (a within-reference key permutation needs only keys) and are the matched null.
    Layout: confirm/<family>/<set>.parquet (real), confirm/<family>/<variant>/<set>.parquet (permuted)."""
    refuse_if_frozen(out)
    width = total_width(extras)
    variants = variants or list(VARIANTS)  # matched null: every variant has its own label-free confirmatory table
    path = out / "wide" / "confirm" / "receipt.json"
    record = json.loads(path.read_text()) if path.is_file() else {
        "schema": CONFIRM_SCHEMA, "label": "features only; no label read or written", "width": width,
        "feature_set_id": PROFILE.feature_set_id, "sets": {}}
    if record["schema"] != CONFIRM_SCHEMA or record["width"] != width:
        raise ValueError(f"{path}: existing confirm receipt has another schema or width; build into a clean --out")
    record["table_code"] = code_identity()
    for name in sets:
        b = load_bank_set(bank, name, extras, peer_dir)
        keep = ~b.keys.pixels_identical.to_numpy().astype(bool)
        keys = b.keys.loc[keep].reset_index(drop=True)
        meta = pd.DataFrame({"pair_key": keys.pair_key, "row_id": keys.row_id, "ref_basename": keys.ref_group,
                             "member_set": name})
        rec = record["sets"].setdefault(name, {"tables": {}})
        rec.update({"bank": b.receipt, "rows": int(keep.sum()), "identical_rows_dropped": int((~keep).sum()),
                    "dropped_pair_keys_sha256": hashlib.sha256("\n".join(b.keys.pair_key[~keep]).encode()).hexdigest()})
        leg = Leg(name, meta, b.X[keep], None if b.peers is None else b.peers[keep], [0.0, 1.0], np.zeros(keep.sum()),
                  {n: np.zeros(keep.sum(), np.float32) for n in AUX_ORACLE})
        for family in families:
            if family == "aux" and b.peers is None:
                rec["tables"].setdefault(family, {})["skipped"] = "peer columns missing from the REEXTRACT peer output"
                continue
            x, added = family_matrix(leg, family, width)  # aux oracle columns stay zero: an oracle needs a label
            for variant in variants:
                xv = x
                if variant != "real":
                    salt = 0 if family == "main" else 500
                    seed = PERM_SEED_BASE + salt + int(variant[1:]) + 1000 * (len(LEG_ORDER) + CONFIRM_SETS.index(name))
                    xv = x.copy()
                    xv[:, added] = permute_within_reference(meta.ref_basename.to_numpy(), meta.pair_key.to_numpy(),
                                                            x[:, added], np.ones(len(x), dtype=bool),
                                                            np.random.default_rng(seed))
                dest = out / "wide" / "confirm" / family / ("" if variant == "real" else variant)
                table = write_table(out, dest / f"{name}.parquet", meta.ref_basename.to_numpy(), np.zeros(len(meta)), xv,
                                    family, f"confirmatory set {name} ({variant}, features only)", width)
                table["keys_sha256"] = write_keys(dest / f"{name}.keys.parquet", meta,
                                                  ["pair_key", "row_id", "ref_basename", "member_set"])
                rec["tables"].setdefault(family, {}).pop("skipped", None)
                rec["tables"][family][variant] = table
        print(json.dumps({"confirm": name, "rows": rec["rows"], "variants": variants}), flush=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"receipt": str(path), "sha256": sha(path)}), flush=True)
    return record


# ------------------------------------------------------------------ keep lists
def refuse_if_frozen(out: Path) -> None:
    if (Path(out) / "wide" / "frozen.json").is_file():
        raise ValueError(f"{out}: frozen (wide/frozen.json); a changed table needs a new root or an explicit unfreeze by the coordinator")


def freeze(out: Path) -> str:
    """Freeze the canon root: record the hash of every receipt the cells and the read must see, after a clean `verify`."""
    import v2_common
    wide = Path(out) / "wide"
    refuse_if_frozen(out)
    verify = json.loads((wide / "verify.json").read_text())
    if not verify.get("all_ok"):
        raise ValueError("verify.json is not all_ok; fix and re-verify before the freeze")
    wide_receipts, widths, ids = {}, set(), set()
    families = ("main",) if PROFILE.revision == 5 else FAMILIES
    variants = ("real",) if PROFILE.revision == 5 else VARIANTS
    for family in families:
        for variant in variants:
            rp = wide / family / variant / "receipt.json"
            if not rp.is_file():
                raise ValueError(f"{family}/{variant}: receipt missing (build every family and variant before the freeze)")
            rec = json.loads(rp.read_text())
            if int(rec.get("formula_revision", 4)) != PROFILE.revision:
                raise ValueError(f"{family}/{variant}: formula revision disagrees with freeze profile")
            if not rec.get("complete"):
                raise ValueError(f"{family}/{variant}: receipt incomplete")
            wide_receipts[f"{family}/{variant}"] = sha(rp)
            widths.add(rec["width"]), ids.add(rec["feature_set_id"])
    confirm = json.loads((wide / "confirm" / "receipt.json").read_text())
    for name in CONFIRM_SETS:
        tables = confirm["sets"][name]["tables"]
        for family in families:
            for variant in variants:
                if variant not in tables.get(family, {}):
                    raise ValueError(f"confirm {name}/{family}/{variant}: table missing")
    if (len(widths) != 1 or len(ids) != 1 or confirm["width"] not in widths
            or confirm["feature_set_id"] not in ids):
        raise ValueError("widths or feature_set_ids disagree across receipts")
    extra = wide / "extra_arms.json"
    record = {"schema": v2_common.FROZEN_SCHEMA, "width": widths.pop(), "feature_set_id": ids.pop(), "wide_receipts": wide_receipts,
              "confirm_receipt_sha256": sha(wide / "confirm" / "receipt.json"), "keep_lists_sha256": sha(wide / "keep_lists.json"),
              "extra_arms_sha256": sha(extra) if extra.is_file() else None, "verify_sha256": sha(wide / "verify.json"),
              "frozen_at": __import__("time").strftime("%Y-%m-%d %H:%M %Z"), "table_code": code_identity()}
    path = wide / "frozen.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"frozen": str(path), "sha256": sha(path), "width": record["width"]}))
    return sha(path)


def write_keeplists(out: Path, extra_arms: dict[str, list[int]], width: int) -> None:
    import v2_common
    refuse_if_frozen(out)
    if extra_arms:
        path = out / "wide" / "extra_arms.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": "rev4-featpot-v2c-extra-arms-v1", "width": width, "arms": extra_arms}) + "\n")
    if Path(v2_common.V2).resolve() != Path(out).resolve():
        raise ValueError("keeplists: pass --root <out> so spec parsing reads this root's extra arms")
    lists = {spec: dict(zip(("family", "variant", "keep"), v2_common.arm_columns(spec))) for spec in v2_common.all_specs()}
    path = out / "wide" / "keep_lists.json"
    path.write_text(json.dumps({"schema": "rev4-featpot-v2-keeplists-v2", "specs": lists}) + "\n")
    print(json.dumps({"keep_lists": str(path), "sha256": sha(path), "specs": len(lists)}))


def admitted_bank(bank: Path, set_name: str, receipt: dict) -> dict:
    manifest_path = bank_file(bank, set_name, "_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    check_manifest(manifest, set_name)
    recorded = receipt["bank"][set_name]
    if (sha(manifest_path) != recorded["manifest_sha256"]
            or sha(bank_file(bank, set_name, "features.parquet")) != recorded["features_sha256"]
            or sha(bank_file(bank, set_name, "keys.parquet")) != recorded["keys_sha256"]
            or manifest["input_contract"] != "legacy-rgb8"):
        raise ValueError(f"{set_name}: bank receipt/input contract changed")
    # Every chunk must be bound to the same executable, arithmetic and IDs.
    for chunk in manifest["chunks"]:
        producer = chunk["extractor_manifest"]
        if (producer["producer_binary_sha256"] != PROFILE.binary
                or producer["feature_set_id"] != PROFILE.feature_set_id
                or int(producer["formula_revision"]) != 5
                or producer["populated_feature_ids"] != [i for lo, hi in PROFILE.requested for i in range(lo, hi)]):
            raise ValueError(f"{set_name}: chunk producer binding changed")
    return recorded


def admit_teachers(bank: Path, source: Path, out: Path) -> None:
    """Fresh metadata-only admission view of frozen Rev5 TRAIN teacher legs.

    The decoder era names the actual producing executable and input contract,
    not guessed dependency commits. This is table provenance, not split/model
    qualification. Human, confirmatory and permuted tables are never admitted
    by this bounded route; the historical instrument remains immutable.
    """
    if PROFILE.revision != 5 or PROFILE.era != "rev5_localwin":
        raise ValueError("admit-teachers requires the frozen Rev5 local-window profile")
    source, out, bank = safe_path(source), safe_path(out), safe_path(bank)
    from v2_common import refuse_immutable_output
    refuse_immutable_output(out, (source, bank))
    if out.exists():
        raise ValueError(f"{out}: admission output must be fresh")
    wide = source / "wide" / "main" / "real"
    receipt_path = wide / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    frozen = json.loads((source / "wide" / "frozen.json").read_text())
    if (frozen["wide_receipts"].get("main/real") != sha(receipt_path)
            or receipt["schema"] != SCHEMA or receipt["family"] != "main" or receipt["variant"] != "real"
            or receipt["formula_revision"] != 5 or receipt["feature_set_id"] != PROFILE.feature_set_id
            or receipt["era"] != PROFILE.era or receipt["extras"]):
        raise ValueError("source is not the frozen real Rev5 main receipt")
    prepared = []
    for teacher, (set_name, _, _) in TEACHERS.items():
        recorded = admitted_bank(bank, set_name, receipt)
        for split in ("fit", "dev"):
            record = receipt["legs"][teacher][split]
            path = safe_path(source / record["rel"])
            expected = wide / f"{teacher}_{split}.parquet"
            if path.resolve() != expected.resolve():
                raise ValueError("teacher receipt points outside the permitted real TRAIN table")
            sidecar = Path(f"{path}.manifest.json")
            stored = json.loads(sidecar.read_text())
            if (sha(path) != record["sha256"] or sha(sidecar) != record["manifest_sha256"]
                    or stored["source_bank_feature_set_id"] != PROFILE.feature_set_id
                    or stored["formula_revision"] != 5):
                raise ValueError(f"{path}: table/sidecar receipt changed")
            declarations = {**stored, "feature_set_id": PROFILE.feature_set_id,
                            "decoder_era": f"legacy-rgb8/extract_features_372col@sha256:{PROFILE.binary}",
                            "admission_scope": "TRAIN teachers only; model qualification remains separate",
                            "source_table_sha256": record["sha256"],
                            "source_manifest_sha256": record["manifest_sha256"],
                            "source_receipt_sha256": sha(receipt_path),
                            "bank_manifest_sha256": recorded["manifest_sha256"],
                            "extractor_build_commit": PROFILE.build,
                            "decoder_binding": "actual producing executable; individual decoder commits not inferred"}
            prepared.append((path, record, declarations))
    out.mkdir(parents=True)
    files = {}
    for path, record, declarations in prepared:
        dest = out / path.name
        shutil.copyfile(path, dest)
        if sha(dest) != record["sha256"]:
            raise ValueError(f"{dest}: byte-preserving copy failed")
        sidecar = Path(f"{dest}.manifest.json")
        sidecar.write_text(json.dumps(declarations, indent=1) + "\n")
        files[dest.name] = {"source": str(path), "sha256": record["sha256"],
                            "manifest_sha256": sha(sidecar), "rows": record["rows"],
                            "references": record["references"]}
    (out / "admission_receipt.json").write_text(json.dumps({
        "schema": "rev5-teacher-admission-v1", "files": files,
        "feature_values_changed": False, "table_code": code_identity()}, indent=1) + "\n")
    print(json.dumps({"admission_view": str(out), "files": len(files), "feature_values_changed": False}))


def admit_recipe(bank: Path, source: Path, out: Path) -> None:
    """Fresh full-recipe view; human scientific role stays explicitly pending.

    Only frozen main/real teacher, design-human and their existing fit/dev
    tables are copied. Row keys exclude target columns. No human targets or
    feature cells are decoded; original Parquet bytes and ordering survive.
    """
    import copy
    from v2_common import admission_input_roots, human_dev, refuse_immutable_output
    from v2_teacher import row_keys_sha, selection_sha
    from e15_coverage import admit_pool
    bank, source, out = safe_path(bank), safe_path(source), safe_path(out)
    refuse_immutable_output(out, (source, bank))
    if PROFILE.revision != 5 or PROFILE.era != "rev5_localwin" or out.exists():
        raise ValueError("admit-recipe needs the Rev5 profile and a fresh output")
    wide = source / "wide/main/real"
    rp = wide / "receipt.json"
    receipt = json.loads(rp.read_text())
    frozen = json.loads((source / "wide/frozen.json").read_text())
    input_roots = (source.resolve(), bank.resolve())
    if frozen.get("schema") == "rev5-recipe-admission-freeze-v1":
        input_roots += admission_input_roots(source)
        refuse_immutable_output(out, input_roots)
    if (frozen["wide_receipts"].get("main/real") != sha(rp) or receipt["schema"] != SCHEMA
            or receipt["family"] != "main" or receipt["variant"] != "real" or not receipt["complete"]
            or receipt["formula_revision"] != 5 or receipt["feature_set_id"] != PROFILE.feature_set_id
            or receipt["era"] != PROFILE.era or receipt["extras"]):
        raise ValueError("source is not the complete frozen real Rev5 recipe")
    banks = {name: admitted_bank(bank, name, receipt) for name in
             (*dict.fromkeys(n for source_name in SOURCE_ORDER for n in SOURCES[source_name]),
              *(v[0] for v in TEACHERS.values()))}
    frames, source_keys = {}, {}
    for name in (*SOURCE_ORDER, *TEACHERS):
        splits = ("full",) if name in SOURCES else ("fit", "dev")
        for split in splits:
            record = receipt["legs"][name][split]
            kp = wide / (f"{name}.keys.parquet" if split == "full" else f"{name}_{split}.keys.parquet")
            want = receipt["legs"][name]["keys_sha256"] if split == "full" else record["keys_sha256"]
            if sha(kp) != want:
                raise ValueError(f"{kp}: frozen row keys changed")
            # Target column is deliberately not read, including human keys.
            frames[name, split] = pq.read_table(kp, columns=META[:-1]).to_pandas()
            source_keys[name, split] = want
    if sha(source / "wide/keep_lists.json") != frozen["keep_lists_sha256"]:
        raise ValueError("source keep lists changed after freeze")
    prepared = []
    for name, leg in receipt["legs"].items():
        if name not in (*SOURCE_ORDER, *TEACHERS, "human_all", *(f"human_without_{h}" for h in SOURCE_ORDER)):
            raise ValueError(f"{name}: not a registered recipe leg")
        splits = ("full",) if name in SOURCES else ("fit", "dev")
        for split in splits:
            rec = leg[split]
            path = safe_path(source / rec["rel"])
            expected = wide / (f"{name}.parquet" if split == "full" else f"{name}_{split}.parquet")
            if path.resolve() != expected.resolve():
                raise ValueError("recipe receipt points outside registered real tables")
            sp = Path(f"{path}.manifest.json")
            stored = json.loads(sp.read_text())
            if (sha(path) != rec["sha256"] or sha(sp) != rec["manifest_sha256"]
                    or stored["source_bank_feature_set_id"] != PROFILE.feature_set_id
                    or stored["formula_revision"] != 5):
                raise ValueError(f"{path}: recipe table/sidecar changed")
            human = name not in TEACHERS
            members = ([name] if name in SOURCES or name in TEACHERS else
                       [h for h in SOURCE_ORDER if name != f"human_without_{h}"])
            parts, selections = [], []
            for member in members:
                source_split = split if member in TEACHERS else "full"
                frame = frames[member, source_split]
                mask = (np.array([human_dev(r) for r in frame.ref_basename]) == (split == "dev")
                        if name.startswith("human_") else np.ones(len(frame), dtype=bool))
                parts.append(frame.loc[mask])
                selections.append({"source": member, "split": source_split,
                                   "source_keys_sha256": source_keys[member, source_split],
                                   "row_indices_sha256": selection_sha(np.flatnonzero(mask)), "rows": int(mask.sum())})
            keys = pa.Table.from_pandas(pd.concat(parts, ignore_index=True), preserve_index=False)
            refs = pq.read_table(path, columns=["ref_basename"])["ref_basename"].to_pylist()
            if refs != keys["ref_basename"].to_pylist() or len(refs) != rec["rows"]:
                raise ValueError(f"{name}/{split}: reconstructed key selection/order differs")
            declarations = {**stored, "feature_set_id": PROFILE.feature_set_id,
                "decoder_era": f"legacy-rgb8/extract_features_372col@sha256:{PROFILE.binary}",
                "extractor_build_commit": PROFILE.build, "table_sha256": rec["sha256"],
                "source_table_sha256": rec["sha256"], "source_manifest_sha256": rec["manifest_sha256"],
                "source_receipt_sha256": sha(rp), "bank_manifest_sha256":
                    {m: banks[m]["manifest_sha256"] for member in members for m in
                     (SOURCES[member] if member in SOURCES else (TEACHERS[member][0],))},
                "row_keys_sha256": row_keys_sha(keys), "row_selection": selections,
                "row_selection_sha256": hashlib.sha256(json.dumps(selections, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "data_role": "design-released-human" if human else "TRAIN oracle teacher",
                "data_role_decision_required": "SHIPPATH-human-production-role" if human else None,
                "human_sources": members if human else [], "feature_values_changed": False,
                "admission_root": str(out.resolve()),
                "immutable_input_roots": [str(p) for p in (out.resolve(), *input_roots)]}
            prepared.append((name, split, path, rec, keys, declarations))
    # Coverage validates before copying its frozen pool. Failure never edits the source.
    admit_pool(source / "e15", out / "e15", immutable_roots=input_roots, admission_root=out)
    dest_wide = out / "wide/main/real"
    dest_wide.mkdir(parents=True)
    new_receipt = copy.deepcopy(receipt)
    for name, split, path, rec, keys, declarations in prepared:
        dest = dest_wide / path.name
        shutil.copyfile(path, dest)
        if sha(dest) != rec["sha256"]:
            raise ValueError("recipe byte-preserving copy failed")
        kp = dest.with_suffix(".keys.parquet")
        pq.write_table(keys, kp, compression="zstd")
        declarations["keys_sha256"] = sha(kp)
        sp = Path(f"{dest}.manifest.json")
        sp.write_text(json.dumps(declarations, indent=1) + "\n")
        updated = {**rec, "manifest_sha256": sha(sp), "keys_sha256": sha(kp)}
        new_receipt["legs"][name][split] = updated
        if split == "full":
            new_receipt["legs"][name]["keys_sha256"] = sha(kp)
    new_receipt["admission_view"] = {"schema": "rev5-recipe-admission-v1", "source_root": str(source.resolve()),
        "bank_root": str(bank.resolve()),
        "immutable_input_roots": [str(p) for p in input_roots],
        "source_receipt_sha256": sha(rp), "feature_values_changed": False,
        "human_role_decision": "PENDING: SHIPPATH-human-production-role", "table_code": code_identity()}
    dest_rp = dest_wide / "receipt.json"
    dest_rp.write_text(json.dumps(new_receipt, indent=1) + "\n")
    shutil.copyfile(source / "wide/keep_lists.json", out / "wide/keep_lists.json")
    (out / "wide/frozen.json").write_text(json.dumps({"schema": "rev5-recipe-admission-freeze-v1",
        "wide_receipts": {"main/real": sha(dest_rp)}, "keep_lists_sha256": sha(out / "wide/keep_lists.json"),
        "auxiliary_files": {str(p.relative_to(out)): sha(p) for p in (out / "e15").iterdir() if p.is_file()},
        "admission_view": new_receipt["admission_view"]}, indent=1) + "\n")
    print(json.dumps({"recipe_admission": str(out), "tables": len(prepared), "feature_values_changed": False,
                      "human_role_decision": "PENDING"}))


def build_assessment(bank: Path, out: Path, names: list[str], ids: list[int], derived_inputs=None) -> dict:
    """Declared Rev5 features-only views; preserve every key including identities."""
    from v2_common import refuse_immutable_output
    from rev5_bank import ASSESSMENT_KEY_COLUMNS
    registered_ids = json.loads((REPO / "benchmarks/costset2_2026-10-03.candidate_ids.json").read_text())["candidates"]["by_v2fy"]
    binding = None
    if derived_inputs is None:
        if PROFILE.revision not in (5, 7) or ids != registered_ids:
            raise ValueError("Rev5 (or by-construction revision 7) by_v2fy420 assessment required")
    elif derived_inputs == "fx2":
        from v2_common import derived_declaration
        path, pin, declaration = derived_declaration("fx2")
        expected = sorted(set(declaration["direct"]) | {i for pair in declaration["products"] for i in pair})
        if PROFILE.revision != 7 or ids != expected:
            raise ValueError("revision-7 assessment requires every pinned fx2 consumed feature, in order")
        binding = dict(token="fx2", path=str(path), sha256=pin)
    else:
        raise ValueError("unregistered assessment derived-input token")
    bank = assessment_path(bank)
    refuse_immutable_output(out, [bank])
    if assessment_path(out).exists():
        raise ValueError("fresh assessment output required")
    prepared = []
    for name in names:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            raise ValueError("assessment set token required")
        base = assessment_path(bank / name)
        paths = {f: assessment_path(base / f) for f in BANK_FILES}
        m = json.loads(paths["_MANIFEST.json"].read_text())
        check_manifest(m, name)
        refuse_immutable_output(out, [bank] + [Path(v) for v in m.get("assessment", {}).get("immutable_roots", [])])
        schema = pq.ParquetFile(paths["keys.parquet"]).schema_arrow
        if any(n in schema.names for n in ["human_score", "target", "mos", "ssim2_gpu"]):
            raise ValueError("label-bearing keys forbidden")
        feature_schema = pq.ParquetFile(paths["features.parquet"]).schema_arrow.names
        if any(n not in ("pair_key", "row_id") and not (n.startswith("f") and n[1:].isdigit()) for n in feature_schema):
            raise ValueError("label-bearing feature bank forbidden")
        if sha(paths["keys.parquet"]) != m["keys_sha256"] or sha(paths["features.parquet"]) != m["features_parquet_sha256"]:
            raise ValueError("changed assessment bank")
        keys = pq.read_table(paths["keys.parquet"], columns=[n for n in schema.names if n in ASSESSMENT_KEY_COLUMNS or n in [
            "width", "height", "reference_file_sha256", "distorted_file_sha256", "reference_pixels_sha256", "distorted_pixels_sha256"]])
        X, feature_keys = read_matrix(paths["features.parquet"], ids, m["rows"])
        if not np.isfinite(X).all() or keys.num_rows != m["rows"] or keys.column("pair_key").to_pylist() != feature_keys.tolist():
            raise ValueError("assessment features/key parity")
        key_order = keys.column("pair_key").to_pylist()
        if len(set(key_order)) != len(key_order) or keys.column("row_id").to_pylist() != list(range(keys.num_rows)):
            raise ValueError("assessment row selection/order must be exact")
        prepared.append((name, m, keys, X, paths))
    out.mkdir(parents=True)
    record = {"schema":"rev5-assessment-tables-v1", "features_only":True,"labels_read":False,
        "feature_ids":ids,"feature_set_id":"basic+v2@w720/"+PROFILE.era+"#62adfc93",
        "formula_revision":PROFILE.revision,"decoder_era":f"legacy-rgb8/extract_features_372col@sha256:{PROFILE.binary}",
        "table_code":code_identity(),"bank_root":str(bank.resolve()),"tables":[]}
    if binding is not None:
        record["derived_inputs"] = binding
    for name, m, keys, X, paths in prepared:
        kp = out / (name + ".keys.parquet");pq.write_table(keys, kp, compression="zstd")
        columns = {k:keys.column(k) for k in keys.column_names}
        columns["ref_basename"] = keys.column("ref_group")
        positions = {v:i for i,v in enumerate(ids)}
        for i in range(720):
            columns[f"f{i}"] = pa.array(X[:,positions[i]] if i in positions else np.full(len(X),np.nan,dtype=np.float32))
        table = out / (name + ".parquet");pq.write_table(pa.table(columns), table, compression="zstd")
        ordered = hashlib.sha256(json.dumps(keys.to_pydict(),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        declaration = {k:record[k] for k in ["feature_set_id","formula_revision","decoder_era","features_only","labels_read"]}
        if binding is not None:
            declaration["derived_inputs"] = binding
        declaration.update(source_bank_feature_set_id=m["feature_set_id"], extractor_build_commit=m["build_commit"],
            extractor_binary_sha256=m["binary_sha256"], source_root=str(bank.resolve()),
            source_files={f:{"path":str(p),"sha256":sha(p)} for f,p in paths.items()},
            rows=keys.num_rows,feature_ids=ids,table_sha256=sha(table),keys_sha256=sha(kp),
            ordered_keys_sha256=ordered,consumed_features_f32_le_sha256=hashlib.sha256(np.asarray(X,dtype="<f4").tobytes()).hexdigest(),
            absent_slots="NaN",assessment=m.get("assessment",{"features_only":True,"scope":"original confirmation bank; future protected label exposure requires separate freeze"}))
        sidecar=Path(str(table)+".manifest.json");sidecar.write_text(json.dumps(declaration,indent=1)+"\n")
        record["tables"].append({"set":name,"path":str(table.resolve()),"sha256":sha(table),"rows":keys.num_rows,
            "declaration":{"path":str(sidecar.resolve()),"sha256":sha(sidecar)},
            "keys":{"path":str(kp.resolve()),"sha256":sha(kp)},"ordered_keys_sha256":ordered})
    (out/"ASSESSMENT.json").write_text(json.dumps(record,indent=1)+"\n")
    return record


def attach_hdr(out: Path, bank: Path) -> None:
    """Attach one admitted E26 fit leg to a fresh SDR root; never pack VAL."""
    from rev5_bank import HDR_TEACHER_SHA, HDR_TRANSFORM
    import e21_cheap_recipe as e21

    refuse_if_frozen(out)
    rp = out / "wide/main/real/receipt.json"
    receipt = json.loads(rp.read_text())
    if receipt.get("formula_revision") != 5 or "hdr" in receipt["legs"]:
        raise ValueError("HDR attachment requires a fresh Rev5 SDR receipt")
    mpath = bank / "_MANIFEST.json"
    man = json.loads(mpath.read_text())
    ids = e21.columns("by_v2fy")
    if (man.get("role") != "train" or man.get("study") != "E26" or man.get("rows") != 7390
            or man.get("teacher_sha256") != HDR_TEACHER_SHA["train"] or man.get("formula_revision") != "Rev5"
            or man.get("requested_ids") != ids or man.get("target_transform") != HDR_TRANSFORM):
        raise ValueError("inadmissible native HDR bank")
    for name, pin in [("keys.parquet", "keys_sha256"), ("features.parquet", "features_parquet_sha256")]:
        if sha(bank / name) != man[pin]:
            raise ValueError("native HDR bank changed after extraction")
    keys = pq.read_table(bank / "keys.parquet").to_pylist()
    if len(keys) != 7390 or any(k["role"] != "train" or not k["agree"] for k in keys):
        raise ValueError("HDR keys are not TRAIN agree-only")
    features = pq.read_table(bank / "features.parquet")
    if features["row_id"].to_pylist() != [k["row_id"] for k in keys]:
        raise ValueError("HDR feature/label row join mismatch")
    width = receipt["width"]
    cols = {"ref_basename": pa.array([k["ref_path"] for k in keys]),
            "human_score": pa.array([10 * k["hdrvdp3_q_jod"] for k in keys], type=pa.float64())}
    for i in range(width):
        values = features[f"f{i}"].to_numpy() if f"f{i}" in features.column_names else np.full(len(keys), np.nan)
        if not (np.isfinite(values).all() if i in ids else np.isnan(values).all()):
            raise ValueError(f"HDR measured/absent contract f{i}")
        cols[f"f{i}"] = pa.array(values.astype(np.float32))
    path = rp.parent / "hdr_fit.parquet"
    pq.write_table(pa.table(cols), path, compression="zstd")
    kpath = path.with_suffix(".keys.parquet")
    pq.write_table(pa.Table.from_pylist(keys), kpath, compression="zstd")
    compact = {k: man[k] for k in ("study", "role", "rows", "teacher_sha256", "target_transform", "requested_ids",
                                  "binary_sha256", "build_commit", "row_order_sha256", "input_contract")}
    # Channel subsets have no reconstructible family-token FeatureSetId. Keep
    # the native owner output as provenance rather than a malformed null
    # trainer declaration; explicit requested IDs and slot provenance own it.
    compact.update(formula_revision=5, population="agree-only",
                   source_bank_feature_set_id=man["research"]["feature_set_id"], bank_manifest_sha256=sha(mpath))
    sidecar = Path(f"{path}.manifest.json")
    sidecar.write_text(json.dumps(compact, indent=1)+"\n")
    receipt["legs"]["hdr"] = {"fit": {"rel": str(path.relative_to(out)), "sha256": sha(path),
        "manifest_sha256": sha(sidecar), "rows": 7390, "keys_sha256": sha(kpath)}, "keys_sha256": sha(kpath)}
    receipt["required_legs"].append("hdr")
    rp.write_text(json.dumps(receipt, indent=1)+"\n")
    print(json.dumps({"hdr_attached": str(path), "receipt_sha256": sha(rp)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "confirm", "keeplists", "verify", "freeze", "admit-teachers", "admit-recipe", "assessment", "hdr-leg"])
    ap.add_argument("--hdr-bank", type=Path)
    ap.add_argument("--assessment-set", action="append", default=[])
    ap.add_argument("--assessment-ids", type=Path)
    ap.add_argument("--assessment-derived", choices=["fx2"], help="explicit registered revision-7 fx2 assessment")
    ap.add_argument("--source-root", type=Path, help="admit-teachers: immutable frozen Rev5 instrument root")
    ap.add_argument("--bank", type=Path, default=CANON_BANK)
    ap.add_argument("--out", "--root", dest="out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--legs", default="all", help="comma list of: human, safesyn, cid22, teachers, all")
    ap.add_argument("--family", choices=FAMILIES, action="append")
    ap.add_argument("--variant", choices=VARIANTS, action="append")
    ap.add_argument("--extra", action="append", default=[], help="FAM=FIRST:WIDTH:TEMPLATE (TEMPLATE has {set})")
    ap.add_argument("--extra-arm", action="append", default=[], help="NAME=ID,ID,... (keeplists)")
    ap.add_argument("--peer-dir", type=Path, default=CANON_PEERS)
    ap.add_argument("--sample", type=int, default=10_000, help="verify: SafeSyn sample rows")
    ap.add_argument("--revision", type=int, choices=[4, 5, 7], default=4,
                    help="bank profile (Rev5: rev5_bank.py output; 7: its revision-7 output, assessment only)")
    ap.add_argument("--expect-era", default="")
    ap.add_argument("--expect-fsid", default="")
    ap.add_argument("--expect-binary", default="")
    ap.add_argument("--expect-build", default="")
    ap.add_argument("--pad-to", type=int, default=0, help="Rev5: NaN-pad tables to this width (the Rev4 tables' width)")
    args = ap.parse_args()
    global PROFILE
    if args.revision in (5, 7):
        if args.bank == CANON_BANK:
            ap.error(f"--revision {args.revision} needs an explicit --bank (the rev5_bank.py bank)")
        if args.revision == 7 and args.action != "assessment":
            ap.error("--revision 7 builds assessment tables only")
        PROFILE = rev5_profile(args.expect_era, args.expect_fsid, args.expect_binary, args.expect_build, args.pad_to or None,
                               args.revision)
    import v2_common
    v2_common.V2 = args.out  # receipts hold root-relative table paths (table_path); the root is --out
    extras = parse_extras(args.extra)
    fams = args.family or list(FAMILIES)
    if args.action == "assessment":
        if not args.assessment_set or args.assessment_ids is None:
            ap.error("assessment needs explicit sets and IDs")
        ids = json.loads(safe_path(args.assessment_ids).read_text())
        print(json.dumps(build_assessment(args.bank,args.out,args.assessment_set,ids,args.assessment_derived)["tables"]))
    elif args.action in ("admit-teachers", "admit-recipe"):
        if args.source_root is None:
            ap.error("admit-teachers requires --source-root")
        (admit_recipe if args.action == "admit-recipe" else admit_teachers)(args.bank, args.source_root, args.out)
    elif args.action == "hdr-leg":
        attach_hdr(args.out, args.hdr_bank)
    elif args.action == "build":
        build(args.bank, args.out, args.legs, fams, args.variant or list(VARIANTS), extras, args.peer_dir)
    elif args.action == "confirm":
        build_confirm(args.bank, args.out, fams, extras, args.peer_dir, args.variant)
    elif args.action == "keeplists":
        arms = {}
        for spec in args.extra_arm:
            name, _, ids = spec.partition("=")
            arms[name] = [int(i) for i in ids.split(",")]
        write_keeplists(args.out, arms, total_width(extras))
    elif args.action == "freeze":
        freeze(args.out)
    else:
        from v2c_verify import verify
        sys.exit(verify(args.bank, args.out, extras, args.sample))


if __name__ == "__main__":
    # The verifier imports this owner for its profile-aware width and bank
    # checks. Keep the CLI's selected profile rather than creating a second
    # module instance with the default Rev4 profile.
    sys.modules["v2c_wide"] = sys.modules[__name__]
    main()
