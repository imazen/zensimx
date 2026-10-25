"""Derive a training root whose columns come from a measurement arm (by-construction exploration).

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Ringing corrected: high-pass form, trained and tested";
"Coarse scales in linear light"). The source root (the v40r4 `v2e29` root every E21 arm trained on) stays untouched.
The new root holds the same tables, rows, keys, labels and split decisions; only the arm's columns are replaced with
values from a bank extracted under it. Every other measured column of the new bank must equal the stored table bit for
bit (float64 bank values cast to float32, as `v2c_wide` writes them); that check is what proves each row joined to the
right pixels and that the arm moved nothing else. Arms (`--arm`, default hpm):

  hpm     `ZENSIM_RING_FORM=hpm`: the 12 ringing columns (v2 `idx::RINGING` over the 12 scale x channel cells)
  linpyr  `ZENSIM_PYRAMID=linear`: every column of scales 1-3, and scale 0's edge-width change (a scale 0/1 ratio)
  rev7    `ZENSIM_FORMULA_REV=7` (raw ringing, owner 2026-10-25): every measured column; the tables declare formula
          revision 7 and the new extractor, and each replaced value must stay close to the stored Rev5 value

  pool   --bin EXTRACTOR --out DIR             extract the E15 coverage-pool selection (44,000 pairs) under the arm
  build  --bank DIR --pool DIR --out ROOT      write the derived root and re-pin its receipt and freeze
  globaff-pairs --bank DIR --pool DIR --out DIR   one pairs TSV per bank set (row_id = bank row) and the pool's
  globaff --root ROOT --bank DIR --sidecars DIR --out ROOT2
                                               widen a derived root with the byc_globaff descriptor at f1867-f1890
  orient --root ROOT --bank DIR --sidecars DIR --out ROOT2
                                               widen a root with the byc_orient damage (Haar H/V/D band-difference RMS
                                               per scale x channel) at f2301-f2336; sidecars from the globaff-pairs TSVs
  gate-table --table T.parquet --decl FX1G.json --out T2.parquet [--channels XYB|Y]
                                               the same gating for one width-1891 table (mixture tables, AIC folds)
  gated --root ROOT2 --decl FX1G.json --out ROOT3 [--channels XYB|Y]
                                               widen a descriptor root with gated damage from f1891: for each of fx1's
                                               differences d (ascending) in a gated channel, d x (1 - share) of its own
                                               scale and channel cell, as an f32 product of f32 factors. XYB (default)
                                               gates all 410 (f1891-f2300); Y only the 164 luminance-cell differences
                                               (f1891-f2054)
  gated-decl --decl FX1G.json --out DECL.json [--channels XYB|Y]
                                               the training declaration of a gated root: fx1's 410 differences and the
                                               gated ids direct, fx1's 410 fragility products

The HDR legs (`hdr_hb4`, `hdr_hc4`, leg `hdr_consensus`) are not re-extracted and are left out of the derived root.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SOURCE = Path("/mnt/v/output/zensim/v40r4-2026-10-08/v2e29")
POOL_SOURCE = Path("/var/tmp/rev5-featpot/v2c/e15")
RING_IDS = [372 + 29 * cell + 26 for cell in range(12)]  # v2 block f372.., 29 per (scale, channel) cell, RINGING = 26
MEASURED = [i for lo, hi in ((0, 228), (372, 720)) for i in range(lo, hi)]
ERA = "rev5_localwin_ringhpm"
HDR = ("hdr_hb4", "hdr_hc4")


def scale_of(i: int) -> int:
    """Pyramid scale of measured feature `i`: (scale, channel) cells of 13 basic, 6 peak and 29 v2 features."""
    if i < 156:
        return i // 39
    if i < 228:
        return (i - 156) // 18
    return (i - 372) // 87


EDGE_WIDTH_S0 = [372 + 29 * ch + 28 for ch in range(3)]  # v2 EDGE_WIDTH_CHANGE = 28: scale 0's two-scale ratio
ARMS = {
    "hpm": {"env": {"ZENSIM_RING_FORM": "hpm"}, "manifest": ("ring_form", "hpm"), "era": ERA, "replaced": RING_IDS,
            "record": "byc_ring_root",
            "note": f"; ringing f{RING_IDS[0]}..f{RING_IDS[-1]} (12 ids) re-extracted under ZENSIM_RING_FORM=hpm (byconstruction)"},
    "linpyr": {"env": {"ZENSIM_PYRAMID": "linear"}, "manifest": ("pyramid", "linear"), "era": "rev5_localwin_linpyr",
               "replaced": sorted([i for i in MEASURED if scale_of(i) > 0] + EDGE_WIDTH_S0), "record": "byc_linpyr_root",
               "note": "; scales 1-3 and scale-0 edge-width change re-extracted under ZENSIM_PYRAMID=linear (byconstruction)"},
    "rev7": {"env": {"ZENSIM_FORMULA_REV": "7"}, "manifest": ("formula_revision", "Rev7"), "era": "rev7_rowlanes",
             "replaced": MEASURED, "record": "byc_rev7_root", "revision": 7, "close": True,
             "note": "; every measured slot re-extracted at formula revision 7 (rowlanes, raw ringing; byconstruction)"},
}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def require_arm(manifests: list[Path], arm: dict) -> None:
    """Every research manifest of the extraction must say the arm ran (and its scale map must be `scale_of`)."""
    if not manifests:
        raise ValueError("no research manifests found")
    key, value = arm["manifest"]
    for m in manifests:
        man = json.loads(m.read_text())
        if str(man.get(key)) != str(value):
            raise ValueError(f"{m}: not extracted under {key}={value}")
        scales = {f["id"]: f["scale"] for f in man["features"]}
        if any(scales[i] != scale_of(i) for i in MEASURED):
            raise ValueError(f"{m}: feature scales disagree with scale_of")


def cmd_pool(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    tsv = out / "pairs.tsv"
    shutil.copyfile(POOL_SOURCE / "extract/pairs.tsv", tsv)
    csvp = out / "features.csv"
    arm = ARMS[a.arm]
    env = dict(os.environ, ZENSIM_FORMULA_REV="5", ZENSIM_ROOT_FORM="sqrt", RAYON_NUM_THREADS=str(a.threads))
    env.update(arm["env"])
    cmd = [a.bin, "--corpus", "pairs-tsv", "--path", str(tsv), "--out", str(csvp), "--restore-cuts", "basic,peaks,v2",
           "--input-contract", "legacy-rgb8", "--era-label", arm["era"], "--force-tier", "native"]
    t0 = time.time()
    with (out / "extract.log").open("w") as lg:
        subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, check=True)
    require_arm([Path(f"{csvp}.research_manifest.json")], arm)
    names = ["row_id"] + [f"f{i}" for i in MEASURED]
    tab = pacsv.read_csv(csvp, convert_options=pacsv.ConvertOptions(
        column_types={"row_id": pa.int64(), **{f"f{i}": pa.float64() for i in MEASURED}}, include_columns=names))
    order = np.argsort(tab.column("row_id").to_numpy(), kind="stable")
    tab = tab.take(pa.array(order))
    if tab.column("row_id").to_pylist() != list(range(tab.num_rows)):
        raise ValueError("pool row_id join failed")
    pq.write_table(tab, out / "features.parquet", compression="zstd")
    meta = {"pairs_sha256": sha(tsv), "binary_sha256": sha(a.bin), "rows": tab.num_rows, "era": arm["era"],
            "extract_wall_s": round(time.time() - t0, 1), "features_sha256": sha(out / "features.parquet"),
            "research_manifest_sha256": sha(f"{csvp}.research_manifest.json")}
    csvp.unlink()
    (out / "_MANIFEST.json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"pool: {tab.num_rows} rows in {meta['extract_wall_s']}s")
    return 0


def load_bank(bank: Path, name: str, arm: dict) -> tuple[dict, np.ndarray]:
    require_arm(sorted((bank / "_work" / name).glob("chunk_*.csv.research_manifest.json")), arm)
    cols = ["pair_key"] + [f"f{i}" for i in MEASURED]
    tab = pq.read_table(bank / name / "features.parquet", columns=cols)
    index = {k: i for i, k in enumerate(tab.column("pair_key").to_pylist())}
    if len(index) != tab.num_rows:
        raise ValueError(f"{name}: duplicate pair_key in bank")
    X = np.stack([tab.column(f"f{i}").to_numpy() for i in MEASURED], axis=1).astype(np.float32)
    return index, X


def close_check(table: pa.Table, new: np.ndarray, what: str, replaced: list[int]) -> None:
    """A re-extraction at another formula revision (rev7) moves values: by rounding almost everywhere, and by more where
    revision 7's cancellation-free forms correct the old forms' noise on near-degenerate content (flat channels: v1
    texture and variance loss, v2 HF magnitude loss, edge-width change, blockiness at the noise floor). A join or
    extractor error instead moves most of a row's columns. Refuse a nonfinite value where the stored one is finite,
    or more than 0.1% of rows with over a quarter of their columns off by more than 1e-2 relative (+1e-6); log the
    relative-change distribution and the columns that moved most. (A first version refused more than 0.1% of cells;
    on safesyn_dev 0.66% of cells moved, at most 49 of a row's 576 columns, against a mean of 546 for rows joined to
    the wrong pair: benchmarks/byconstruction_2026-10-22.md, "Revision 7 with raw ringing".)"""
    pos = {i: j for j, i in enumerate(MEASURED)}
    old = np.stack([table.column(f"f{i}").to_numpy() for i in replaced], axis=1).astype(np.float64)
    cur = new[:, [pos[i] for i in replaced]].astype(np.float64)
    fin = np.isfinite(old)
    if (fin & ~np.isfinite(cur)).any():
        raise ValueError(f"{what}: a re-extracted value is nonfinite where the stored one is finite")
    both = fin & np.isfinite(cur)
    d = np.where(both, np.abs(cur - old), 0.0)
    far = both & (d > 1e-2 * np.abs(old) + 1e-6)
    rel = d[both] / np.maximum(np.abs(old[both]), 1e-12)
    per_row = far.sum(axis=1)
    bad_rows = int((per_row > len(replaced) / 4).sum())
    top = sorted(((int(far[:, j].sum()), replaced[j]) for j in range(len(replaced))), reverse=True)[:6]
    log(f"{what}: relative change median {np.median(rel):.2e} p99 {np.quantile(rel, 0.99):.2e}; "
        f"{int(far.sum())} of {int(both.sum())} cells beyond 1e-2 relative + 1e-6, at most {int(per_row.max())} of "
        f"{len(replaced)} in a row; most: " + ", ".join(f"f{i} {c}" for c, i in top if c))
    if bad_rows > 1e-3 * len(per_row):
        raise ValueError(f"{what}: {bad_rows} rows moved in over a quarter of their columns; the join is wrong")


def patch(table: pa.Table, new: np.ndarray, what: str, replaced: list[int], close: bool = False) -> pa.Table:
    """Check every measured column outside `replaced` bit for bit (and, with `close`, every replaced value against
    `close_check`), then swap in the `replaced` columns."""
    if close:
        close_check(table, new, what, replaced)
    pos = {i: j for j, i in enumerate(MEASURED)}
    other = [i for i in MEASURED if i not in replaced]
    for i in other:
        old = table.column(f"f{i}").to_numpy()
        if not np.array_equal(old.view(np.uint32), new[:, pos[i]].view(np.uint32)):
            bad = int((old.view(np.uint32) != new[:, pos[i]].view(np.uint32)).sum())
            raise ValueError(f"{what}: f{i} differs on {bad} rows; the join or the extractor is wrong")
    moved = 0
    for i in replaced:
        j = table.column_names.index(f"f{i}")
        col = np.ascontiguousarray(new[:, pos[i]])
        moved += int((table.column(j).to_numpy().view(np.uint32) != col.view(np.uint32)).sum())
        table = table.set_column(j, f"f{i}", pa.array(col))
    log(f"{what}: {table.num_rows} rows, {len(other)} columns identical, {moved} of {table.num_rows * len(replaced)} "
        f"replaced cells changed")
    return table


def write(table: pa.Table, path: Path) -> None:
    names = [c for c in table.column_names if c.startswith("f") and c[1:].isdigit()]
    pq.write_table(table, path, compression="zstd", use_dictionary=["ref_basename"],
                   use_byte_stream_split=["human_score", *names])


def cmd_build(a) -> int:
    arm = ARMS[a.arm]
    era, replaced = arm["era"], arm["replaced"]
    bank, pool, out = Path(a.bank), Path(a.pool), Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    real = out / "wide/main/real"
    real.mkdir(parents=True)
    src_real = SOURCE / "wide/main/real"
    banks, bank_manifests = {}, {}
    fsids = set()
    key, value = arm["manifest"]
    record = {"source_root": str(SOURCE), "source_frozen_sha256": sha(SOURCE / "wide/frozen.json"), key: value,
              "replaced_ids": replaced, "bank": str(bank), "pool": str(pool), "tables": {}}
    for t in sorted(src_real.glob("*.parquet")):
        if t.name.endswith(".keys.parquet") or t.stem in HDR:
            continue
        keys = pq.read_table(src_real / f"{t.stem}.keys.parquet", columns=["pair_key", "member_set"])
        rows = []
        new = np.empty((keys.num_rows, len(MEASURED)), np.float32)
        members = keys.column("member_set").to_pylist()
        pks = keys.column("pair_key").to_pylist()
        for m in sorted(set(members)):
            if m not in banks:
                banks[m] = load_bank(bank, m, arm)
                man = json.loads((bank / m / "_MANIFEST.json").read_text())
                fsids.add(man["feature_set_id"])
                bank_manifests[m] = {"manifest_sha256": sha(bank / m / "_MANIFEST.json"),
                                     "features_sha256": man["features_parquet_sha256"], "keys_sha256": man["keys_sha256"],
                                     "rows": man["rows"], "feature_set_id": man["feature_set_id"],
                                     "binary_sha256": man["binary_sha256"], "build_commit": man["build_commit"]}
            index, X = banks[m]
            sel = [r for r, mm in enumerate(members) if mm == m]
            new[sel] = X[[index[pks[r]] for r in sel]]
        table = patch(pq.read_table(t), new, t.stem, replaced, arm.get("close", False))
        write(table, real / t.name)
        shutil.copyfile(src_real / f"{t.stem}.keys.parquet", real / f"{t.stem}.keys.parquet")
        record["tables"][t.name] = {"source_sha256": sha(t), "sha256": sha(real / t.name)}
    if len(fsids) != 1:
        raise ValueError(f"bank feature_set_id differs between sets: {fsids}")
    fsid = fsids.pop()
    if f"/{era}#" not in fsid:
        raise ValueError(f"bank was not extracted with era {era}: {fsid}")

    # Coverage pool: the stored pool is the selection minus identical and single-rung rows, in order. Match it as a
    # subsequence on its ladder fields, then let the bit-for-bit check confirm the alignment.
    import pandas as pd
    require_arm([pool / "features.csv.research_manifest.json"] if (pool / "features.csv.research_manifest.json").exists()
                else [], arm)
    sel = pd.read_parquet(POOL_SOURCE / "selection.parquet")
    pkeys = pd.read_parquet(SOURCE / "e15/coverage_pool.keys.parquet")
    fields = ["source_filename", "type", "severity_level", "severity", "sign"]
    want = list(pkeys[fields].astype(str).itertuples(index=False, name=None))
    have = list(sel[fields].astype(str).itertuples(index=False, name=None))
    take, j = [], 0
    for i, row in enumerate(have):
        if j < len(want) and row == want[j]:
            take.append(i)
            j += 1
    if j != len(want):
        raise ValueError(f"coverage pool: matched {j} of {len(want)} rows")
    ptab = pq.read_table(pool / "features.parquet")
    X = np.stack([ptab.column(f"f{i}").to_numpy() for i in MEASURED], axis=1)[take].astype(np.float32)
    table = patch(pq.read_table(SOURCE / "e15/coverage_pool.parquet"), X, "coverage_pool", replaced,
                  arm.get("close", False))
    (out / "e15").mkdir()
    write(table, out / "e15/coverage_pool.parquet")
    shutil.copyfile(SOURCE / "e15/coverage_pool.keys.parquet", out / "e15/coverage_pool.keys.parquet")
    record["tables"]["e15/coverage_pool.parquet"] = {"source_sha256": sha(SOURCE / "e15/coverage_pool.parquet"),
                                                     "sha256": sha(out / "e15/coverage_pool.parquet")}
    record["pool_manifest_sha256"] = sha(pool / "_MANIFEST.json")

    # Declarations: same rows, roles and decisions; new table hashes, feature identity and the derivation.
    note = arm["note"]
    bank_root = str(bank.resolve())
    new_roots = [str(SOURCE.resolve()), bank_root, str(pool.resolve()), str(POOL_SOURCE.resolve())]

    def declare(src_manifest: Path, table: Path) -> None:
        m = json.loads(src_manifest.read_text())
        m["byc_derived_from"] = {"table_sha256": m.get("table_sha256"), "manifest_sha256": sha(src_manifest)}
        m["table_sha256"] = sha(table)
        if "source_table_sha256" in m:
            m["source_table_sha256"] = m["table_sha256"]
        m["feature_set_id"] = fsid
        m["source_bank_feature_set_id"] = fsid
        m["feature_values_changed"] = True
        m[key] = value
        if "revision" in arm:
            # The tables now hold another revision's values: declare it, and the extractor that produced them.
            binaries = {b["binary_sha256"] for b in bank_manifests.values()}
            commits = {b["build_commit"] for b in bank_manifests.values()}
            if len(binaries) != 1 or len(commits) != 1:
                raise ValueError(f"bank sets were extracted by different binaries: {binaries} {commits}")
            m["formula_revision"] = arm["revision"]
            m["decoder_era"] = f"legacy-rgb8/extract_features_372col@sha256:{next(iter(binaries))}"
            m["extractor_build_commit"] = next(iter(commits))
        m["composite"] = m.get("composite", "") + note
        m["immutable_input_roots"] = [*m.get("immutable_input_roots", []), *new_roots]
        Path(f"{table}.manifest.json").write_text(json.dumps(m, indent=1) + "\n")

    for name in record["tables"]:
        if name.startswith("e15/"):
            declare(SOURCE / f"{name}.manifest.json", out / name)
        else:
            declare(src_real / f"{name}.manifest.json", real / name)

    for rel in ("human_role_decision.json", "source_bindings/frozen.json", "source_bindings/receipt.json",
                "wide/keep_lists.json"):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / rel, out / rel)

    receipt = json.loads((src_real / "receipt.json").read_text())
    receipt["legs"].pop("hdr_consensus", None)
    for leg in receipt["legs"].values():
        for part in leg.values():
            if isinstance(part, dict) and "rel" in part:
                p = out / part["rel"]
                part["sha256"] = sha(p)
                part["manifest_sha256"] = sha(Path(f"{p}.manifest.json"))
    receipt["byc_source_bank"] = receipt["bank"]
    receipt["bank"] = bank_manifests
    receipt["feature_set_id"] = fsid
    receipt["era"] = era
    receipt[key] = value
    if "revision" in arm:
        receipt["formula_revision"] = arm["revision"]
    view = dict(receipt["admission_view"])
    view["bank_root"] = bank_root
    view["immutable_input_roots"] = [*view["immutable_input_roots"], *new_roots]
    receipt["admission_view"] = view
    (real / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")

    frozen = json.loads((SOURCE / "wide/frozen.json").read_text())
    frozen["admission_view"] = view
    frozen["wide_receipts"] = {"main/real": sha(real / "receipt.json")}
    aux = {}
    for rel in frozen["auxiliary_files"]:
        if any(h in rel for h in HDR):
            continue
        aux[rel] = sha(out / rel)
    frozen["auxiliary_files"] = aux
    frozen["keep_lists_sha256"] = sha(out / "wide/keep_lists.json")
    frozen[arm["record"]] = record
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    log(f"root written: {out} (fsid {fsid})")
    return 0


GLOBAFF_IDS = list(range(1867, 1891))
GLOBAFF_WIDTH = GLOBAFF_IDS[-1] + 1
# `byc_fsid basic+peaks+v2 1891 rev5_localwin_ringhpm_globaff 0-227,372-719,1867-1890` (hash from zensim::feature_set_id).
GLOBAFF_FSID = "basic+peaks+v2@w1891/rev5_localwin_ringhpm_globaff#d56ad477"


def cmd_globaff_pairs(a) -> int:
    """Pairs TSVs for byc_globaff: each bank set's rows in bank order, and the coverage-pool selection."""
    bank, out = Path(a.bank), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for d in sorted(p for p in (bank / "_work").iterdir() if p.is_dir()):
        chunks = sorted(d.glob("pairs_*.tsv"))
        lines = [chunks[0].read_text().splitlines()[0]]
        for c in chunks:
            lines += c.read_text().splitlines()[1:]
        (out / f"{d.name}.tsv").write_text("\n".join(lines) + "\n")
        log(f"{d.name}: {len(lines) - 1} pairs")
    shutil.copyfile(Path(a.pool) / "pairs.tsv", out / "pool.tsv")
    return 0


# Research columns a widened root appends (`widen_root`). The declarations list them under `byc_globaff_ids`, the key
# strict admission (Python and the Rust trainer) reads for research ids on a `byc_derived_from` root, whatever family.
WIDEN = {
    "globaff": {"ids": GLOBAFF_IDS, "prefix": "g", "fsid": GLOBAFF_FSID, "finite": slice(0, None, 2),
                "note": "; byc_globaff global affine-change descriptor at f1867-f1890 (share, explained per scale x channel)"},
    # `byc_fsid basic+peaks+v2 2337 rev5_localwin_orient 0-227,372-719,2301-2336` (hash from zensim::feature_set_id).
    "orient": {"ids": list(range(2301, 2337)), "prefix": "o",
               "fsid": "basic+peaks+v2@w2337/rev5_localwin_orient#24a93918", "finite": slice(None),
               "note": "; byc_orient Haar H/V/D band-difference RMS per scale x channel at f2301-f2336"},
}


def read_sidecar(path: Path, prefix: str = "g", n: int = len(GLOBAFF_IDS)) -> np.ndarray:
    tab = pacsv.read_csv(path)
    rid = tab.column("row_id").to_numpy()
    if not np.array_equal(np.sort(rid), np.arange(len(rid))):
        raise ValueError(f"{path}: row ids are not 0..n-1")
    X = np.stack([tab.column(f"{prefix}{i}").to_numpy() for i in range(n)], axis=1)
    out = np.empty_like(X)
    out[rid] = X
    return out.astype(np.float32)


def widen(table: pa.Table, values: np.ndarray, what: str, spec: dict | None = None) -> pa.Table:
    """Append NaN from the stored width (1853) up to the family's first id, then its columns."""
    spec = spec or WIDEN["globaff"]
    ids = spec["ids"]
    width = sum(1 for c in table.column_names if c.startswith("f") and c[1:].isdigit())
    if width != GLOBAFF_IDS[0] - 14 or not np.isfinite(values[:, spec["finite"]]).all():
        raise ValueError(f"{what}: width {width} or nonfinite values")
    nan = pa.array(np.full(table.num_rows, np.nan, np.float32))
    for i in range(width, ids[0]):
        table = table.append_column(f"f{i}", nan)
    for j, i in enumerate(ids):
        table = table.append_column(f"f{i}", pa.array(np.ascontiguousarray(values[:, j])))
    return table


def cmd_globaff(a) -> int:
    return widen_root(a, "globaff")


def cmd_orient(a) -> int:
    return widen_root(a, "orient")


def widen_root(a, name: str) -> int:
    spec = WIDEN[name]
    ids, prefix, fsid = spec["ids"], spec["prefix"], spec["fsid"]
    root, bank, side, out = Path(a.root), Path(a.bank), Path(a.sidecars), Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    shutil.copytree(root, out)
    real = out / "wide/main/real"
    cache = {}
    record = {"source_root": str(root), "source_frozen_sha256": sha(root / "wide/frozen.json"), "ids": ids,
              "sidecars": {}, "tables": {}}
    # A stored root still carries the HDR legs; like `build`, the widened root leaves them out (no sidecar covers them).
    for p in sorted(real.iterdir()):
        if p.name.split(".")[0] in HDR:
            p.unlink()
    for t in sorted(real.glob("*.parquet")):
        if t.name.endswith(".keys.parquet"):
            continue
        keys = pq.read_table(real / f"{t.stem}.keys.parquet", columns=["pair_key", "member_set"])
        members, pks = keys.column("member_set").to_pylist(), keys.column("pair_key").to_pylist()
        vals = np.empty((keys.num_rows, len(ids)), np.float32)
        for m in sorted(set(members)):
            if m not in cache:
                order = pq.read_table(bank / m / "features.parquet", columns=["pair_key"]).column("pair_key").to_pylist()
                cache[m] = ({k: i for i, k in enumerate(order)}, read_sidecar(side / f"{m}.csv", prefix, len(ids)))
                record["sidecars"][m] = sha(side / f"{m}.csv")
            index, X = cache[m]
            sel = [r for r, mm in enumerate(members) if mm == m]
            vals[sel] = X[[index[pks[r]] for r in sel]]
        write(widen(pq.read_table(t), vals, t.stem, spec), t)
        record["tables"][t.name] = sha(t)
    # Coverage pool: the same subsequence alignment `build` checked bit for bit.
    import pandas as pd
    sel = pd.read_parquet(POOL_SOURCE / "selection.parquet")
    pkeys = pd.read_parquet(out / "e15/coverage_pool.keys.parquet")
    fields = ["source_filename", "type", "severity_level", "severity", "sign"]
    want = list(pkeys[fields].astype(str).itertuples(index=False, name=None))
    take, j = [], 0
    for i, row in enumerate(sel[fields].astype(str).itertuples(index=False, name=None)):
        if j < len(want) and row == want[j]:
            take.append(i)
            j += 1
    if j != len(want):
        raise ValueError("coverage pool alignment failed")
    pool = out / "e15/coverage_pool.parquet"
    write(widen(pq.read_table(pool), read_sidecar(side / "pool.csv", prefix, len(ids))[take], "coverage_pool", spec), pool)
    record["sidecars"]["pool"] = sha(side / "pool.csv")
    record["tables"]["e15/coverage_pool.parquet"] = sha(pool)

    note = spec["note"]
    for table_name in list(record["tables"]):
        path = out / table_name if table_name.startswith("e15/") else real / table_name
        mp = Path(f"{path}.manifest.json")
        m = json.loads(mp.read_text())
        derived = {"table_sha256": m.get("table_sha256"), "manifest_sha256": sha(mp)}
        m[f"byc_{name}_derived_from"] = derived
        m.setdefault("byc_derived_from", derived)  # a stored (not yet derived) root becomes a derived one here
        m["table_sha256"] = sha(path)
        if "source_table_sha256" in m:
            m["source_table_sha256"] = m["table_sha256"]
        m["byc_globaff_ids"] = ids
        m["feature_set_id"] = fsid
        m["source_bank_feature_set_id"] = fsid
        m["composite"] = m.get("composite", "") + note
        m["immutable_input_roots"] = [*m.get("immutable_input_roots", []), str(root.resolve()), str(side.resolve())]
        mp.write_text(json.dumps(m, indent=1) + "\n")
    receipt = json.loads((real / "receipt.json").read_text())
    receipt["legs"].pop("hdr_consensus", None)
    for leg in receipt["legs"].values():
        for part in leg.values():
            if isinstance(part, dict) and "rel" in part:
                q = out / part["rel"]
                part["sha256"] = sha(q)
                part["manifest_sha256"] = sha(Path(f"{q}.manifest.json"))
    receipt["width"] = ids[-1] + 1
    receipt["feature_set_id"] = fsid
    receipt[f"byc_{name}"] = record
    view = dict(receipt["admission_view"])
    view["immutable_input_roots"] = [*view["immutable_input_roots"], str(root.resolve()), str(side.resolve())]
    receipt["admission_view"] = view
    (real / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    frozen = json.loads((out / "wide/frozen.json").read_text())
    frozen["admission_view"] = view
    frozen["wide_receipts"] = {"main/real": sha(real / "receipt.json")}
    frozen["auxiliary_files"] = {rel: sha(out / rel) for rel in frozen["auxiliary_files"] if not any(h in rel for h in HDR)}
    frozen[f"byc_{name}"] = record
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    log(f"{name} root written: {out}")
    return 0


GATED_FIRST = GLOBAFF_WIDTH
# Gated channels (XYB order, cell k = 3 * scale + channel) -> (gated difference count, feature set id, declaration
# registration text). `byc_fsid basic+peaks+v2 <first + count> <era> 0-227,372-719,1867-<first + count - 1>` (hash from
# zensim::feature_set_id).
GATED = {
    "XYB": ((0, 1, 2), 410, "basic+peaks+v2@w2301/rev5_localwin_ringhpm_globaffgate#45e0e84c",
            "exploration only: benchmarks/byconstruction_2026-10-22.md (fx1 plus gated damage at research ids f1891-f2300: "
            "each difference x (1 - byc_globaff share of its scale and channel cell); the descriptor itself is not read)"),
    "Y": ((1,), 164, "basic+peaks+v2@w2055/rev5_localwin_ringhpm_globaffgatey#7b645e91",
          "exploration only: benchmarks/byconstruction_2026-10-22.md (fx1 plus gated luminance damage at research ids "
          "f1891-f2054: each Y-cell difference x (1 - byc_globaff share of its scale's Y cell); the descriptor itself is "
          "not read)"),
}
GATED_WIDTH = GATED_FIRST + GATED["XYB"][1]
GATED_FSID = GATED["XYB"][2]


def gated_plan(decl: Path, channels: str = "XYB") -> tuple[list[int], list[int]]:
    """fx1's differences in a gated channel, in ascending order, and the share id of each one's (scale, channel) cell:
    the declaration pairs each difference with the `pjnd_fragility` slot of its own cell, `372 + 29k + 21` (E33
    section 5.1, k = 3 * scale + channel), and the descriptor's share of cell k sits at `1867 + 2k`."""
    gated_channels, count, _, _ = GATED[channels]
    pairs = json.loads(decl.read_text())["products"]
    if [a for a, _ in pairs] != sorted(a for a, _ in pairs) or len(pairs) != 410:
        raise ValueError(f"{decl}: expected 410 differences in ascending order")
    diffs, cells = [], []
    for a, f in pairs:
        k, rem = divmod(f - 372 - 21, 29)
        if rem or not 0 <= k < 12:
            raise ValueError(f"{decl}: {f} is not a pjnd_fragility slot")
        if k % 3 in gated_channels:
            diffs.append(a)
            cells.append(GLOBAFF_IDS[0] + 2 * k)
    if len(diffs) != count:
        raise ValueError(f"{decl}: {len(diffs)} differences in channels {channels}, expected {count}")
    return diffs, cells


def gate_table(table: pa.Table, diffs: list[int], shares: list[int], what: str) -> pa.Table:
    width = sum(1 for c in table.column_names if c.startswith("f") and c[1:].isdigit())
    if width != GATED_FIRST:
        raise ValueError(f"{what}: width {width}, expected the descriptor root's {GATED_FIRST}")
    for j, (d, sh) in enumerate(zip(diffs, shares)):
        dv = table.column(f"f{d}").to_numpy(zero_copy_only=False).astype(np.float32)
        u = np.float32(1) - table.column(f"f{sh}").to_numpy(zero_copy_only=False).astype(np.float32)
        g = dv * u
        if not np.isfinite(g).all() or (g < 0).any():
            raise ValueError(f"{what}: f{d} x (1 - f{sh}) is negative or nonfinite")
        table = table.append_column(f"f{GATED_FIRST + j}", pa.array(g))
    return table


def cmd_gate_table(a) -> int:
    """One table; a mixture table's `table.json` beside it is carried over with the gating recorded."""
    src, out = Path(a.table), Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    diffs, shares = gated_plan(Path(a.decl), a.channels)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(gate_table(pq.read_table(src), diffs, shares, src.name), out)
    meta = src.parent / "table.json"
    if src.name == "table.parquet" and out.name == "table.parquet" and meta.is_file():
        m = json.loads(meta.read_text())
        m["gated"] = {"from_table_sha256": sha(src), "ids": [GATED_FIRST, GATED_FIRST + len(diffs) - 1],
                      **({} if a.channels == "XYB" else {"channels": a.channels}), "builder": "ring_root.py gate-table"}
        (out.parent / "table.json").write_text(json.dumps(m, indent=1) + "\n")
    log(f"gated ({a.channels}) {src} -> {out} ({sha(out)[:12]})")
    return 0


def cmd_gated_decl(a) -> int:
    """The declaration a gated root trains with (`zensim-fx1-v1`): fx1's differences and the gated ids direct (the
    gated ids also unpaired), fx1's fragility products; the descriptor itself is not read."""
    out = Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    src = json.loads(Path(a.decl).read_text())
    diffs, _ = gated_plan(Path(a.decl), a.channels)
    ids = list(range(GATED_FIRST, GATED_FIRST + len(diffs)))
    decl = {"schema": "zensim-fx1-v1", "registration": GATED[a.channels][3],
            "direct": [i for i in src["direct"] if i not in GLOBAFF_IDS] + ids, "products": src["products"],
            "unpaired": ids}
    out.write_text(json.dumps(decl, indent=1) + "\n")
    log(f"declaration ({a.channels}) {out} ({sha(out)})")
    return 0


def cmd_gated(a) -> int:
    root, out = Path(a.root), Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    diffs, shares = gated_plan(Path(a.decl), a.channels)
    width, fsid = GATED_FIRST + len(diffs), GATED[a.channels][2]
    shutil.copytree(root, out)
    real = out / "wide/main/real"
    record = {"source_root": str(root), "source_frozen_sha256": sha(root / "wide/frozen.json"),
              "declaration_sha256": sha(Path(a.decl)), "ids": list(range(GATED_FIRST, width)),
              "differences": diffs, "share_ids": shares, "tables": {}}
    if a.channels != "XYB":  # the all-channel root's records stay as they were first written
        record["channels"] = a.channels
    tables = [t for t in sorted(real.glob("*.parquet")) if not t.name.endswith(".keys.parquet")]
    for t in [*tables, out / "e15/coverage_pool.parquet"]:
        write(gate_table(pq.read_table(t), diffs, shares, t.stem), t)
        record["tables"][t.name if t.parent == real else "e15/coverage_pool.parquet"] = sha(t)
        log(f"gated {t.name}")
    note = (f"; gated damage at f{GATED_FIRST}-f{width - 1} (each fx1 difference"
            + ("" if a.channels == "XYB" else f" in channels {a.channels}") + " x (1 - byc_globaff share of its cell))")
    ids = list(range(GLOBAFF_IDS[0], width))
    for name in list(record["tables"]):
        path = out / name if name.startswith("e15/") else real / name
        mp = Path(f"{path}.manifest.json")
        m = json.loads(mp.read_text())
        m["byc_gated_derived_from"] = {"table_sha256": m.get("table_sha256"), "manifest_sha256": sha(mp)}
        m["table_sha256"] = sha(path)
        if "source_table_sha256" in m:
            m["source_table_sha256"] = m["table_sha256"]
        m["byc_globaff_ids"] = ids
        m["feature_set_id"] = fsid
        m["source_bank_feature_set_id"] = fsid
        m["composite"] = m.get("composite", "") + note
        m["immutable_input_roots"] = [*m.get("immutable_input_roots", []), str(root.resolve())]
        mp.write_text(json.dumps(m, indent=1) + "\n")
    receipt = json.loads((real / "receipt.json").read_text())
    for leg in receipt["legs"].values():
        for part in leg.values():
            if isinstance(part, dict) and "rel" in part:
                q = out / part["rel"]
                part["sha256"] = sha(q)
                part["manifest_sha256"] = sha(Path(f"{q}.manifest.json"))
    receipt["width"] = width
    receipt["feature_set_id"] = fsid
    receipt["byc_gated"] = record
    view = dict(receipt["admission_view"])
    view["immutable_input_roots"] = [*view["immutable_input_roots"], str(root.resolve())]
    receipt["admission_view"] = view
    (real / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    frozen = json.loads((out / "wide/frozen.json").read_text())
    frozen["admission_view"] = view
    frozen["wide_receipts"] = {"main/real": sha(real / "receipt.json")}
    frozen["auxiliary_files"] = {rel: sha(out / rel) for rel in frozen["auxiliary_files"]}
    frozen["byc_gated"] = record
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    log(f"gated root written: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--arm", choices=sorted(ARMS), default="hpm")
    p.add_argument("--bin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=24)
    p.set_defaults(fn=cmd_pool)
    b = sub.add_parser("build")
    b.add_argument("--arm", choices=sorted(ARMS), default="hpm")
    b.add_argument("--bank", required=True)
    b.add_argument("--pool", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    g = sub.add_parser("globaff-pairs")
    g.add_argument("--bank", required=True)
    g.add_argument("--pool", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_globaff_pairs)
    w = sub.add_parser("globaff")
    w.add_argument("--root", required=True)
    w.add_argument("--bank", required=True)
    w.add_argument("--sidecars", required=True)
    w.add_argument("--out", required=True)
    w.set_defaults(fn=cmd_globaff)
    o = sub.add_parser("orient")
    o.add_argument("--root", required=True)
    o.add_argument("--bank", required=True)
    o.add_argument("--sidecars", required=True)
    o.add_argument("--out", required=True)
    o.set_defaults(fn=cmd_orient)
    t = sub.add_parser("gate-table")
    t.add_argument("--table", required=True)
    t.add_argument("--decl", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--channels", choices=sorted(GATED), default="XYB")
    t.set_defaults(fn=cmd_gate_table)
    q = sub.add_parser("gated")
    q.add_argument("--root", required=True)
    q.add_argument("--decl", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--channels", choices=sorted(GATED), default="XYB")
    q.set_defaults(fn=cmd_gated)
    gd = sub.add_parser("gated-decl")
    gd.add_argument("--decl", required=True)
    gd.add_argument("--out", required=True)
    gd.add_argument("--channels", choices=sorted(GATED), default="XYB")
    gd.set_defaults(fn=cmd_gated_decl)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
