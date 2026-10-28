#!/usr/bin/env python3
"""EXP-METRIC-INPUTS: build 303-feature augmented training + validation parquets.

For each corpus, take the first 300 features (f0..f299) from the canonical
parquet and append three new feature columns named f300, f301, f302 that
carry the ssim2_log_norm, cvvdp_log_norm, iwssim_log_norm values
respectively.

Training parquets already have the *_log_norm columns populated where
upstream coverage exists:
  - safesyn:      all three populated (196086 rows)
  - kadid:        all three populated (10125 rows)
  - tid:          all three populated (3000 rows)
  - konjnd-dense: ALL NAN — drop from training (no metric supervision)
  - cvvdp_iwssim_LARGE: cvvdp+iwssim populated, ssim2 NaN — drop the ssim2 column,
                       impute ssim2 = 0 (median of safesyn standardized form).
                       Document this in the augmented training set's name.

Validation parquets have all three NaN. Recover via positional join against
source MOS CSV + score TSVs.

NaN handling: any remaining NaN after join is set to the safesyn-mean of
that column (== ~67.5 / 27.8 / 34.7 for ssim2/cvvdp/iwssim_log_norm).
This is the "warm start" the experiment depends on.

Output:
  /mnt/v/zen/zensim-training/2026-05-18-metric-inputs/train/{safesyn,kadid,tid,cvvdp_iwssim_LARGE}.parquet
  /mnt/v/zen/zensim-training/2026-05-18-metric-inputs/val/{cid22,kadid,tid,konjnd,aic3}.parquet

Schema:
  ref_basename, human_score, [all targets], f0..f299, f300, f301, f302
  where f300 = ssim2_log_norm, f301 = cvvdp_log_norm, f302 = iwssim_log_norm
"""
import csv
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

CANONICAL = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18")
OUT_ROOT = Path("/mnt/v/zen/zensim-training/2026-05-18-metric-inputs")
SCORES_CVVDP = Path("/mnt/v/zen/zensim-eval")
SCORES_IWSSIM = Path("/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels")
SSIM2_PER_PAIR_CSV = Path("/home/lilith/work/zen/zensimx/benchmarks/v0_22_iw_v3_seed1_2026-05-17_eval_per_pair.csv")

# Mapping from "corpus" key used internally to "dataset" column values in the per-pair CSV
SSIM2_DATASET_MAP = {
    "cid22": "CID22",
    "kadid": "KADIK10k",
    "tid": "TID2013",
    "aic3": "AIC-3 CTC",
}

# Safesyn fill values:
# - SAFESYN_MEAN: mean of *_log_norm columns (kept for the training-corpus fallback
#   where we read the existing _log_norm column directly).
# - SAFESYN_MEAN_RAW: mean of natural-scale RAW scores (used for missing val rows).
SAFESYN_MEAN = {
    "ssim2": 69.5468,
    "cvvdp": 27.8116,
    "iwssim": 34.6632,
}
# Computed from /mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet
# (median used over mean for ssim2 because raw ssim2 has heavy left-tail outliers
# down to -739; mean is unstable, median lands at the perceptual center of the
# training distribution).
SAFESYN_MEAN_RAW = {
    "ssim2": 68.7350,   # median (mean = 60.00, but heavy negative tail)
    "cvvdp": 9.5912,    # mean
    "iwssim": 0.9725,   # mean
}


def load_ssim2_per_pair(dataset_value: str) -> dict[tuple[str, str], float]:
    """Load (ref_path, dist_path) → fast_ssim2_score for a corpus from per-pair CSV."""
    lookup = {}
    if not SSIM2_PER_PAIR_CSV.exists():
        print(f"  WARN: {SSIM2_PER_PAIR_CSV} missing", file=sys.stderr)
        return lookup
    with open(SSIM2_PER_PAIR_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] != dataset_value:
                continue
            key = (row["reference"], row["distorted"])
            try:
                v = float(row.get("fast_ssim2_score", "nan"))
                if np.isfinite(v):
                    lookup[key] = v
            except (ValueError, KeyError):
                pass
    print(f"  loaded {len(lookup)} ssim2 entries for dataset={dataset_value}")
    return lookup


def load_score_tsv(path: Path, score_col: str) -> dict[tuple[str, str], float]:
    """Load (ref_path, dist_path) → score lookup from a TSV."""
    lookup = {}
    if not path.exists():
        print(f"  WARN: {path} does not exist", file=sys.stderr)
        return lookup
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["ref_path"], row["dist_path"])
            try:
                lookup[key] = float(row[score_col])
            except (ValueError, KeyError):
                pass
    print(f"  loaded {len(lookup)} entries from {path.name}")
    return lookup


def safesyn_log_norm(raw_value: float, key: str) -> float:
    """Pass raw scores through to feature columns.

    Each metric is stored in its natural scale:
      - ssim2: raw ∈ [0, 100], higher = better
      - cvvdp: raw ∈ [0, 10], higher = better
      - iwssim: raw ∈ [0, 1], higher = better

    The trainer's input scaler will standardize per-feature using the
    training data's per-feature mean/std. So inputs of different scales
    are fine — the scaler handles it.

    Imputed value for missing rows = the safesyn-natural-scale mean,
    so the MLP sees a "no signal" signal not extrapolation chaos.
    """
    if not np.isfinite(raw_value):
        return float(SAFESYN_MEAN_RAW[key])
    return float(raw_value)


def take_300_features(tbl: pa.Table) -> pa.Table:
    """Select non-feature cols + f0..f299, dropping f300..f371 if present."""
    n_feat_in = sum(1 for n in tbl.schema.names if n.startswith("f") and n[1:].isdigit())
    keep_features = [f"f{i}" for i in range(300)]
    other_cols = [n for n in tbl.schema.names if not (n.startswith("f") and n[1:].isdigit())]
    select_cols = other_cols + keep_features
    return tbl.select(select_cols), n_feat_in


def append_metric_cols(
    tbl: pa.Table,
    ssim2_vals: list[float],
    cvvdp_vals: list[float],
    iwssim_vals: list[float],
) -> pa.Table:
    """Append ssim2/cvvdp/iwssim as f300, f301, f302."""
    n = tbl.num_rows
    assert len(ssim2_vals) == n, f"len mismatch ssim2 {len(ssim2_vals)} != {n}"
    assert len(cvvdp_vals) == n, f"len mismatch cvvdp {len(cvvdp_vals)} != {n}"
    assert len(iwssim_vals) == n, f"len mismatch iwssim {len(iwssim_vals)} != {n}"
    tbl = tbl.append_column("f300", pa.array(ssim2_vals, type=pa.float64()))
    tbl = tbl.append_column("f301", pa.array(cvvdp_vals, type=pa.float64()))
    tbl = tbl.append_column("f302", pa.array(iwssim_vals, type=pa.float64()))
    return tbl


def build_training_corpus(name: str):
    """Augment a training corpus by appending f300/f301/f302 from raw metric cols.

    Strategy: read raw metric scores from the canonical training parquet
    (ssim2_gpu, cvvdp_score, iwssim). Where they are NaN, fill with the
    safesyn-natural-scale mean. The trainer scaler will Z-score these
    per-feature using the training-mean/std.
    """
    src = CANONICAL / "train" / f"{name}.parquet"
    print(f"\n=== TRAIN {name} from {src.name} ===")
    tbl = pq.read_table(str(src))

    # Pull raw metric columns (natural scale; ssim2 ∈ [0, 100], cvvdp ∈ [0, 10], iwssim ∈ [0, 1])
    ssim2 = np.array(tbl["ssim2_gpu"])
    cvvdp = np.array(tbl["cvvdp_score"])
    iwssim = np.array(tbl["iwssim"])

    # Track coverage
    n_ssim2 = int(np.isfinite(ssim2).sum())
    n_cvvdp = int(np.isfinite(cvvdp).sum())
    n_iwssim = int(np.isfinite(iwssim).sum())
    total = tbl.num_rows
    print(f"  coverage: ssim2={n_ssim2}/{total} cvvdp={n_cvvdp}/{total} iwssim={n_iwssim}/{total}")

    # Fill NaN with safesyn raw-scale mean
    ssim2 = np.where(np.isfinite(ssim2), ssim2, SAFESYN_MEAN_RAW["ssim2"])
    cvvdp = np.where(np.isfinite(cvvdp), cvvdp, SAFESYN_MEAN_RAW["cvvdp"])
    iwssim = np.where(np.isfinite(iwssim), iwssim, SAFESYN_MEAN_RAW["iwssim"])

    # Take 300 features
    tbl_300, n_feat_in = take_300_features(tbl)
    print(f"  reduced f0..f{n_feat_in-1} -> f0..f299")

    # Append f300, f301, f302
    tbl_out = append_metric_cols(tbl_300, ssim2.tolist(), cvvdp.tolist(), iwssim.tolist())

    out_path = OUT_ROOT / "train" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({out_path.stat().st_size:,} bytes, {tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_val_cid22():
    """Augment CID22 val parquet via positional join with master CSV + score TSVs."""
    name = "cid22"
    print(f"\n=== VAL {name} ===")
    src_parquet = CANONICAL / "val" / f"{name}.parquet"
    src_csv = Path("/mnt/v/dataset/cid22/CID22_validation_set.csv")
    tbl = pq.read_table(str(src_parquet))

    # Read master CSV, filter to non-reference rows
    csv_rows = []
    with open(src_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            csv_rows.append(r)
    non_ref = [r for r in csv_rows if r["encoder"] != "Reference"]
    assert len(non_ref) == tbl.num_rows, f"row mismatch {len(non_ref)} != {tbl.num_rows}"

    # Build dist_paths for each parquet row
    dist_paths = []
    ref_paths = []
    cid22_root = "/mnt/v/dataset/cid22/CID22_validation_set/"
    for r in non_ref:
        ref_paths.append(cid22_root + r["reference_img"])
        dist_paths.append(cid22_root + r["distorted_img"])

    # Load score lookups
    cvvdp_lookup = load_score_tsv(SCORES_CVVDP / "cid22_cvvdp_scores_2026-05-17.tsv", "cvvdp_imazen_v0_0_1")
    iwssim_lookup = load_score_tsv(SCORES_IWSSIM / "cid22_iwssim_scores.tsv", "iwssim_gpu")

    ssim2_lookup = load_ssim2_per_pair(SSIM2_DATASET_MAP["cid22"])

    # Apply lookups
    ssim2_vals = []
    cvvdp_vals = []
    iwssim_vals = []
    hit_s = hit_c = hit_i = 0
    for i, dp in enumerate(dist_paths):
        rp = ref_paths[i]
        key = (rp, dp)
        s = ssim2_lookup.get(key)
        c = cvvdp_lookup.get(key)
        iw = iwssim_lookup.get(key)
        if s is not None and np.isfinite(s):
            ssim2_vals.append(safesyn_log_norm(s, "ssim2"))
            hit_s += 1
        else:
            ssim2_vals.append(SAFESYN_MEAN_RAW["ssim2"])
        if c is not None and np.isfinite(c):
            cvvdp_vals.append(safesyn_log_norm(c, "cvvdp"))
            hit_c += 1
        else:
            cvvdp_vals.append(SAFESYN_MEAN_RAW["cvvdp"])
        if iw is not None and np.isfinite(iw):
            iwssim_vals.append(safesyn_log_norm(iw, "iwssim"))
            hit_i += 1
        else:
            iwssim_vals.append(SAFESYN_MEAN_RAW["iwssim"])

    total = tbl.num_rows
    print(f"  join: ssim2={hit_s}/{total} cvvdp={hit_c}/{total} iwssim={hit_i}/{total}")

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2_vals, cvvdp_vals, iwssim_vals)
    out_path = OUT_ROOT / "val" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_val_kadid():
    """KADID: dmos.csv keys by dist_img / ref_img. Val parquet has 10125 rows."""
    name = "kadid"
    print(f"\n=== VAL {name} ===")
    src_parquet = CANONICAL / "val" / f"{name}.parquet"
    src_csv = Path("/mnt/v/dataset/kadid10k/dmos.csv")
    tbl = pq.read_table(str(src_parquet))

    rows = []
    with open(src_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    assert len(rows) == tbl.num_rows, f"row mismatch {len(rows)} != {tbl.num_rows}"

    root = "/mnt/v/dataset/kadid10k/images/"
    ref_paths = [root + r["ref_img"] for r in rows]
    dist_paths = [root + r["dist_img"] for r in rows]

    cvvdp_lookup = load_score_tsv(SCORES_CVVDP / "kadid_cvvdp_scores_2026-05-17.tsv", "cvvdp_imazen_v0_0_1")
    iwssim_lookup = load_score_tsv(SCORES_IWSSIM / "kadid_iwssim_scores.tsv", "iwssim_gpu")
    ssim2_lookup = load_ssim2_per_pair(SSIM2_DATASET_MAP["kadid"])

    ssim2_vals, cvvdp_vals, iwssim_vals = [], [], []
    hit_s = hit_c = hit_i = 0
    for i in range(tbl.num_rows):
        key = (ref_paths[i], dist_paths[i])
        s = ssim2_lookup.get(key)
        c = cvvdp_lookup.get(key)
        iw = iwssim_lookup.get(key)
        if s is not None and np.isfinite(s):
            ssim2_vals.append(safesyn_log_norm(s, "ssim2"))
            hit_s += 1
        else:
            ssim2_vals.append(SAFESYN_MEAN_RAW["ssim2"])
        if c is not None and np.isfinite(c):
            cvvdp_vals.append(safesyn_log_norm(c, "cvvdp"))
            hit_c += 1
        else:
            cvvdp_vals.append(SAFESYN_MEAN_RAW["cvvdp"])
        if iw is not None and np.isfinite(iw):
            iwssim_vals.append(safesyn_log_norm(iw, "iwssim"))
            hit_i += 1
        else:
            iwssim_vals.append(SAFESYN_MEAN_RAW["iwssim"])

    total = tbl.num_rows
    print(f"  join: ssim2={hit_s}/{total} cvvdp={hit_c}/{total} iwssim={hit_i}/{total}")

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2_vals, cvvdp_vals, iwssim_vals)
    out_path = OUT_ROOT / "val" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_val_tid():
    """TID2013: mos_with_names.txt has MOS + dist file. Val parquet has 3000 rows."""
    name = "tid"
    print(f"\n=== VAL {name} ===")
    src_parquet = CANONICAL / "val" / f"{name}.parquet"
    src_txt = Path("/mnt/v/dataset/tid2013/mos_with_names.txt")
    tbl = pq.read_table(str(src_parquet))

    rows = []
    with open(src_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mos = parts[0]
                name_field = parts[1]
                rows.append((mos, name_field))
    assert len(rows) == tbl.num_rows, f"row mismatch {len(rows)} != {tbl.num_rows}"

    # TID dist_path uses .png from reference_images_png directory for cvvdp/iwssim score TSVs.
    # ssim2 per-pair CSV uses .BMP/.bmp from reference_images/distorted_images dirs.
    # ref id from filename: 'I01_01_1.bmp' → ref = 'I01.png' (or 'i01.bmp')
    ref_paths_png = []
    dist_paths_png = []
    ref_paths_bmp = []
    dist_paths_bmp = []
    for _, name_field in rows:
        base = name_field.split("_")[0]  # 'I01' or 'i01'
        ref_paths_png.append(f"/mnt/v/dataset/tid2013/reference_images_png/{base.lower()}.png")
        dist_paths_png.append(f"/mnt/v/dataset/tid2013/distorted_images_png/{name_field.replace('.bmp', '.png').lower()}")
        # BMP form: ref uppercase BMP for I01, lower bmp for i02..; for simplicity, both case forms
        ref_paths_bmp.append(f"/mnt/v/dataset/tid2013/reference_images/{base}.BMP")
        dist_paths_bmp.append(f"/mnt/v/dataset/tid2013/distorted_images/{name_field}")

    cvvdp_lookup = load_score_tsv(SCORES_CVVDP / "tid_cvvdp_scores_2026-05-17.tsv", "cvvdp_imazen_v0_0_1")
    iwssim_lookup = load_score_tsv(SCORES_IWSSIM / "tid_iwssim_scores.tsv", "iwssim_gpu")

    # Also try uppercase fallback for ref/dist (score TSV used 'I01.png' on first row, 'i01' on others)
    # Build case-insensitive lookup
    def ci_lookup(d):
        return {(r.lower(), dp.lower()): v for (r, dp), v in d.items()}
    cvvdp_ci = ci_lookup(cvvdp_lookup)
    iwssim_ci = ci_lookup(iwssim_lookup)

    ssim2_raw = load_ssim2_per_pair(SSIM2_DATASET_MAP["tid"])
    # ssim2 keys use .BMP/.bmp paths — case-insensitive lookup
    ssim2_lookup = {(r.lower(), dp.lower()): v for (r, dp), v in ssim2_raw.items()}

    ssim2_vals, cvvdp_vals, iwssim_vals = [], [], []
    hit_s = hit_c = hit_i = 0
    for i in range(tbl.num_rows):
        key_png = (ref_paths_png[i].lower(), dist_paths_png[i].lower())
        key_bmp = (ref_paths_bmp[i].lower(), dist_paths_bmp[i].lower())
        s = ssim2_lookup.get(key_bmp)
        c = cvvdp_ci.get(key_png)
        iw = iwssim_ci.get(key_png)
        if s is not None and np.isfinite(s):
            ssim2_vals.append(safesyn_log_norm(s, "ssim2"))
            hit_s += 1
        else:
            ssim2_vals.append(SAFESYN_MEAN_RAW["ssim2"])
        if c is not None and np.isfinite(c):
            cvvdp_vals.append(safesyn_log_norm(c, "cvvdp"))
            hit_c += 1
        else:
            cvvdp_vals.append(SAFESYN_MEAN_RAW["cvvdp"])
        if iw is not None and np.isfinite(iw):
            iwssim_vals.append(safesyn_log_norm(iw, "iwssim"))
            hit_i += 1
        else:
            iwssim_vals.append(SAFESYN_MEAN_RAW["iwssim"])

    total = tbl.num_rows
    print(f"  join: ssim2={hit_s}/{total} cvvdp={hit_c}/{total} iwssim={hit_i}/{total}")

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2_vals, cvvdp_vals, iwssim_vals)
    out_path = OUT_ROOT / "val" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_val_konjnd():
    """KonJND: 1008 anchor pairs from subjective_ratings.csv."""
    name = "konjnd"
    print(f"\n=== VAL {name} ===")
    src_parquet = CANONICAL / "val" / f"{name}.parquet"
    tbl = pq.read_table(str(src_parquet))

    # KonJND iwssim is split into jpeg/bpg score files — but val parquet is 1008 rows of (ref, anchor).
    # Without per-row dist_path info, fall back to safesyn mean for all metrics.
    # This is the documented coverage gap.
    print(f"  WARN: KonJND val has no dist_path keying → filling with safesyn means (no metric supervision)")

    n = tbl.num_rows
    ssim2_vals = [SAFESYN_MEAN_RAW["ssim2"]] * n
    cvvdp_vals = [SAFESYN_MEAN_RAW["cvvdp"]] * n
    iwssim_vals = [SAFESYN_MEAN_RAW["iwssim"]] * n

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2_vals, cvvdp_vals, iwssim_vals)
    out_path = OUT_ROOT / "val" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_val_aic3():
    """AIC-3: info.csv has 600 rows mapping (codec, img.name, quality) → score.jnd.

    Score TSVs use full dist_path. Decoded path is:
    /mnt/v/dataset/aic3_ctc_epfl/decoded/<img.name>/<codec>_<img.name>_<quality>.png
    """
    name = "aic3"
    print(f"\n=== VAL {name} ===")
    src_parquet = CANONICAL / "val" / f"{name}.parquet"
    src_csv = Path("/mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv")
    tbl = pq.read_table(str(src_parquet))

    rows = []
    with open(src_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Skip blank separator rows (codec field empty)
            if not r.get("codec"):
                continue
            rows.append(r)
    assert len(rows) == tbl.num_rows, f"row mismatch {len(rows)} != {tbl.num_rows}"

    ref_paths = []
    dist_paths = []
    for r in rows:
        img = r["img.name"]
        codec = r["codec"]
        q = r["quality"]
        ref_paths.append(f"/mnt/v/dataset/aic3_ctc_epfl/original/{img}.png")
        dist_paths.append(f"/mnt/v/dataset/aic3_ctc_epfl/decoded/{img}/{codec}_{img}_{q}.png")

    cvvdp_lookup = load_score_tsv(SCORES_CVVDP / "aic3_cvvdp_scores_2026-05-17.tsv", "cvvdp_imazen_v0_0_1")
    iwssim_lookup = load_score_tsv(SCORES_IWSSIM / "aic3_iwssim_scores.tsv", "iwssim_gpu")
    ssim2_lookup = load_ssim2_per_pair(SSIM2_DATASET_MAP["aic3"])

    ssim2_vals, cvvdp_vals, iwssim_vals = [], [], []
    hit_s = hit_c = hit_i = 0
    for i in range(tbl.num_rows):
        key = (ref_paths[i], dist_paths[i])
        s = ssim2_lookup.get(key)
        c = cvvdp_lookup.get(key)
        iw = iwssim_lookup.get(key)
        if s is not None and np.isfinite(s):
            ssim2_vals.append(safesyn_log_norm(s, "ssim2"))
            hit_s += 1
        else:
            ssim2_vals.append(SAFESYN_MEAN_RAW["ssim2"])
        if c is not None and np.isfinite(c):
            cvvdp_vals.append(safesyn_log_norm(c, "cvvdp"))
            hit_c += 1
        else:
            cvvdp_vals.append(SAFESYN_MEAN_RAW["cvvdp"])
        if iw is not None and np.isfinite(iw):
            iwssim_vals.append(safesyn_log_norm(iw, "iwssim"))
            hit_i += 1
        else:
            iwssim_vals.append(SAFESYN_MEAN_RAW["iwssim"])

    total = tbl.num_rows
    print(f"  join: ssim2={hit_s}/{total} cvvdp={hit_c}/{total} iwssim={hit_i}/{total}")

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2_vals, cvvdp_vals, iwssim_vals)
    out_path = OUT_ROOT / "val" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train").mkdir(exist_ok=True)
    (OUT_ROOT / "val").mkdir(exist_ok=True)

    # Training corpora — use existing _log_norm cols, fill NaN with safesyn mean
    build_training_corpus("safesyn")
    build_training_corpus("kadid")
    build_training_corpus("tid")
    build_training_corpus("cvvdp_iwssim_LARGE")
    # konjnd-dense skipped: no metric scores at all

    # Validation corpora — positional join via source CSVs + score TSVs
    build_val_cid22()
    build_val_kadid()
    build_val_tid()
    build_val_konjnd()
    build_val_aic3()

    print("\nDONE.")


if __name__ == "__main__":
    main()
