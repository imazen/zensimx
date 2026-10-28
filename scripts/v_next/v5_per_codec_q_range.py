#!/usr/bin/env python3
"""V5 per-codec q-range analysis (2026-05-24 PM).

For each codec at /mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet
(1000 refs × 19 q levels), score every (ref, q) pair through the v5
packed bake, then compute:

- median score per q level (per codec) — the codec's dial behavior
- score range per codec (min..max across all q)
- which q values are "discretely targetable" — adjacent-q median scores
  differing by ≥ TOL (1.0 score unit, the dial's cross-codec p50)
- piecewise zones: do the dial dynamics differ above vs below score 60?

Output: markdown report with per-codec dial coverage tables.
"""
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/home/lilith/work/zen/zensimx")
BAKE = REPO / "zensim-experimental/weights/v_tuner_v11_2026-05-24.bin"
PREDICT = REPO / "target/release/predict_features_with_bake"
PARQ_ROOT = Path("/mnt/v/zen/picker-training/2026-05-19/butter")
OUT_MD = REPO / "benchmarks/v_tuner_v5_per_codec_q_range_2026-05-24.md"

CODECS = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]
N_FEATURES = 372
TOL = 1.0  # score units — minimum gap for adjacent-q discrete targetability


def score_features(features: np.ndarray) -> np.ndarray:
    """Score (n_rows, n_features) f32 → (n_rows,) f64 via predict_features_with_bake."""
    n_rows, n_features = features.shape
    buf = bytearray()
    buf += struct.pack("<II", n_features, n_rows)
    buf += features.astype(np.float32, copy=False).tobytes(order="C")
    with tempfile.NamedTemporaryFile(suffix=".features.bin", delete=False) as f:
        f.write(buf)
        feats_path = f.name
    try:
        out = subprocess.check_output(
            [str(PREDICT), "--bake", str(BAKE),
             "--bake-post", "clamp", "--features-file", feats_path],
        )
    finally:
        Path(feats_path).unlink(missing_ok=True)
    return np.array([float(x) for x in out.decode().split()], dtype=np.float64)


def per_codec_table(codec: str) -> tuple[list[int], np.ndarray]:
    """For a codec, return (q_values_sorted, per_q_scores_2D[n_refs, n_q])."""
    p = PARQ_ROOT / f"{codec}.parquet"
    table = pq.read_table(p)
    n_rows = table.num_rows
    feat_cols = [f"f{i}" for i in range(N_FEATURES)]
    qs = table.column("q").to_numpy()
    refs = table.column("ref_basename").to_pylist()
    feats = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in feat_cols]
    ).astype(np.float32, copy=False)
    print(f"  {codec}: scoring {n_rows} rows × {N_FEATURES} features...")
    scores = score_features(feats)

    # Build per-(ref, q) score grid.
    unique_qs = sorted(set(int(q) for q in qs))
    unique_refs = sorted(set(refs))
    score_grid = np.full((len(unique_refs), len(unique_qs)), np.nan)
    ref_idx = {r: i for i, r in enumerate(unique_refs)}
    q_idx = {q: i for i, q in enumerate(unique_qs)}
    for i in range(n_rows):
        r = refs[i]
        q = int(qs[i])
        score_grid[ref_idx[r], q_idx[q]] = scores[i]

    return unique_qs, score_grid


def main() -> None:
    md = []
    md.append("# V_tuner_v5 per-codec q-range analysis (2026-05-24 PM)")
    md.append("")
    md.append(f"- **Bake:** `{BAKE.relative_to(REPO)}` (packed i8 + zerobias + lz4, 54 KB)")
    md.append(f"- **Per-codec parquets:** `{PARQ_ROOT}` (1000 refs × 19 q each)")
    md.append(f"- **Discrete-targetability tolerance:** {TOL} score unit "
              f"(adjacent-q median scores must differ by ≥ TOL)")
    md.append("")
    md.append("## Per-codec dial coverage")
    md.append("")

    all_codec_data = {}
    for codec in CODECS:
        unique_qs, score_grid = per_codec_table(codec)
        all_codec_data[codec] = (unique_qs, score_grid)

    for codec in CODECS:
        unique_qs, score_grid = all_codec_data[codec]
        med = np.nanmedian(score_grid, axis=0)
        p25 = np.nanquantile(score_grid, 0.25, axis=0)
        p75 = np.nanquantile(score_grid, 0.75, axis=0)
        smin = np.nanmin(score_grid, axis=0)
        smax = np.nanmax(score_grid, axis=0)

        md.append(f"### {codec}")
        md.append("")
        md.append(f"- score range (across refs and q): "
                  f"{np.nanmin(score_grid):.1f}..{np.nanmax(score_grid):.1f}")
        md.append(f"- median dial range (q={unique_qs[0]} → q={unique_qs[-1]}): "
                  f"{med[0]:.1f} → {med[-1]:.1f}  "
                  f"(span = {med[-1] - med[0]:.1f} units)")
        md.append("")
        md.append("| q | median | p25 | p75 | min | max | Δ vs prev q |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|")
        prev_med = None
        for i, q in enumerate(unique_qs):
            delta = "" if prev_med is None else f"{med[i] - prev_med:+.2f}"
            md.append(f"| {q} | {med[i]:.2f} | {p25[i]:.2f} | "
                      f"{p75[i]:.2f} | {smin[i]:.2f} | {smax[i]:.2f} | {delta} |")
            prev_med = med[i]
        md.append("")
        # Discrete-targetability: adjacent q pairs with median gap >= TOL.
        gaps = np.diff(med)
        targetable_pairs = sum(1 for g in gaps if abs(g) >= TOL)
        flat_pairs = sum(1 for g in gaps if abs(g) < TOL)
        # Find continuous discrete-targetable range.
        if targetable_pairs > 0:
            first_disc_i = next(i for i, g in enumerate(gaps) if abs(g) >= TOL)
            last_disc_i = len(gaps) - 1 - next(i for i, g in enumerate(reversed(gaps)) if abs(g) >= TOL)
            q_disc_lo, q_disc_hi = unique_qs[first_disc_i], unique_qs[last_disc_i + 1]
        else:
            q_disc_lo, q_disc_hi = None, None
        md.append(f"**Discrete targetability** (adjacent q with |Δ median| ≥ {TOL} score units):")
        md.append("")
        md.append(f"- Targetable adjacent q pairs: **{targetable_pairs} / {len(gaps)}**")
        md.append(f"- Flat (tied) adjacent q pairs: {flat_pairs}")
        if q_disc_lo is not None:
            md.append(f"- Continuous discrete-targetable q range: "
                      f"**q={q_disc_lo} → q={q_disc_hi}** "
                      f"(score {med[unique_qs.index(q_disc_lo)]:.1f} → "
                      f"{med[unique_qs.index(q_disc_hi)]:.1f})")
        md.append("")

    # Piecewise zone analysis.
    md.append("## Piecewise zone analysis — above vs below score 60")
    md.append("")
    md.append("Question: are dial dynamics different above vs below JND (score 60)?")
    md.append("If yes, piecewise calibration (separate splines for the two zones)")
    md.append("could tighten cross-codec consistency in the harder zone.")
    md.append("")
    md.append("| codec | n_q in 0-60 | n_q in 60-100 | per-q Δ p50 in 0-60 | per-q Δ p50 in 60-100 |")
    md.append("|---|--:|--:|--:|--:|")
    for codec in CODECS:
        unique_qs, score_grid = all_codec_data[codec]
        med = np.nanmedian(score_grid, axis=0)
        gaps = np.abs(np.diff(med))
        # Classify each gap by whether the median falls below or above 60.
        zone_below = [g for i, g in enumerate(gaps) if (med[i] + med[i+1]) / 2 < 60]
        zone_above = [g for i, g in enumerate(gaps) if (med[i] + med[i+1]) / 2 >= 60]
        n_below = len(zone_below)
        n_above = len(zone_above)
        p50_below = np.median(zone_below) if zone_below else float("nan")
        p50_above = np.median(zone_above) if zone_above else float("nan")
        md.append(f"| {codec} | {n_below} | {n_above} | {p50_below:.2f} | {p50_above:.2f} |")
    md.append("")
    md.append("**Read:** If per-q Δ p50 is similar above and below 60, the dial is")
    md.append("consistent across zones — no piecewise needed. If the below-60 zone has")
    md.append("much wider per-q Δ, the dial is coarser there and could benefit from")
    md.append("piecewise calibration (denser low-q spline knots).")
    md.append("")
    md.append("## Headline")
    md.append("")
    total_targetable = 0
    total_pairs = 0
    for codec in CODECS:
        unique_qs, score_grid = all_codec_data[codec]
        med = np.nanmedian(score_grid, axis=0)
        gaps = np.abs(np.diff(med))
        total_targetable += sum(1 for g in gaps if g >= TOL)
        total_pairs += len(gaps)
    md.append(f"- Across all 4 codecs × 18 adjacent-q pairs each: "
              f"**{total_targetable}/{total_pairs}** adjacent q pairs are "
              f"discretely targetable at ±{TOL} score unit tolerance.")
    md.append("")

    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
