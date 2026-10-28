#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V7 empirical multi-band anchor builder (2026-05-19).

V6 (commit 1dd61fc, PreviewV0_5TunerV2) shipped with multi-band anchor
parquet using rule-of-thumb target scores:

    butter_pnorm3   target_score (V6, rule of thumb)
    0.3             90
    0.8             75
    1.5             63   <- CID22 paper PJND anchor (the only empirical one)
    2.5             45
    4.0             25
    6.0             10

V7 fix: replace the 5 unmoored values with empirically-derived medians
from the canonical ssim2 + cvvdp score parquets. For each (codec, band)
collect every (image, q) within butter_pnorm3 ∈ [band ± 0.5], look up
ssim2_gpu and cvvdp_imazen, normalize cvvdp via the safesyn-corpus
-log(10 - cvvdp) min-max transform (so the V7 target_score lives in
the same units the trainer's mix_cv40_iw60 target column does), and
emit per-codec band medians for both metrics.

The V6 rule (`63 @ PJND`) is calibrated to ssim2's 0-100 range, not
cvvdp_log_norm's safesyn-corpus-normalized 0-100 range — at PJND
(cvvdp ≈ 9.95) the safesyn-normalized cvvdp_log_norm is ~32, not 63.
For V7's `target_score` column (which the trainer MSEs against the
score-shaped bake output, also living in 0-100 score units), the
primary signal is **median ssim2** — it's the only one already
calibrated to the V6 anchor's 0-100 range with PJND at ~63. cvvdp
medians are reported alongside for transparency but NOT used as the
target (mixing cvvdp_log_norm-units into a ssim2-units target was
the V6 quiet design choice; V7 makes it explicit).

Output anchor parquet has the V5/V6 schema (ref_basename, codec, q,
butter_pnorm3, butter_target, target_score, anchor_source, human_score,
anchor_weight, f0..f371) so it drops directly into the V6 trainer.

Side product: a markdown comparison table at the path given by
--comparison-md, showing V6 rule-of-thumb vs V7 empirical medians
(per band, aggregate, and per codec).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# (butter_pnorm3_target, V6_rule_of_thumb_target_score). Same band centers as V6.
ANCHOR_BANDS: list[tuple[float, float]] = [
    (0.3, 90.0),
    (0.8, 75.0),
    (1.5, 63.0),
    (2.5, 45.0),
    (4.0, 25.0),
    (6.0, 10.0),
]

CODECS = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]
BUTTER_DIR = Path("/mnt/v/zen/picker-training/2026-05-19/butter")
SSIM2_PATH = Path(
    "/mnt/v/zen/zensim-training/canonical-2026-05-18/scores/ssim2_imazen.parquet"
)
CVVDP_PATH = Path(
    "/mnt/v/zen/zensim-training/canonical-2026-05-18/scores/cvvdp_imazen_v0_0_1.parquet"
)
# cvvdp_log_norm normalization basis: use the trainer's reference
# (safesyn corpus min/max). This matches what the trainer's
# mix_cv40_iw60 target column was built from in
# canonical-2026-05-18/train/safesyn.parquet — so per-band cvvdp
# medians get reported in the same units the trainer's target lives.
SAFESYN_PATH = Path(
    "/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet"
)
DEFAULT_OUT = Path(
    "/mnt/v/zen/zensim-training/2026-05-19-empirical-band-anchors/anchors_empirical_372col.parquet"
)
DEFAULT_COMPARISON = Path(
    "/home/lilith/work/zen/zensimx/benchmarks/v_tuner_v7_anchor_target_comparison_2026-05-19.md"
)


def cvvdp_to_score_0_100(cv: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Match the canonical training corpus' cvvdp_log_norm transform.

    raw_log = -log(10 - cv + 1e-6); then min-max scale to [0, 100].
    """
    raw_log = -np.log(10.0 - cv + 1e-6)
    return (raw_log - lo) / (hi - lo) * 100.0


def build_score_lookups() -> tuple[
    dict[tuple[str, str, int], float],
    dict[tuple[str, str, int], float],
    float,
    float,
]:
    """Build (basename, codec, q) → median ssim2 / cvvdp lookups.

    Multiple knob_tuple_json entries per (image, codec, q) → take the
    median. Returns the lookups plus the cvvdp_log min/max so we can
    apply the same normalization to anchor medians.
    """
    print("=== building score lookups ===")
    print(f"loading ssim2 from {SSIM2_PATH}")
    st = pq.read_table(SSIM2_PATH).to_pandas()
    st["basename"] = st["image_path"].map(os.path.basename)
    ssim2_lookup: dict[tuple[str, str, int], float] = {}
    for (bn, cd, q), grp in st.groupby(["basename", "codec", "q"]):
        ssim2_lookup[(bn, cd, int(q))] = float(grp["ssim2_gpu"].median())
    print(f"  {len(ssim2_lookup)} ssim2 (basename, codec, q) keys")

    print(f"loading cvvdp from {CVVDP_PATH}")
    ct = pq.read_table(CVVDP_PATH).to_pandas()
    ct["basename"] = ct["image_path"].map(os.path.basename)
    # Compute cvvdp_log min/max from the SAFESYN training corpus so
    # per-band cvvdp medians get reported in the same units the
    # trainer's mix_cv40_iw60 target lives in.
    print(f"loading safesyn cvvdp range from {SAFESYN_PATH}")
    safesyn_cv = pq.read_table(SAFESYN_PATH, columns=["cvvdp_score"]).column("cvvdp_score").to_pylist()
    safesyn_cv = np.array([v for v in safesyn_cv if v is not None and not np.isnan(v)])
    safesyn_raw_log = -np.log(10.0 - safesyn_cv + 1e-6)
    lo = float(np.nanmin(safesyn_raw_log))
    hi = float(np.nanmax(safesyn_raw_log))
    print(f"  cvvdp_log range from safesyn: [{lo:.4f}, {hi:.4f}] (n={len(safesyn_cv)})")
    cvvdp_lookup: dict[tuple[str, str, int], float] = {}
    for (bn, cd, q), grp in ct.groupby(["basename", "codec", "q"]):
        cvvdp_lookup[(bn, cd, int(q))] = float(grp["cvvdp_imazen_v0_0_1"].median())
    print(f"  {len(cvvdp_lookup)} cvvdp (basename, codec, q) keys")
    return ssim2_lookup, cvvdp_lookup, lo, hi


def compute_band_medians(
    ssim2_lookup: dict, cvvdp_lookup: dict, cvvdp_lo: float, cvvdp_hi: float,
    max_distance: float
) -> tuple[
    dict[tuple[str, float], dict[str, float]],
    dict[tuple[str, float], dict[str, int]],
]:
    """For each (codec, band) compute the empirical ssim2 and cvvdp medians
    from the joined (butter, score) corpus, plus per-band cell-counts.
    """
    print("=== computing per-(codec, band) empirical medians ===")
    medians: dict[tuple[str, float], dict[str, float]] = {}
    counts: dict[tuple[str, float], dict[str, int]] = {}

    for codec in CODECS:
        path = BUTTER_DIR / f"{codec}.parquet"
        if not path.exists():
            continue
        df = pq.read_table(path, columns=["ref_basename", "codec", "q", "butter_pnorm3"]).to_pandas()
        print(f"{codec}: {len(df)} butter rows; joining scores")

        for butter_target, _v6_target in ANCHOR_BANDS:
            cell_ssim2: list[float] = []
            cell_cvvdp_score: list[float] = []
            cell_seen = 0
            cell_no_score = 0
            cell_filtered = 0

            mask = (df["butter_pnorm3"] - butter_target).abs() <= max_distance
            sub = df[mask]
            cell_seen = len(sub)
            for _, row in sub.iterrows():
                key = (str(row["ref_basename"]), codec, int(row["q"]))
                s = ssim2_lookup.get(key)
                c = cvvdp_lookup.get(key)
                if s is None and c is None:
                    cell_no_score += 1
                    continue
                if s is not None:
                    cell_ssim2.append(s)
                if c is not None:
                    cell_cvvdp_score.append(c)

            # Drop the band if we have no joined data for either signal.
            if not cell_ssim2 and not cell_cvvdp_score:
                cell_filtered = cell_seen
                medians[(codec, butter_target)] = {
                    "ssim2_median": float("nan"),
                    "cvvdp_score_median": float("nan"),
                    "cvvdp_norm_median": float("nan"),
                    "joint_target": float("nan"),
                }
                counts[(codec, butter_target)] = {
                    "butter_rows_in_band": cell_seen,
                    "ssim2_n": 0,
                    "cvvdp_n": 0,
                    "no_score_rows": cell_no_score,
                }
                continue

            ssim2_median = float(np.median(cell_ssim2)) if cell_ssim2 else float("nan")
            cvvdp_score_median = (
                float(np.median(cell_cvvdp_score)) if cell_cvvdp_score else float("nan")
            )
            cvvdp_norm_median = (
                float(cvvdp_to_score_0_100(np.array([cvvdp_score_median]), cvvdp_lo, cvvdp_hi)[0])
                if not np.isnan(cvvdp_score_median)
                else float("nan")
            )
            # target_score is ssim2-primary: ssim2's 0-100 range is what
            # the V6 anchor's rule-of-thumb numbers (90/75/63/45/25/10)
            # are calibrated to, and PJND lands at ~63 in ssim2 (CID22
            # paper). cvvdp_log_norm in safesyn-corpus units lives in
            # ~0-60 (compression-product distortion regime is 10-35) —
            # mixing it 50/50 with ssim2 systematically biases the
            # anchor target down. Use ssim2 alone as the anchor target;
            # report cvvdp alongside in the comparison table.
            joint = ssim2_median if not np.isnan(ssim2_median) else cvvdp_norm_median

            medians[(codec, butter_target)] = {
                "ssim2_median": ssim2_median,
                "cvvdp_score_median": cvvdp_score_median,
                "cvvdp_norm_median": cvvdp_norm_median,
                "joint_target": joint,
            }
            counts[(codec, butter_target)] = {
                "butter_rows_in_band": cell_seen,
                "ssim2_n": len(cell_ssim2),
                "cvvdp_n": len(cell_cvvdp_score),
                "no_score_rows": cell_no_score,
            }
            print(
                f"  band={butter_target:>3.1f}: "
                f"n_butter={cell_seen} ssim2_n={len(cell_ssim2)} cvvdp_n={len(cell_cvvdp_score)} "
                f"ssim2_med={ssim2_median:.2f} cvvdp_med={cvvdp_score_median:.4f} "
                f"cvvdp_norm={cvvdp_norm_median:.2f} joint={joint:.2f}"
            )
    return medians, counts


def aggregate_band_targets(
    medians: dict[tuple[str, float], dict[str, float]],
) -> dict[float, dict[str, float]]:
    """Per-band aggregate (median over codecs that have data) for the
    comparison table."""
    out: dict[float, dict[str, float]] = {}
    for butter_target, _ in ANCHOR_BANDS:
        s = [
            medians[(c, butter_target)]["ssim2_median"]
            for c in CODECS
            if (c, butter_target) in medians
            and not np.isnan(medians[(c, butter_target)]["ssim2_median"])
        ]
        cn = [
            medians[(c, butter_target)]["cvvdp_norm_median"]
            for c in CODECS
            if (c, butter_target) in medians
            and not np.isnan(medians[(c, butter_target)]["cvvdp_norm_median"])
        ]
        joint = [
            medians[(c, butter_target)]["joint_target"]
            for c in CODECS
            if (c, butter_target) in medians
            and not np.isnan(medians[(c, butter_target)]["joint_target"])
        ]
        out[butter_target] = {
            "ssim2_agg": float(np.median(s)) if s else float("nan"),
            "cvvdp_norm_agg": float(np.median(cn)) if cn else float("nan"),
            "joint_agg": float(np.median(joint)) if joint else float("nan"),
        }
    return out


def build_anchor_rows(
    medians: dict[tuple[str, float], dict[str, float]],
    max_distance: float,
) -> list[dict]:
    """Build per-(image, codec, band) anchor rows with empirical
    per-(codec, band) joint target."""
    print("=== building anchor rows ===")
    rows: list[dict] = []
    feature_cols = [f"f{i}" for i in range(372)]
    filter_stats: dict[tuple[str, float], dict[str, int]] = {}

    for codec in CODECS:
        path = BUTTER_DIR / f"{codec}.parquet"
        if not path.exists():
            continue
        df = pq.read_table(path).to_pandas()

        for source, group in df.groupby("ref_basename"):
            for butter_target, _v6_target in ANCHOR_BANDS:
                key = (codec, butter_target)
                slot = filter_stats.setdefault(key, {"emitted": 0, "filtered": 0})

                med = medians.get(key)
                if med is None or np.isnan(med.get("joint_target", float("nan"))):
                    slot["filtered"] += 1
                    continue
                target_score = med["joint_target"]

                distances = (group["butter_pnorm3"] - butter_target).abs()
                idx = distances.idxmin()
                best = group.loc[idx]
                best_distance = float(distances.loc[idx])
                if best_distance > max_distance:
                    slot["filtered"] += 1
                    continue
                slot["emitted"] += 1

                row = {
                    "ref_basename": str(source),
                    "anchor_source": f"empirical_{codec}_b{butter_target}_s{target_score:.1f}",
                    "human_score": float(target_score),
                    "anchor_weight": 1.0,
                    "q": int(best["q"]),
                    "butter_pnorm3": float(best["butter_pnorm3"]),
                    "butter_target": float(butter_target),
                    "target_score": float(target_score),
                    "codec": codec,
                }
                for col in feature_cols:
                    if col in best.index:
                        val = best[col]
                        row[col] = float(val) if val is not None else 0.0
                    else:
                        row[col] = 0.0
                rows.append(row)

    print(f"total anchor rows: {len(rows)}")
    print("per-(codec, band) row counts:")
    print("  codec       butter_target  emitted  filtered  filter%")
    for codec in CODECS:
        for butter_target, _ in ANCHOR_BANDS:
            slot = filter_stats.get((codec, butter_target))
            if slot is None:
                continue
            tot = slot["emitted"] + slot["filtered"]
            pct = 100.0 * slot["filtered"] / tot if tot else 0.0
            print(
                f"  {codec:10s}  {butter_target:13.2f}  {slot['emitted']:7d}  {slot['filtered']:8d}  {pct:6.2f}%"
            )
    return rows


def write_parquet(rows: list[dict], out: Path) -> None:
    if not rows:
        raise SystemExit("no anchor rows built")
    feature_cols = [f"f{i}" for i in range(372)]
    cols = {
        "ref_basename": pa.array([r["ref_basename"] for r in rows], type=pa.string()),
        "anchor_source": pa.array([r["anchor_source"] for r in rows], type=pa.string()),
        "human_score": pa.array([r["human_score"] for r in rows], type=pa.float64()),
        "anchor_weight": pa.array([r["anchor_weight"] for r in rows], type=pa.float64()),
        "q": pa.array([r["q"] for r in rows], type=pa.int64()),
        "butter_pnorm3": pa.array([r["butter_pnorm3"] for r in rows], type=pa.float64()),
        "butter_target": pa.array([r["butter_target"] for r in rows], type=pa.float64()),
        "target_score": pa.array([r["target_score"] for r in rows], type=pa.float64()),
        "codec": pa.array([r["codec"] for r in rows], type=pa.string()),
    }
    for col in feature_cols:
        cols[col] = pa.array([r[col] for r in rows], type=pa.float32())
    tbl = pa.table(cols)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out, compression="zstd", compression_level=15)
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, "
        f"{tbl.num_rows} rows x {tbl.num_columns} cols)"
    )


def write_comparison_md(
    medians: dict[tuple[str, float], dict[str, float]],
    counts: dict[tuple[str, float], dict[str, int]],
    aggregate: dict[float, dict[str, float]],
    out: Path,
) -> None:
    lines: list[str] = []
    lines.append("# V6 rule-of-thumb vs V7 empirical anchor targets (2026-05-19)")
    lines.append("")
    lines.append(
        "V6 (`PreviewV0_5TunerV2`, commit `1dd61fc`) shipped with hand-set "
        "anchor targets per butter band. Only the 1.5 anchor (CID22 PJND, 63) "
        "had empirical grounding. V7 replaces the other 5 with medians from "
        "the canonical ssim2 + cvvdp score parquets at "
        "`/mnt/v/zen/zensim-training/canonical-2026-05-18/scores/`."
    )
    lines.append("")
    lines.append(
        "Per-band aggregation: union all (codec, image, q) within "
        "`butter_pnorm3 ∈ [band ± 0.5]`, lookup ssim2_gpu and "
        "cvvdp_imazen_v0_0_1 by (basename, codec, q), normalize cvvdp via "
        "`-log(10 - cvvdp)` and min-max to [0, 100] using the score parquet's "
        "global min/max, then per-(codec, band) median."
    )
    lines.append("")
    lines.append("## Aggregate per-band targets (median across codecs)")
    lines.append("")
    lines.append(
        "`target_score` (V7) = median ssim2 alone, since the V6 "
        "rule-of-thumb numbers are calibrated to ssim2's 0-100 range and "
        "PJND lands at ~63 in ssim2 per CID22 paper. cvvdp_log_norm "
        "(safesyn-corpus normed) shown as parallel signal — it lives in "
        "~10-35 across the compression-product distortion regime, "
        "structurally below ssim2."
    )
    lines.append("")
    lines.append("| butter_pnorm3 | V6 rule | empirical ssim2 (used as V7 target) | empirical cvvdp (norm) | Δ (ssim2 − V6) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for butter_target, v6_score in ANCHOR_BANDS:
        row = aggregate.get(butter_target, {})
        s = row.get("ssim2_agg", float("nan"))
        c = row.get("cvvdp_norm_agg", float("nan"))
        delta = s - v6_score if not np.isnan(s) else float("nan")
        lines.append(
            f"| {butter_target} | {v6_score:.1f} | "
            f"{s:.2f} | {c:.2f} | {delta:+.2f} |"
        )
    lines.append("")
    lines.append("## Per-codec per-band empirical medians")
    lines.append("")
    for codec in CODECS:
        lines.append(f"### {codec}")
        lines.append("")
        lines.append(
            "| butter_pnorm3 | V6 rule | ssim2 median | ssim2 n | cvvdp median (raw) | "
            "cvvdp median (norm) | cvvdp n | joint target |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for butter_target, v6_score in ANCHOR_BANDS:
            med = medians.get((codec, butter_target))
            cnt = counts.get((codec, butter_target))
            if med is None or cnt is None:
                lines.append(
                    f"| {butter_target} | {v6_score:.1f} | — | 0 | — | — | 0 | — |"
                )
                continue
            lines.append(
                f"| {butter_target} | {v6_score:.1f} | "
                f"{med['ssim2_median']:.2f} | {cnt['ssim2_n']} | "
                f"{med['cvvdp_score_median']:.4f} | "
                f"{med['cvvdp_norm_median']:.2f} | {cnt['cvvdp_n']} | "
                f"{med['joint_target']:.2f} |"
            )
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote comparison table to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--comparison-md", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--max-distance", type=float, default=0.5)
    args = parser.parse_args()

    ssim2_lookup, cvvdp_lookup, cvvdp_lo, cvvdp_hi = build_score_lookups()
    medians, counts = compute_band_medians(
        ssim2_lookup, cvvdp_lookup, cvvdp_lo, cvvdp_hi, args.max_distance
    )
    aggregate = aggregate_band_targets(medians)
    rows = build_anchor_rows(medians, args.max_distance)
    write_parquet(rows, args.out)
    write_comparison_md(medians, counts, aggregate, args.comparison_md)


if __name__ == "__main__":
    main()
