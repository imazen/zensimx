#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V10 anchor parquet builder (2026-05-20).

V10 design (user direction 2026-05-20): reallocate the zensim score-space
so lossless = 100, JND = 80, JOD = 50, borderline (q=0 worst codec) = 0,
and below 0 = pathological. This replaces V9's table (JND=60 / JOD=30 /
lossless=100 / clamp at 0).

V10 anchor band table (11 bands, [0, 100] range with unclamped extrapolation):

| butter_pnorm3 | target_score | semantic                          |
|---:|---:|---|
| 0.05  | 100 | mathematically lossless             |
| 0.30  |  95 | near-lossless                       |
| 0.60  |  90 | visually identical                  |
| 1.50  |  80 | JND (PJND threshold)                |
| 2.50  |  65 | mildly noticeable                   |
| 4.00  |  50 | JOD (just objectionable)            |
| 5.50  |  35 | 3x-DPI resize-out — usable at scale |
| 7.00  |  20 | clear artifacts even at scale       |
| 9.00  |  10 | very degraded                       |
| 12.00 |   0 | borderline unacceptable             |
| >12   |  <0 | pathological (linear extrapolation) |

Adaptations for the available butter parquets (which cover q=5..95 and
don't reach butter=12.0):

- max_distance widened to 0.5 (from V9's 0.4) because V10 has 11 bands
  spanning the same butter range so the per-band tolerance budget is
  proportionally smaller.
- For the **lossless / score=100 band (butter=0.05)**: explicit zenjxl
  q=95 rows added (butter ~= 0.005 for zenjxl at q=95). Other codecs'
  min-butter rows (closest to 0.05) keep the normal band assignment.
- For the **worst-q floor / score=0 band (butter=12.0)**: explicit q=5
  rows with butter >= 6.0 added. These widen the worst-floor anchor
  pool. The unclamped V10 spline / extrapolation lets the predicted
  score go below 0 for the most extreme cases.

Output: `/mnt/v/zen/zensim-training/2026-05-20-v10-anchors/anchors_v10_372col.parquet`
Sibling comparison table at the --comparison-md path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# (butter_pnorm3_target, V10_target_score). 11 bands, denser low-q.
ANCHOR_BANDS_V10: list[tuple[float, float]] = [
    (0.05, 100.0),   # lossless
    (0.30,  95.0),   # near-lossless
    (0.60,  90.0),   # visually identical
    (1.50,  80.0),   # JND
    (2.50,  65.0),
    (4.00,  50.0),   # JOD
    (5.50,  35.0),
    (7.00,  20.0),
    (9.00,  10.0),
    (12.00,  0.0),   # worst-q floor / borderline unacceptable
]

# V9 (8 bands) for comparison.
ANCHOR_BANDS_V9: list[tuple[float, float]] = [
    (0.05, 100.0),
    (0.30,  90.0),
    (0.60,  80.0),
    (1.50,  60.0),
    (2.50,  50.0),
    (4.00,  30.0),
    (7.00,  10.0),
    (12.00,  0.0),
]

CODECS = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]
BUTTER_DIR = Path("/mnt/v/zen/picker-training/2026-05-19/butter")
DEFAULT_OUT = Path(
    "/mnt/v/zen/zensim-training/2026-05-20-v10-anchors/anchors_v10_372col.parquet"
)

# Empirical max butter per codec (from butter parquets):
#   zenjpeg max=11.4, zenwebp max=5.8, zenavif max=8.7, zenjxl max=9.3
# None reach butter=12.0, so the floor band needs widened tolerance and
# explicit q=5 fallback rows.
WORST_FLOOR_BUTTER_MIN = 6.0
WORST_FLOOR_TARGET_SCORE = 0.0
LOSSLESS_BUTTER_MAX = 0.10
LOSSLESS_TARGET_SCORE = 100.0


def build_anchor_rows(
    max_distance: float,
) -> tuple[list[dict], dict[tuple[str, float], dict[str, int]]]:
    """Build per-(image, codec, band) anchor rows with V10 fixed targets.

    Returns (rows, per-(codec, band) emit stats).
    """
    print("=== building anchor rows ===")
    rows: list[dict] = []
    feature_cols = [f"f{i}" for i in range(372)]
    filter_stats: dict[tuple[str, float], dict[str, int]] = {}

    for codec in CODECS:
        path = BUTTER_DIR / f"{codec}.parquet"
        if not path.exists():
            print(f"warn: {path} missing; skipping {codec}")
            continue
        df = pq.read_table(path).to_pandas()
        print(f"{codec}: {len(df)} butter rows; selecting per-(source, band)")

        for source, group in df.groupby("ref_basename"):
            for butter_target, target_score in ANCHOR_BANDS_V10:
                key = (codec, butter_target)
                slot = filter_stats.setdefault(
                    key, {"emitted": 0, "filtered": 0, "tot_distance": 0.0}
                )

                distances = (group["butter_pnorm3"] - butter_target).abs()
                idx = distances.idxmin()
                best = group.loc[idx]
                best_distance = float(distances.loc[idx])
                if best_distance > max_distance:
                    slot["filtered"] += 1
                    continue
                slot["emitted"] += 1
                slot["tot_distance"] += best_distance

                row = {
                    "ref_basename": str(source),
                    "anchor_source": (
                        f"v10_{codec}_b{butter_target}_s{target_score:.1f}"
                    ),
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

            # === explicit "worst-q floor" rows: butter >= 6.0 → score=0 ===
            heavy = group[group["butter_pnorm3"] >= WORST_FLOOR_BUTTER_MIN]
            for idx, best in heavy.iterrows():
                slot = filter_stats.setdefault(
                    (codec, -1.0),
                    {"emitted": 0, "filtered": 0, "tot_distance": 0.0},
                )
                slot["emitted"] += 1
                row = {
                    "ref_basename": str(source),
                    "anchor_source": f"v10_{codec}_worstfloor_q{int(best['q'])}",
                    "human_score": float(WORST_FLOOR_TARGET_SCORE),
                    "anchor_weight": 1.0,
                    "q": int(best["q"]),
                    "butter_pnorm3": float(best["butter_pnorm3"]),
                    "butter_target": -1.0,
                    "target_score": float(WORST_FLOOR_TARGET_SCORE),
                    "codec": codec,
                }
                for col in feature_cols:
                    if col in best.index:
                        val = best[col]
                        row[col] = float(val) if val is not None else 0.0
                    else:
                        row[col] = 0.0
                rows.append(row)

            # === explicit "lossless" rows: butter <= 0.10 → score=100 ===
            lossless = group[group["butter_pnorm3"] <= LOSSLESS_BUTTER_MAX]
            for idx, best in lossless.iterrows():
                slot = filter_stats.setdefault(
                    (codec, -2.0),
                    {"emitted": 0, "filtered": 0, "tot_distance": 0.0},
                )
                slot["emitted"] += 1
                row = {
                    "ref_basename": str(source),
                    "anchor_source": f"v10_{codec}_lossless_q{int(best['q'])}",
                    "human_score": float(LOSSLESS_TARGET_SCORE),
                    "anchor_weight": 1.0,
                    "q": int(best["q"]),
                    "butter_pnorm3": float(best["butter_pnorm3"]),
                    "butter_target": -2.0,
                    "target_score": float(LOSSLESS_TARGET_SCORE),
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
    print(
        f"  {'codec':10s}  {'band':>10s}  {'target':>6s}  "
        f"{'emitted':>7s}  {'filtered':>8s}  {'mean_d':>6s}  filter%"
    )
    band_keys = list(ANCHOR_BANDS_V10) + [(-1.0, 0.0), (-2.0, 100.0)]
    for codec in CODECS:
        for butter_target, target_score in band_keys:
            slot = filter_stats.get((codec, butter_target))
            if slot is None:
                continue
            tot = slot["emitted"] + slot["filtered"]
            pct = 100.0 * slot["filtered"] / tot if tot else 0.0
            mean_d = (
                slot["tot_distance"] / slot["emitted"] if slot["emitted"] else 0.0
            )
            band_label = (
                "worstfloor" if butter_target == -1.0
                else "lossless" if butter_target == -2.0
                else f"{butter_target:5.2f}"
            )
            print(
                f"  {codec:10s}  {band_label:>10s}  {target_score:6.1f}  "
                f"{slot['emitted']:7d}  {slot['filtered']:8d}  "
                f"{mean_d:6.3f}  {pct:6.2f}%"
            )
    return rows, filter_stats


def write_parquet(rows: list[dict], out: Path) -> None:
    if not rows:
        raise SystemExit("no anchor rows built")
    feature_cols = [f"f{i}" for i in range(372)]
    cols = {
        "ref_basename": pa.array(
            [r["ref_basename"] for r in rows], type=pa.string()
        ),
        "anchor_source": pa.array(
            [r["anchor_source"] for r in rows], type=pa.string()
        ),
        "human_score": pa.array(
            [r["human_score"] for r in rows], type=pa.float64()
        ),
        "anchor_weight": pa.array(
            [r["anchor_weight"] for r in rows], type=pa.float64()
        ),
        "q": pa.array([r["q"] for r in rows], type=pa.int64()),
        "butter_pnorm3": pa.array(
            [r["butter_pnorm3"] for r in rows], type=pa.float64()
        ),
        "butter_target": pa.array(
            [r["butter_target"] for r in rows], type=pa.float64()
        ),
        "target_score": pa.array(
            [r["target_score"] for r in rows], type=pa.float64()
        ),
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
    filter_stats: dict[tuple[str, float], dict[str, int]],
    out: Path,
) -> None:
    """V9 vs V10 anchor-target comparison."""
    lines: list[str] = []
    lines.append(
        "# V9 vs V10 anchor target comparison (EXP-CROSS-CODEC-V10, "
        "2026-05-20)"
    )
    lines.append("")
    lines.append(
        "V10 reallocates the zensim score-space per user direction "
        "2026-05-20. Lossless = 100. JND = 80 (was 60 in V9). JOD = 50 "
        "(was 30 in V9). Borderline (q=0 worst codec) = 0. Below 0 = "
        "pathological / unreasonable (no clamp; linear extrapolation)."
    )
    lines.append("")
    lines.append("## Per-band target table")
    lines.append("")
    lines.append(
        "| butter_pnorm3 | V9 target | **V10 target** | semantic |"
    )
    lines.append("|---:|---:|---:|---|")
    v9_dict = dict(ANCHOR_BANDS_V9)
    v10_dict = dict(ANCHOR_BANDS_V10)
    all_bands = sorted(set(v9_dict.keys()) | set(v10_dict.keys()))
    for band in all_bands:
        v9 = v9_dict.get(band)
        v10 = v10_dict.get(band)
        v9_str = f"{v9:.1f}" if v9 is not None else "—"
        v10_str = f"**{v10:.1f}**" if v10 is not None else "—"
        if band == 1.5:
            sem = "**JND** (PJND threshold)"
        elif band == 4.0:
            sem = "**JOD** (just objectionable)"
        elif band == 0.05:
            sem = "lossless / q=95-100 best codec"
        elif band == 0.3:
            sem = "near-lossless"
        elif band == 0.6:
            sem = "visually identical"
        elif band == 2.5:
            sem = "mildly noticeable"
        elif band == 5.5:
            sem = "3x-DPI resize-out — usable at scale"
        elif band == 7.0:
            sem = "clear artifacts even at scale"
        elif band == 9.0:
            sem = "very degraded"
        elif band == 12.0:
            sem = "worst-q floor / borderline unacceptable"
        else:
            sem = "(intermediate)"
        lines.append(f"| {band} | {v9_str} | {v10_str} | {sem} |")
    lines.append("")
    lines.append("## V10 design rationale")
    lines.append("")
    lines.append(
        "- **Lossless = 100, JND = 80, JOD = 50, q=0-floor = 0.** "
        "Wider perceptibility band (50 score units between JOD and "
        "JND) gives the user-facing dial more resolution where "
        "compression product decisions live."
    )
    lines.append(
        "- **Below 0 = pathological.** V10 removes the [0, 100] hard "
        "clamp in apply_mlp_scoring AND in the bake-aware tools' "
        "default post mode. The PCHIP spline extrapolates linearly "
        "below xs[0] / above xs[-1] using the endpoint slope, so the "
        "worst codec at q=0 (butter >> 12) maps to a negative score. "
        "This signals 'unreasonable distortion' rather than collapsing "
        "to a tie block."
    )
    lines.append(
        "- **11 bands instead of V9's 8.** Denser sampling of the "
        "perceptibility curve (5.5 and 9.0 added). Tradeoff: max_distance "
        "widened to 0.5 so each per-band anchor pool stays similar in "
        "size."
    )
    lines.append("")
    lines.append("## Realizability note")
    lines.append("")
    lines.append(
        "Butter parquets (`/mnt/v/zen/picker-training/2026-05-19/"
        "butter/<codec>.parquet`) only cover q=5..95 and don't "
        "reach butter=12.0 (max butter per codec: zenjpeg=11.4, "
        "zenwebp=5.8, zenavif=8.7, zenjxl=9.3)."
    )
    lines.append("")
    lines.append("V10 adapts by:")
    lines.append("")
    lines.append(
        "- Widening `max_distance` from V9's 0.4 -> 0.5 to allow "
        "the 12.0 band to claim the closest-available high-butter row."
    )
    lines.append(
        "- Adding explicit \"worstfloor\" anchor rows: every row "
        f"with butter >= {WORST_FLOOR_BUTTER_MIN} from the q=5..95 "
        "butter parquets, target_score=0."
    )
    lines.append(
        "- Adding explicit \"lossless\" anchor rows: every row "
        f"with butter <= {LOSSLESS_BUTTER_MAX}, target_score=100."
    )
    lines.append("")
    lines.append(
        "The post-network PCHIP spline calibration applied AFTER "
        "training, with unclamped linear extrapolation, lets the "
        "spline output flow through to the final score uninhibited. "
        "Reasonable codec output lands in [0, 100]; pathological "
        "output extrapolates negative."
    )
    lines.append("")
    lines.append("## Per-codec row counts (V10 build)")
    lines.append("")
    lines.append(
        "| codec | band | target_score | emitted | filtered | filter% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    band_keys = list(ANCHOR_BANDS_V10) + [(-1.0, 0.0), (-2.0, 100.0)]
    for codec in CODECS:
        for butter_target, target_score in band_keys:
            slot = filter_stats.get((codec, butter_target))
            if slot is None:
                continue
            tot = slot["emitted"] + slot["filtered"]
            pct = 100.0 * slot["filtered"] / tot if tot else 0.0
            band_label = (
                "worstfloor" if butter_target == -1.0
                else "lossless" if butter_target == -2.0
                else f"{butter_target}"
            )
            lines.append(
                f"| {codec} | {band_label} | {target_score:.1f} | "
                f"{slot['emitted']} | {slot['filtered']} | {pct:.2f}% |"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote comparison table to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--comparison-md",
        type=Path,
        default=Path(
            "/home/lilith/work/zen/zensimx/benchmarks/"
            "v10_anchor_design_2026-05-20.md"
        ),
    )
    parser.add_argument("--max-distance", type=float, default=0.5)
    args = parser.parse_args()

    rows, stats = build_anchor_rows(args.max_distance)
    write_parquet(rows, args.out)
    write_comparison_md(stats, args.comparison_md)


if __name__ == "__main__":
    main()
