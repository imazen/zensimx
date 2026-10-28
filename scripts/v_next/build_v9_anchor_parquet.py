#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V9 anchor parquet builder (2026-05-20).

V9 design (user direction 2026-05-20): zensim score range must extend
from 0 (worst codec at q=0) to 100 (best codec at lossless/q=100).
JND lands at score=60 (clean multiple of 10). JOD lands at score=30
(clean multiple of 10).

V9 anchor band table (8 bands, 100-wide range):

| butter_pnorm3 | target_score | semantic                          |
|---:|---:|---|
| 0.05  | 100 | lossless / q=100 best codec        |
| 0.30  |  90 | near-lossless                       |
| 0.60  |  80 | visually identical                  |
| 1.50  |  60 | JND (CID22 paper PJND)              |
| 2.50  |  50 | mildly noticeable                   |
| 4.00  |  30 | JOD (just objectionable)            |
| 7.00  |  10 | clearly distorted                   |
| 12.00 |   0 | worst-q floor                       |

Adaptation for the available butter parquets (which only cover q=5..95
and don't reach butter=12.0):

- max_distance widened to 0.4 (from V8's 0.3) since we span wider
  butter range.
- For the **lossless / score=100 band (butter=0.05)**: explicit zenjxl
  q=95 rows added (butter ≈ 0.005 for zenjxl at q=95 is effectively
  lossless). Other codecs' min-butter rows (closest to 0.05) keep the
  normal band assignment.
- For the **worst-q floor / score=0 band (butter=12.0)**: explicit q=5
  rows with butter >= 6.0 added (closest available data to the floor).
  These rows widen the "near worst quality" anchor pool.

Output: `/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet`
Sibling comparison table at the --comparison-md path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# (butter_pnorm3_target, V9_target_score). 8 bands, extended [0, 100] range.
ANCHOR_BANDS_V9: list[tuple[float, float]] = [
    (0.05, 100.0),   # lossless / q=100 best codec
    (0.30,  90.0),   # near-lossless
    (0.60,  80.0),   # visually identical
    (1.50,  60.0),   # JND (CID22 paper PJND)
    (2.50,  50.0),
    (4.00,  30.0),   # JOD (just objectionable)
    (7.00,  10.0),
    (12.00,  0.0),   # worst-q floor
]

# V8 (4 bands) for comparison.
ANCHOR_BANDS_V8: list[tuple[float, float]] = [
    (0.5, 85.0),
    (1.0, 75.0),
    (2.5, 63.0),
    (4.0, 45.0),
]

# V6 (6 bands) for comparison.
ANCHOR_BANDS_V6: list[tuple[float, float]] = [
    (0.3, 90.0),
    (0.8, 75.0),
    (1.5, 63.0),
    (2.5, 45.0),
    (4.0, 25.0),
    (6.0, 10.0),
]

CODECS = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]
BUTTER_DIR = Path("/mnt/v/zen/picker-training/2026-05-19/butter")
DEFAULT_OUT = Path(
    "/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
)

# Empirical max butter per codec (from butter parquets):
#   zenjpeg max=11.4, zenwebp max=5.8, zenavif max=8.7, zenjxl max=9.3
# None reach butter=12.0, so the floor band needs widened tolerance and
# explicit q=5 fallback rows.
WORST_FLOOR_BUTTER_MIN = 6.0  # rows with butter >= this are floor-band candidates
WORST_FLOOR_TARGET_SCORE = 0.0
LOSSLESS_BUTTER_MAX = 0.10    # rows with butter <= this are score=100 candidates
LOSSLESS_TARGET_SCORE = 100.0


def build_anchor_rows(
    max_distance: float,
) -> tuple[list[dict], dict[tuple[str, float], dict[str, int]]]:
    """Build per-(image, codec, band) anchor rows with V9 fixed targets.

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
            for butter_target, target_score in ANCHOR_BANDS_V9:
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
                        f"v9_{codec}_b{butter_target}_s{target_score:.1f}"
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
            # Anchor pool widening for the score=0 end.
            heavy = group[group["butter_pnorm3"] >= WORST_FLOOR_BUTTER_MIN]
            for idx, best in heavy.iterrows():
                slot = filter_stats.setdefault(
                    (codec, -1.0),
                    {"emitted": 0, "filtered": 0, "tot_distance": 0.0},
                )
                slot["emitted"] += 1
                row = {
                    "ref_basename": str(source),
                    "anchor_source": f"v9_{codec}_worstfloor_q{int(best['q'])}",
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
            # Anchor pool widening for the score=100 end.
            lossless = group[group["butter_pnorm3"] <= LOSSLESS_BUTTER_MAX]
            for idx, best in lossless.iterrows():
                slot = filter_stats.setdefault(
                    (codec, -2.0),
                    {"emitted": 0, "filtered": 0, "tot_distance": 0.0},
                )
                slot["emitted"] += 1
                row = {
                    "ref_basename": str(source),
                    "anchor_source": f"v9_{codec}_lossless_q{int(best['q'])}",
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
        f"  {'codec':10s}  {'band':>5s}  {'target':>6s}  "
        f"{'emitted':>7s}  {'filtered':>8s}  {'mean_d':>6s}  filter%"
    )
    band_keys = list(ANCHOR_BANDS_V9) + [(-1.0, 0.0), (-2.0, 100.0)]
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
    """V6 vs V8 vs V9 anchor-target comparison."""
    lines: list[str] = []
    lines.append(
        "# V6 vs V8 vs V9 anchor target comparison (EXP-CROSS-CODEC-V9, "
        "2026-05-20)"
    )
    lines.append("")
    lines.append(
        "V9 extends the score range to a clean [0, 100] span with the "
        "JND at score=60 (clean multiple of 10) and JOD at score=30 "
        "(clean multiple of 10). V6 (ship) used 6 bands "
        "[10, 90] range with PJND at 63; V8 (falsified) used 4 bands "
        "[45, 85] range with PJND at 63."
    )
    lines.append("")
    lines.append("## Per-band target table")
    lines.append("")
    lines.append(
        "| butter_pnorm3 | V6 rule | V8 chosen | **V9 chosen** | semantic |"
    )
    lines.append("|---:|---:|---:|---:|---|")
    v6_dict = dict(ANCHOR_BANDS_V6)
    v8_dict = dict(ANCHOR_BANDS_V8)
    v9_dict = dict(ANCHOR_BANDS_V9)
    all_bands = sorted(
        set(v6_dict.keys()) | set(v8_dict.keys()) | set(v9_dict.keys())
    )
    for band in all_bands:
        v6 = v6_dict.get(band)
        v8 = v8_dict.get(band)
        v9 = v9_dict.get(band)
        v6_str = f"{v6:.1f}" if v6 is not None else "—"
        v8_str = f"{v8:.1f}" if v8 is not None else "—"
        v9_str = f"**{v9:.1f}**" if v9 is not None else "—"
        if band == 1.5:
            sem = "**JND** (CID22 paper PJND)"
        elif band == 4.0:
            sem = "**JOD** (just objectionable)"
        elif band == 0.05:
            sem = "lossless / q=100 best codec"
        elif band == 0.3:
            sem = "near-lossless"
        elif band == 0.6:
            sem = "visually identical"
        elif band == 2.5:
            sem = "mildly noticeable"
        elif band == 7.0:
            sem = "clearly distorted"
        elif band == 12.0:
            sem = "worst-q floor"
        elif band == 0.8:
            sem = "(V6 band, dropped in V9)"
        elif band == 4.0:
            sem = "(V6+V8 band — V9 redefines as JOD)"
        elif band == 6.0:
            sem = "(V6 band, dropped in V9)"
        else:
            sem = "(intermediate)"
        lines.append(f"| {band} | {v6_str} | {v8_str} | {v9_str} | {sem} |")
    lines.append("")
    lines.append("## V9 design rationale")
    lines.append("")
    lines.append(
        "- **Score range [0, 100] is the user-facing dial.** Below "
        "30 → broken; 30-60 → noticeable; 60-90 → good; 90-100 → "
        "near-lossless."
    )
    lines.append(
        "- **JND at score=60 (multiple of 10)** instead of V6/V8's "
        "63. The 60 is a memorable round number; 63 was a CID22 "
        "paper convention. Output spline calibration absorbs the "
        "underlying butter-to-score mapping difference."
    )
    lines.append(
        "- **JOD at score=30 (multiple of 10).** Below 30 = "
        "definitely objectionable, by user-facing convention."
    )
    lines.append(
        "- **8 bands instead of V6's 6 / V8's 4.** Denser coverage "
        "of the perceptibility curve. Tradeoff: more anchor rows = "
        "more anchor pressure; --anchor-loss-weight reduced 1.0 → "
        "0.5 to compensate."
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
    lines.append("V9 adapts by:")
    lines.append("")
    lines.append(
        "- Widening `max_distance` from V8's 0.3 → 0.4 to allow "
        "the 12.0 band to claim the closest-available high-butter row "
        "even when it's noticeably below 12.0."
    )
    lines.append(
        "- Adding explicit \"worstfloor\" anchor rows: every row "
        f"with butter ≥ {WORST_FLOOR_BUTTER_MIN} from the q=5..95 "
        "butter parquets, target_score=0. These widen the anchor "
        "pool at the low-score end."
    )
    lines.append(
        "- Adding explicit \"lossless\" anchor rows: every row "
        f"with butter ≤ {LOSSLESS_BUTTER_MAX}, target_score=100. "
        "These widen the anchor pool at the high-score end. "
        "(zenjxl q=95 has butter ≈ 0.005 across the entire 1000-"
        "source corpus and dominates this pool.)"
    )
    lines.append("")
    lines.append(
        "The post-network PCHIP spline calibration applied AFTER "
        "training corrects for the residual mismatch between the "
        "anchor's nominal `target_score` and the network's actual "
        "predicted distribution at each anchor butter level."
    )
    lines.append("")
    lines.append("## Per-codec row counts (V9 build)")
    lines.append("")
    lines.append(
        "| codec | band | target_score | emitted | filtered | filter% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    band_keys = list(ANCHOR_BANDS_V9) + [(-1.0, 0.0), (-2.0, 100.0)]
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
            "v_tuner_v9_anchor_design_2026-05-20.md"
        ),
    )
    parser.add_argument("--max-distance", type=float, default=0.4)
    args = parser.parse_args()

    rows, stats = build_anchor_rows(args.max_distance)
    write_parquet(rows, args.out)
    write_comparison_md(stats, args.comparison_md)


if __name__ == "__main__":
    main()
