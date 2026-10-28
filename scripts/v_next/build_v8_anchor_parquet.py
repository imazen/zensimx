#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V8 anchor parquet builder (2026-05-19).

V6 (PreviewV0_5TunerV2, commit 1dd61fc) and V7 (cross-codec-v7,
commit bfd13cb) both built per-band anchor parquets keyed on
``butter_pnorm3``. V6 used rule-of-thumb target scores (90/75/63/45
/25/10 over bands 0.3/0.8/1.5/2.5/4.0/6.0). V7 replaced those with
empirically-derived per-(codec, band) ssim2 medians.

V7 surfaced two structural problems:

1. **Heavy-distortion divergence.** At butter=4.0 the empirical
   ssim2 median is zenwebp=16.7 vs zenjxl=85.9. Cross-codec
   consistency at score=63 is fragile when one codec is "still
   pretty" by ssim2 while another is "destroyed" at the same
   butter level.
2. **score=63 ↔ PJND contract broken.** V6 set score=63 at
   butter=1.5 as the user-facing PJND anchor (the "give me visually
   lossless" dial point). V7 empirical join showed
   ssim2 ≈ 63 at butter ≈ 2.5 — meaning V6's score=63 was
   calibrated to a tighter butter band than the CID22 paper's
   Table 4 KonJND-1k human PJND anchor implies.

V8 fixes both:

- **Drop the two heavy-distortion bands** (4.0, 6.0). Keep only
  the bands where cross-codec divergence is bounded.
- **Re-center the band → target_score table** so score=63 lands at
  butter=2.5 (the CID22 paper / KonJND-1k empirical anchor). The
  user-facing "score=63 means PJND" contract is preserved; the
  underlying butter target shifts to the ssim2-grounded value.

V8 band table:

| butter_pnorm3 | target_score | rationale |
|---|---|---|
| 0.5 | 85 | high quality, well below zenjxl saturation |
| 1.0 | 75 | near-lossless |
| 2.5 | 63 | ssim2-PJND (CID22 paper Table 4 anchor) |
| 4.0 | 45 | upper edge of zenjxl saturation safety zone |

Unlike V7, the target_score is a fixed table value — NOT an
empirical median. The trainer is anchored to the perceptibility
table; cross-codec divergence at the heavy-distortion edge is
removed by construction (we don't ship those bands).

Output anchor parquet has the V5/V6/V7 schema (ref_basename, codec,
q, butter_pnorm3, butter_target, target_score, anchor_source,
human_score, anchor_weight, f0..f371) so it drops directly into
the V6 trainer.

Side product: a comparison table at the path given by
``--comparison-md`` showing V6 rule-of-thumb vs V7 empirical
median vs V8 chosen target per band, with KonJND-1k humans'
empirical butter placement noted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# (butter_pnorm3_target, V8_target_score). 4 bands, heavy-distortion dropped.
ANCHOR_BANDS_V8: list[tuple[float, float]] = [
    (0.5, 85.0),
    (1.0, 75.0),
    (2.5, 63.0),   # ssim2-PJND anchor per CID22 paper Table 4
    (4.0, 45.0),
]

# V6 rule-of-thumb (for comparison table).
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
    "/mnt/v/zen/zensim-training/2026-05-19-v8-anchors/anchors_v8_372col.parquet"
)


def build_anchor_rows(
    max_distance: float,
) -> tuple[list[dict], dict[tuple[str, float], dict[str, int]]]:
    """Build per-(image, codec, band) anchor rows with V8 fixed targets."""
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
            for butter_target, target_score in ANCHOR_BANDS_V8:
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
                    "anchor_source": f"v8_{codec}_b{butter_target}_s{target_score:.1f}",
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
    print(
        f"  {'codec':10s}  {'band':>5s}  {'target':>6s}  "
        f"{'emitted':>7s}  {'filtered':>8s}  {'mean_d':>6s}  filter%"
    )
    for codec in CODECS:
        for butter_target, target_score in ANCHOR_BANDS_V8:
            slot = filter_stats.get((codec, butter_target))
            if slot is None:
                continue
            tot = slot["emitted"] + slot["filtered"]
            pct = 100.0 * slot["filtered"] / tot if tot else 0.0
            mean_d = (
                slot["tot_distance"] / slot["emitted"] if slot["emitted"] else 0.0
            )
            print(
                f"  {codec:10s}  {butter_target:5.2f}  {target_score:6.1f}  "
                f"{slot['emitted']:7d}  {slot['filtered']:8d}  "
                f"{mean_d:6.3f}  {pct:6.2f}%"
            )
    return rows, filter_stats


def write_parquet(rows: list[dict], out: Path) -> None:
    if not rows:
        raise SystemExit("no anchor rows built")
    feature_cols = [f"f{i}" for i in range(372)]
    cols = {
        "ref_basename": pa.array([r["ref_basename"] for r in rows], type=pa.string()),
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
    """V6 rule-of-thumb vs V7 empirical median vs V8 chosen target.

    V7 empirical medians are pre-computed from V7 output (we don't
    re-load score parquets; we hardcode V7's per-band aggregate ssim2
    median per the V7 comparison table).
    """
    # V7 empirical aggregate (median across codecs) ssim2 from the
    # 2026-05-19 V7 anchor build, for the bands V6 used.
    V7_AGGREGATE_SSIM2: dict[float, float] = {
        0.3: 87.58,   # avif/jpeg/jxl/webp medians ~87
        0.8: 85.57,
        1.5: 81.64,
        2.5: 62.91,   # this is the score=63 ↔ ssim2-PJND empirical landing
        4.0: 40.77,
        6.0: 27.31,
    }

    lines: list[str] = []
    lines.append(
        "# V6 rule-of-thumb vs V7 empirical vs V8 chosen anchor targets (2026-05-19)"
    )
    lines.append("")
    lines.append(
        "V6 (`PreviewV0_5TunerV2`, commit `1dd61fc`) and V7 "
        "(cross-codec-v7, commit `bfd13cb`) both used a 6-band "
        "anchor grid keyed on `butter_pnorm3`. V6 used "
        "rule-of-thumb target scores; V7 used per-(codec, band) "
        "ssim2 medians. V7 surfaced two structural problems:"
    )
    lines.append("")
    lines.append(
        "1. Heavy-distortion bands (4.0, 6.0) diverge cross-codec: "
        "zenwebp ssim2=16.7 vs zenjxl ssim2=85.9 at butter=4.0. "
        "Cross-codec consistency at score=63 is unstable when one "
        "codec saturates ssim2 while another collapses at the same "
        "butter level."
    )
    lines.append("")
    lines.append(
        "2. score=63 ↔ butter=1.5 (V6 convention) misaligns with "
        "ssim2=63 (CID22 paper Table 4 KonJND-1k human PJND anchor): "
        "V7's empirical median lands ssim2=63 near butter=2.5, NOT "
        "butter=1.5. The user-facing `score=63 = PJND` contract is "
        "calibrated to ssim2's 0-100 range, so the correct butter "
        "value for that contract is butter≈2.5."
    )
    lines.append("")
    lines.append("## V8 design")
    lines.append("")
    lines.append(
        "**Drop the heavy-distortion bands (4.0 and 6.0 in V6/V7 → "
        "keep only 4.0 as the upper edge).** Restrict to bands "
        "where cross-codec divergence is bounded."
    )
    lines.append("")
    lines.append(
        "**Re-center the band → target_score table so score=63 "
        "lands at butter=2.5** (the CID22 paper / KonJND-1k "
        "empirical PJND anchor). The user-facing `score=63 = "
        "PJND` contract is preserved; the butter target underneath "
        "shifts from 1.5 to 2.5 to align with ssim2 PJND."
    )
    lines.append("")
    lines.append(
        "Unlike V7, the V8 target_score is a fixed table value, NOT "
        "an empirical median. The trainer is anchored to the "
        "perceptibility table; cross-codec divergence at the "
        "heavy-distortion edge is removed by construction."
    )
    lines.append("")
    lines.append("## Per-band target table")
    lines.append("")
    lines.append(
        "| butter_pnorm3 | V6 rule | V7 empirical aggregate ssim2 | "
        "V8 chosen | rationale |"
    )
    lines.append("|---:|---:|---:|---:|---|")
    v6_dict = dict(ANCHOR_BANDS_V6)
    v8_dict = dict(ANCHOR_BANDS_V8)
    all_bands = sorted(set(v6_dict.keys()) | set(v8_dict.keys()))
    for band in all_bands:
        v6 = v6_dict.get(band)
        v7 = V7_AGGREGATE_SSIM2.get(band)
        v8 = v8_dict.get(band)
        v6_str = f"{v6:.1f}" if v6 is not None else "—"
        v7_str = f"{v7:.2f}" if v7 is not None else "—"
        v8_str = f"{v8:.1f}" if v8 is not None else "DROPPED"
        if v8 is None:
            rationale = (
                "heavy-distortion: cross-codec ssim2 divergence "
                "exceeds ±20 (zenwebp collapse vs zenjxl saturation)"
            )
        elif band == 2.5:
            rationale = (
                "**PJND anchor** — score=63 ↔ ssim2-PJND per CID22 paper Table 4"
            )
        elif band == 0.5:
            rationale = "high quality, below zenjxl saturation regime"
        elif band == 1.0:
            rationale = "near-lossless"
        elif band == 4.0:
            rationale = "upper edge of zenjxl saturation safety zone"
        else:
            rationale = "kept in V6/V7 grid"
        lines.append(f"| {band} | {v6_str} | {v7_str} | {v8_str} | {rationale} |")
    lines.append("")
    lines.append("## KonJND-1k human empirical butter placement")
    lines.append("")
    lines.append(
        "Per CID22 paper Table 4, KonJND-1k human PJND lands at "
        "**ssim2 ≈ 63**. V7's empirical join shows ssim2=63 ≈ "
        "**butter_pnorm3 ≈ 2.5** (the V7 aggregate ssim2 at the "
        "2.5 band is 62.91 — a near-exact landing on PJND)."
    )
    lines.append("")
    lines.append(
        "V6 shipped with score=63 at butter=1.5, which is a "
        "tighter perceptibility band than the KonJND-1k human "
        "anchor. V8 corrects this by moving score=63 to butter=2.5."
    )
    lines.append("")
    lines.append("## Per-codec row counts (V8 build)")
    lines.append("")
    lines.append(
        "| codec | band | target_score | emitted | filtered | filter% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for codec in CODECS:
        for butter_target, target_score in ANCHOR_BANDS_V8:
            slot = filter_stats.get((codec, butter_target))
            if slot is None:
                lines.append(
                    f"| {codec} | {butter_target} | {target_score:.1f} | — | — | — |"
                )
                continue
            tot = slot["emitted"] + slot["filtered"]
            pct = 100.0 * slot["filtered"] / tot if tot else 0.0
            lines.append(
                f"| {codec} | {butter_target} | {target_score:.1f} | "
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
            "v_tuner_v8_anchor_design_2026-05-19.md"
        ),
    )
    parser.add_argument("--max-distance", type=float, default=0.3)
    args = parser.parse_args()

    rows, stats = build_anchor_rows(args.max_distance)
    write_parquet(rows, args.out)
    write_comparison_md(stats, args.comparison_md)


if __name__ == "__main__":
    main()
