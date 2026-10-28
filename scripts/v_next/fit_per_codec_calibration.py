#!/usr/bin/env python3
"""Fit per-codec affine calibration for PreviewV0_5Tuner (2026-05-19).

Goal: at any given perceptual quality (anchored by ssim2 score), the
Tuner output should agree across codecs. Currently the Tuner raw is
codec-dependent — at PJND (~ssim2=63), jpeg yields raw X, webp yields
raw Y, avif yields raw Z, all different. That means "user types
score=63" lands on different visual qualities per codec.

Fit α_C, β_C per codec C such that:
    calibrated_C = α_C + β_C * tuner_raw_C
minimizes MSE against ssim2_score on the per-codec q-sweep cache.

ssim2 is the empirical PJND anchor per the CID22 paper (Table 4): mean
KonJND-1k PJND threshold lands at ssim2 ≈ 63. By fitting Tuner to ssim2
within each codec, we make "calibrated_C = 63" mean the same visual
quality across codecs by construction.

Input data: pre-encoded codec outputs cached at
  /mnt/v/output/zensim/cross_codec_consistency_2026-05-19/work/
that the existing cross_codec_jnd_eval.py produced. 10 refs × 19 q ×
3 codecs = 570 pairs.

Output:
  - Per-codec fit summary printed to stdout.
  - Rust source snippet (codec_calibration.rs) ready to paste.
  - JSON sidecar with raw fit data.

Methodology:
  - Score every (ref, dist@q) with PreviewV0_5Tuner → tuner_raw_C.
  - Score every (ref, dist@q) with ssim2 (CPU) → ssim2_C.
  - Linear regression per codec C: ssim2 = α + β * tuner_raw.
  - Verify fit with R² and residual MSE.
  - For zenjxl + zenpng (no data here): use mean of {jpeg, webp, avif}
    for zenjxl; (0, 1) identity for zenpng (lossless).
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

WORK_DIR = Path("/mnt/v/output/zensim/cross_codec_consistency_2026-05-19/work")
ZENSIM_BIN = Path("/home/lilith/work/zen/zensimx/target/release/examples/zensim_score_named")
ZEN_METRICS = Path("/home/lilith/work/zen/zenmetrics/target/release/zenmetrics")
SOURCES_DIR = Path("/mnt/v/input/zensim/sources")

# Same 10 images as cross_codec_jnd_eval.py
SELECTED_IMAGES = [
    "00b13be94a4867dd_1022x818_512sq.png",
    "0543403b53d39228_512sq.png",
    "1248582_512sq.png",
    "2866385_512sq.png",
    "3d4f8bdb5b3733d3_512sq.png",
    "8f0cf3087dd4497f_512sq.png",
    "c20c6d8b7bdd7059_512sq.png",
    "gen-chart__00028_s4059c457_512sq.png",
    "gen-chart__00059_s1bd0eb8a_512sq.png",
    "gen-chart__00125_s02bb2eae_512sq.png",
]

CODECS = ["jpeg", "webp", "avif"]
Q_GRID = list(range(5, 100, 5))  # 19 points


def tuner_score(ref_path: Path, dist_path: Path) -> float:
    """Run zensim_score_named with v0_5_tuner profile."""
    out = subprocess.check_output(
        [str(ZENSIM_BIN), "v0_5_tuner", str(ref_path), str(dist_path)],
        text=True,
    )
    return float(out.strip())


def write_pairs_tsv(pairs, out_tsv):
    """Write a 2-col ref_path/dist_path TSV for zen-metrics batch."""
    with open(out_tsv, "w") as f:
        f.write("ref_path\tdist_path\n")
        for r, d in pairs:
            f.write(f"{r}\t{d}\n")


def read_pairs_tsv(in_tsv):
    """Read zen-metrics batch output: header + rows with ref_path, dist_path, ssim2."""
    rows = []
    with open(in_tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_ssim2 = header.index("ssim2")
        idx_ref = header.index("ref_path")
        idx_dist = header.index("dist_path")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append((parts[idx_ref], parts[idx_dist], float(parts[idx_ssim2])))
    return rows


def batch_ssim2(pairs, out_tsv):
    """Use zen-metrics batch to score ssim2 on a list of (ref, dist) pairs."""
    tsv_in = out_tsv.with_suffix(".pairs.tsv")
    write_pairs_tsv(pairs, tsv_in)
    cmd = [
        str(ZEN_METRICS),
        "batch",
        "--metric", "ssim2",
        "--pairs", str(tsv_in),
        "--output", str(out_tsv),
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return read_pairs_tsv(out_tsv)


def linear_fit(xs, ys):
    """Ordinary least squares: y = α + β·x. Returns (α, β, r2, mse)."""
    n = len(xs)
    if n < 2:
        return 0.0, 1.0, float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return my, 0.0, 0.0, sum((y - my) ** 2 for y in ys) / n
    beta = sxy / sxx
    alpha = my - beta * mx
    syy = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (alpha + beta * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / syy if syy > 0 else 0.0)
    mse = ss_res / n
    return alpha, beta, r2, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/v/output/zensim/per_codec_calibration_2026-05-19")
    ap.add_argument("--n-images", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# fit_per_codec_calibration start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    print(f"# work dir: {WORK_DIR}", flush=True)
    print(f"# zen-metrics: {ZEN_METRICS}", flush=True)
    print(f"# tuner bin: {ZENSIM_BIN}", flush=True)

    images = SELECTED_IMAGES[: args.n_images]
    per_codec_data = {}  # codec -> list of (image_id, q, tuner_raw, ssim2)

    # Phase 1: gather (tuner_raw, ssim2) pairs per codec
    for codec in CODECS:
        print(f"\n# codec={codec}", flush=True)
        pairs = []  # list of (ref_path, dist_png_path, image_id, q)
        for img_name in images:
            img_id = img_name.replace(".png", "")
            ref_path = SOURCES_DIR / img_name
            for q in Q_GRID:
                dec_path = WORK_DIR / f"{img_id}__{codec}_q{q:03d}.png"
                if not dec_path.exists():
                    print(f"  WARN: missing {dec_path}", flush=True)
                    continue
                pairs.append((str(ref_path), str(dec_path), img_id, q))

        # ssim2 batch
        pairs_for_ssim2 = [(p[0], p[1]) for p in pairs]
        ssim2_out = out_dir / f"ssim2_{codec}.tsv"
        ssim2_rows = batch_ssim2(pairs_for_ssim2, ssim2_out)
        # Map ref+dist -> ssim2
        ssim2_map = {(r, d): s for (r, d, s) in ssim2_rows}

        # tuner scoring (slower; per-pair subprocess)
        rows = []
        for ref_str, dist_str, img_id, q in pairs:
            key = (ref_str, dist_str)
            ssim2 = ssim2_map.get(key)
            if ssim2 is None:
                continue
            tuner = tuner_score(Path(ref_str), Path(dist_str))
            rows.append({
                "image_id": img_id,
                "q": q,
                "tuner_raw": tuner,
                "ssim2": ssim2,
            })
        per_codec_data[codec] = rows
        print(f"  {len(rows)} (tuner, ssim2) pairs", flush=True)

    # Phase 2: fit per-codec (α, β)
    fits = {}
    for codec in CODECS:
        rows = per_codec_data[codec]
        xs = [r["tuner_raw"] for r in rows]
        ys = [r["ssim2"] for r in rows]
        alpha, beta, r2, mse = linear_fit(xs, ys)
        fits[codec] = {
            "alpha": alpha,
            "beta": beta,
            "r2": r2,
            "mse": mse,
            "n": len(rows),
            "tuner_raw_min": min(xs) if xs else float("nan"),
            "tuner_raw_max": max(xs) if xs else float("nan"),
            "tuner_raw_mean": sum(xs)/len(xs) if xs else float("nan"),
            "ssim2_min": min(ys) if ys else float("nan"),
            "ssim2_max": max(ys) if ys else float("nan"),
            "ssim2_mean": sum(ys)/len(ys) if ys else float("nan"),
        }
        print(
            f"  {codec}: n={len(rows)} α={alpha:.4f} β={beta:.4f} R²={r2:.4f} MSE={mse:.3f}",
            flush=True,
        )

    # Phase 3: zenjxl placeholder = mean of jpeg/webp/avif
    mean_alpha = sum(fits[c]["alpha"] for c in CODECS) / len(CODECS)
    mean_beta = sum(fits[c]["beta"] for c in CODECS) / len(CODECS)
    fits["zenjxl"] = {
        "alpha": mean_alpha,
        "beta": mean_beta,
        "r2": float("nan"),
        "mse": float("nan"),
        "n": 0,
        "source": "mean of jpeg/webp/avif (no per-codec data)",
    }
    # zenpng = lossless → identity
    fits["zenpng"] = {
        "alpha": 0.0,
        "beta": 1.0,
        "r2": float("nan"),
        "mse": float("nan"),
        "n": 0,
        "source": "lossless codec — identity",
    }
    # Identity (no calibration) for completeness
    fits["identity"] = {"alpha": 0.0, "beta": 1.0, "r2": float("nan"), "mse": float("nan"), "n": 0}

    # Save JSON
    out_json = out_dir / "fits.json"
    with open(out_json, "w") as f:
        json.dump({
            "fits": fits,
            "per_codec_data": per_codec_data,
            "n_images": len(images),
            "q_grid": Q_GRID,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    print(f"\n# fits JSON: {out_json}", flush=True)

    # Rust snippet
    rust_lines = []
    rust_lines.append("// Auto-generated by scripts/v_next/fit_per_codec_calibration.py")
    rust_lines.append(f"// Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    rust_lines.append("// Profile: PreviewV0_5Tuner; anchor: ssim2 (CPU)")
    rust_lines.append("// fit_per_codec_calibration.py + cross_codec_consistency_2026-05-19/work/")
    rust_lines.append("")
    for c in ["jpeg", "webp", "avif", "zenjxl", "zenpng"]:
        f_ = fits[c]
        rust_lines.append(f"// {c}: α={f_['alpha']:.6f} β={f_['beta']:.6f} R²={f_['r2']:.4f} n={f_['n']}")
    rust_lines.append("")
    print("\n".join(rust_lines))

    # Summary table
    print("\n## Per-codec calibration table\n")
    print("| Codec | α | β | R² | MSE | n |")
    print("|---|---:|---:|---:|---:|---:|")
    for c in ["jpeg", "webp", "avif", "zenjxl", "zenpng"]:
        f_ = fits[c]
        r2 = f_["r2"]
        mse = f_["mse"]
        r2_str = f"{r2:.4f}" if (r2 == r2) else "n/a"
        mse_str = f"{mse:.3f}" if (mse == mse) else "n/a"
        print(f"| {c} | {f_['alpha']:.4f} | {f_['beta']:.4f} | {r2_str} | {mse_str} | {f_['n']} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
