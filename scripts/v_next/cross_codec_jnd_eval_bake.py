#!/usr/bin/env python3
"""Cross-codec JND consistency eval for an arbitrary bake (TunerV2 evaluation).

Same methodology as scripts/v_next/cross_codec_jnd_eval.py — but uses
`score_pair_with_bake --bake <path>` instead of the named-profile binary.
Targets the 6 standard score levels {30, 50, 63, 70, 80, 90} across
JPEG/WebP/AVIF on 10 source images, and reports cross-codec butter mean
+ score spread per target.

Usage:
  cross_codec_jnd_eval_bake.py --bake PATH [--label NAME] [--bake-post clamp|raw|mapped]

Defaults:
  bake_post = "clamp" (today's Tuner is score-shaped after MSE training)
  label    = basename(bake)

Output:
  /mnt/v/output/zensim/cross_codec_consistency_2026-05-19/<label>/<label>_T<T>.tsv
  + summary JSON at the parent dir + summary table on stdout.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path("/home/lilith/work/zen/zensimx")
SCORE_BAKE_BIN = ROOT / "target/release/score_pair_with_bake"
ZEN_METRICS = Path("/home/lilith/work/zen/zenmetrics/target/release/zenmetrics")
OUT_DIR = Path("/mnt/v/output/zensim/cross_codec_consistency_2026-05-19")

# Same 10 images as cross_codec_jnd_eval.py for direct comparability
SELECTED_IMAGES = [
    "/mnt/v/input/zensim/sources/00b13be94a4867dd_1022x818_512sq.png",
    "/mnt/v/input/zensim/sources/0543403b53d39228_512sq.png",
    "/mnt/v/input/zensim/sources/1248582_512sq.png",
    "/mnt/v/input/zensim/sources/2866385_512sq.png",
    "/mnt/v/input/zensim/sources/3d4f8bdb5b3733d3_512sq.png",
    "/mnt/v/input/zensim/sources/8f0cf3087dd4497f_512sq.png",
    "/mnt/v/input/zensim/sources/c20c6d8b7bdd7059_512sq.png",
    "/mnt/v/input/zensim/sources/gen-chart__00028_s4059c457_512sq.png",
    "/mnt/v/input/zensim/sources/gen-chart__00059_s1bd0eb8a_512sq.png",
    "/mnt/v/input/zensim/sources/gen-chart__00125_s02bb2eae_512sq.png",
]

TARGETS = [30, 50, 63, 70, 80, 90]
CODECS = ["jpeg", "webp", "avif"]
Q_GRID = list(range(5, 100, 5))


def encode(src, q, out_path, codec):
    img = Image.open(src).convert("RGB")
    if codec == "jpeg":
        img.save(out_path, "JPEG", quality=int(q), subsampling=2, optimize=False, progressive=False)
    elif codec == "webp":
        img.save(out_path, "WEBP", quality=int(q), method=4)
    elif codec == "avif":
        img.save(out_path, "AVIF", quality=int(q), speed=6)
    else:
        raise ValueError(codec)


def decode_to_png(in_path, out_png):
    img = Image.open(in_path).convert("RGB")
    img.save(out_png, "PNG")


def bake_score(bake_path, bake_post, ref, dist):
    out = subprocess.check_output(
        [
            str(SCORE_BAKE_BIN),
            "--bake", str(bake_path),
            "--bake-post", bake_post,
            "--ref", str(ref),
            "--dist", str(dist),
        ],
        text=True,
    )
    return float(out.strip())


def butteraugli_pair(a, b):
    out = subprocess.check_output(
        [str(ZEN_METRICS), "score", "--metric", "butteraugli",
         "--reference", str(a), "--distorted", str(b)],
        text=True,
    )
    bmax = bp3 = None
    for tok in out.split():
        if tok.startswith("butteraugli_max="):
            try: bmax = float(tok.split("=", 1)[1])
            except: pass
        elif tok.startswith("butteraugli_pnorm3="):
            try: bp3 = float(tok.split("=", 1)[1])
            except: pass
    return bmax, bp3


def find_q_closest(bake, bake_post, ref, codec, work_dir, image_id, target, score_cache):
    best_q = None; best_score = None; best_path = None
    for q in Q_GRID:
        ext = "jpg" if codec == "jpeg" else "webp" if codec == "webp" else "avif"
        enc = work_dir / f"{image_id}__{codec}_q{q:03d}.{ext}"
        dec = work_dir / f"{image_id}__{codec}_q{q:03d}.png"
        if not enc.exists():
            encode(ref, q, enc, codec)
        if not dec.exists():
            decode_to_png(enc, dec)
        key = (str(bake), str(dec))
        if key in score_cache:
            s = score_cache[key]
        else:
            s = bake_score(bake, bake_post, ref, dec)
            score_cache[key] = s
        if best_q is None or abs(s - target) < abs(best_score - target):
            best_q = q; best_score = s; best_path = dec
    return best_q, best_score, best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True, type=Path)
    ap.add_argument("--label")
    ap.add_argument("--bake-post", default="clamp", help="raw|clamp|mapped[:A,B]")
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    label = args.label or args.bake.stem
    out_dir = Path(args.out_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.out_dir) / "work_shared"  # reuse encodes across bakes
    work_dir.mkdir(parents=True, exist_ok=True)

    images = SELECTED_IMAGES[: args.n_images]
    print(f"# bake={args.bake.name} label={label} bake_post={args.bake_post}", flush=True)
    print(f"# {len(images)} images × {len(TARGETS)} targets × {len(CODECS)} codecs", flush=True)

    score_cache = {}
    all_rows = []
    summary = {}
    for target in TARGETS:
        print(f"\n# target={target}", flush=True)
        rows = []
        for img_path in images:
            img_path = Path(img_path)
            image_id = img_path.stem
            results = {}
            for codec in CODECS:
                q, s, dec = find_q_closest(
                    args.bake, args.bake_post, img_path, codec, work_dir,
                    image_id, target, score_cache,
                )
                results[codec] = (q, s, dec)
            # Pairwise butter
            pair_bmax = []
            pair_bp3 = []
            names = CODECS
            for i in range(3):
                for j in range(i + 1, 3):
                    _, _, pi = results[names[i]]
                    _, _, pj = results[names[j]]
                    bmax, bp3 = butteraugli_pair(pi, pj)
                    if bmax is not None:
                        pair_bmax.append((names[i], names[j], bmax, bp3))
                        pair_bp3.append(bp3)
            mean_max = sum(p[2] for p in pair_bmax) / len(pair_bmax) if pair_bmax else float("nan")
            max_max = max(p[2] for p in pair_bmax) if pair_bmax else float("nan")
            mean_p3 = sum(pair_bp3) / len(pair_bp3) if pair_bp3 else float("nan")

            achieved = [results[c][1] for c in CODECS]
            score_spread = max(achieved) - min(achieved)
            dist_from_target = sum(abs(a - target) for a in achieved) / len(achieved)

            row = {
                "target": target, "image_id": image_id,
                "jpeg_q": results["jpeg"][0], "jpeg_zensim": results["jpeg"][1],
                "webp_q": results["webp"][0], "webp_zensim": results["webp"][1],
                "avif_q": results["avif"][0], "avif_zensim": results["avif"][1],
                "score_spread": score_spread,
                "dist_from_target": dist_from_target,
                "pair_butter_max_mean": mean_max,
                "pair_butter_max_worst": max_max,
                "pair_butter_pnorm3_mean": mean_p3,
                "pair_jpeg_webp_max": next((p[2] for p in pair_bmax if p[0] == "jpeg" and p[1] == "webp"), float("nan")),
                "pair_jpeg_avif_max": next((p[2] for p in pair_bmax if p[0] == "jpeg" and p[1] == "avif"), float("nan")),
                "pair_webp_avif_max": next((p[2] for p in pair_bmax if p[0] == "webp" and p[1] == "avif"), float("nan")),
            }
            rows.append(row); all_rows.append(row)
            print(
                f"  {image_id}: j(q{row['jpeg_q']:>2} s{row['jpeg_zensim']:>5.1f}) "
                f"w(q{row['webp_q']:>2} s{row['webp_zensim']:>5.1f}) "
                f"a(q{row['avif_q']:>2} s{row['avif_zensim']:>5.1f}) "
                f"butter={mean_max:.2f} spread={score_spread:.1f}",
                flush=True,
            )
        valid_b = [r["pair_butter_max_mean"] for r in rows if r["pair_butter_max_mean"] == r["pair_butter_max_mean"]]
        valid_p3 = [r["pair_butter_pnorm3_mean"] for r in rows if r["pair_butter_pnorm3_mean"] == r["pair_butter_pnorm3_mean"]]
        valid_spr = [r["score_spread"] for r in rows]
        valid_d = [r["dist_from_target"] for r in rows]
        summary[target] = {
            "mean_butter_max": (sum(valid_b)/len(valid_b)) if valid_b else float("nan"),
            "mean_butter_pnorm3": (sum(valid_p3)/len(valid_p3)) if valid_p3 else float("nan"),
            "mean_score_spread": sum(valid_spr)/len(valid_spr),
            "mean_dist_from_target": sum(valid_d)/len(valid_d),
            "n": len(rows),
        }

    raw_tsv = out_dir / f"{label}_raw.tsv"
    if all_rows:
        cols = list(all_rows[0].keys())
        with open(raw_tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in all_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"\nraw TSV: {raw_tsv}", flush=True)
    summary_json = out_dir / f"{label}_summary.json"
    with open(summary_json, "w") as f:
        json.dump({f"{label}@T{k}": v for k, v in summary.items()}, f, indent=2)
    print(f"summary JSON: {summary_json}", flush=True)

    print(f"\n## {label} — mean pairwise butter_max (lower = consistent)")
    print("| Target | " + " | ".join(f"T={t}" for t in TARGETS) + " |")
    print("|---|" + "|".join(["---:"] * len(TARGETS)) + "|")
    print(f"| {label} | " + " | ".join(
        f"{summary[t]['mean_butter_max']:.2f}" if summary[t]['mean_butter_max'] == summary[t]['mean_butter_max'] else "n/a"
        for t in TARGETS) + " |")
    print(f"\n## {label} — mean dist_from_target")
    print(f"| {label} | " + " | ".join(f"{summary[t]['mean_dist_from_target']:.2f}" for t in TARGETS) + " |")
    print(f"\n## {label} — mean score_spread")
    print(f"| {label} | " + " | ".join(f"{summary[t]['mean_score_spread']:.2f}" for t in TARGETS) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
