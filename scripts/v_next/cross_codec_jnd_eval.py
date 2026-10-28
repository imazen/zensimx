#!/usr/bin/env python3
"""Cross-codec JND consistency eval for V_05 profiles (2026-05-19).

For each of K source images and a target zensim score T under profile P:
  1. For each codec C in {JPEG, WebP, AVIF}, q-search to find the q whose
     zensim score under profile P is closest to T.
  2. Decode all 3 outputs back to PNG.
  3. Compute pairwise butteraugli (max + pnorm3) between the 3 decoded
     PNGs. The pair-wise mean should be SMALL — meaning the codecs landed
     at perceptually similar quality at the same zensim target.

Profiles tested:
  - v0_5_balanced (V_22-mix-LARGE+iwssim, 41 KB)
  - v0_5_compression (V_24-per-sample-α s4, 44 KB)
  - v0_5_ensemble (V_05-ensemble classifier + Balanced + Compression bakes)
  - v0_5_tuner (V_tuner-v2-s2 calibrated, 261 KB)

Output: per-(profile, target) TSV at OUT_DIR/<profile>_T<NN>.tsv plus
final summary table aggregated by (profile, target) → mean pairwise butter.

The "tied-rate dead zone" documented in CLAUDE.md applies to
Balanced/Compression/Ensemble: their raw output range is roughly
0-20 (distance-shaped), so target scores like 70/80/90 are
unreachable. We RECORD that as "target_unreachable" rather than
silently skipping — that IS the finding.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path("/home/lilith/work/zen/zensimx")
ZENSIM_BIN = ROOT / "target/release/examples/zensim_score_named"
ZEN_METRICS = Path("/home/lilith/work/zen/zenmetrics/target/release/zenmetrics")
QSWEEP_TSV = Path("/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv")
OUT_DIR = Path("/mnt/v/output/zensim/cross_codec_consistency_2026-05-19")


# 10 diverse images selected to span content types:
# 5 photos (web crawl), 3 generated charts (line-art), 2 hex-named
# (mixed content, likely synthetic).
SELECTED_IMAGES = [
    "/mnt/v/input/zensim/sources/00b13be94a4867dd_1022x818_512sq.png",  # photo
    "/mnt/v/input/zensim/sources/0543403b53d39228_512sq.png",            # photo
    "/mnt/v/input/zensim/sources/1248582_512sq.png",                     # photo (numeric ID, likely flickr-style)
    "/mnt/v/input/zensim/sources/2866385_512sq.png",                     # photo (numeric ID)
    "/mnt/v/input/zensim/sources/3d4f8bdb5b3733d3_512sq.png",            # photo
    "/mnt/v/input/zensim/sources/8f0cf3087dd4497f_512sq.png",            # photo
    "/mnt/v/input/zensim/sources/c20c6d8b7bdd7059_512sq.png",            # photo
    "/mnt/v/input/zensim/sources/gen-chart__00028_s4059c457_512sq.png",  # line-art / generated chart
    "/mnt/v/input/zensim/sources/gen-chart__00059_s1bd0eb8a_512sq.png",  # line-art
    "/mnt/v/input/zensim/sources/gen-chart__00125_s02bb2eae_512sq.png",  # line-art
]


PROFILES = ["v0_5_balanced", "v0_5_compression", "v0_5_ensemble", "v0_5_tuner"]
TARGETS = [30, 50, 63, 70, 80, 90]
CODECS = ["jpeg", "webp", "avif"]
Q_GRID = list(range(5, 100, 5))  # 19 points; common to all 3 codecs


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


def zensim_score(profile, ref, dist):
    """Score (ref, dist) under named profile, return float."""
    out = subprocess.check_output(
        [str(ZENSIM_BIN), profile, str(ref), str(dist)],
        text=True,
    )
    return float(out.strip())


def butteraugli_pair(a, b):
    """Return (max, pnorm3) butteraugli for the pair."""
    out = subprocess.check_output(
        [
            str(ZEN_METRICS),
            "score",
            "--metric",
            "butteraugli",
            "--reference",
            str(a),
            "--distorted",
            str(b),
        ],
        text=True,
    )
    bmax = bp3 = None
    for tok in out.split():
        if tok.startswith("butteraugli_max="):
            try:
                bmax = float(tok.split("=", 1)[1])
            except Exception:
                pass
        elif tok.startswith("butteraugli_pnorm3="):
            try:
                bp3 = float(tok.split("=", 1)[1])
            except Exception:
                pass
    return bmax, bp3


def find_q_closest_to_target(profile, ref, codec, work_dir, image_id, target):
    """Sweep Q_GRID, return (q*, achieved_score, decoded_png) closest to target.

    Reuses encodes/scores via filesystem caching.
    """
    best_q = None
    best_score = None
    best_path = None
    scores_by_q = {}
    for q in Q_GRID:
        ext = "jpg" if codec == "jpeg" else "webp" if codec == "webp" else "avif"
        enc_path = work_dir / f"{image_id}__{codec}_q{q:03d}.{ext}"
        dec_path = work_dir / f"{image_id}__{codec}_q{q:03d}.png"
        if not enc_path.exists():
            encode(ref, q, enc_path, codec)
        if not dec_path.exists():
            decode_to_png(enc_path, dec_path)
        s = zensim_score(profile, ref, dec_path)
        scores_by_q[q] = s
        if best_q is None or abs(s - target) < abs(best_score - target):
            best_q = q
            best_score = s
            best_path = dec_path
    return best_q, best_score, best_path, scores_by_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    images = SELECTED_IMAGES[: args.n_images]
    print(f"# eval start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    print(f"# {len(images)} images, {len(PROFILES)} profiles, {len(TARGETS)} targets, {len(CODECS)} codecs", flush=True)

    all_rows = []
    summary = {}  # (profile, target) -> list of mean_max per image

    for profile in PROFILES:
        for target in TARGETS:
            print(f"\n# profile={profile} target={target}", flush=True)
            rows_for_pt = []
            for img_path in images:
                img_path = Path(img_path)
                image_id = img_path.stem
                results = {}
                for codec in CODECS:
                    q, s, dec, sweep = find_q_closest_to_target(
                        profile, img_path, codec, work_dir, image_id, target
                    )
                    results[codec] = (q, s, dec, sweep)
                # Pairwise butteraugli on the 3 decoded outputs
                names = CODECS
                pair_bmax = []
                pair_bp3 = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        _, _, pi, _ = results[names[i]]
                        _, _, pj, _ = results[names[j]]
                        bmax, bp3 = butteraugli_pair(pi, pj)
                        if bmax is not None:
                            pair_bmax.append((names[i], names[j], bmax, bp3))
                            pair_bp3.append(bp3)
                mean_max = (
                    sum(p[2] for p in pair_bmax) / len(pair_bmax)
                    if pair_bmax
                    else float("nan")
                )
                max_max = max(p[2] for p in pair_bmax) if pair_bmax else float("nan")
                mean_p3 = sum(pair_bp3) / len(pair_bp3) if pair_bp3 else float("nan")

                # Score spread (max - min achieved)
                achieved = [results[c][1] for c in CODECS]
                score_spread = max(achieved) - min(achieved)
                # Distance from target (mean across codecs)
                dist_from_target = sum(abs(a - target) for a in achieved) / len(achieved)

                row = {
                    "profile": profile,
                    "target": target,
                    "image_id": image_id,
                    "jpeg_q": results["jpeg"][0],
                    "jpeg_zensim": results["jpeg"][1],
                    "webp_q": results["webp"][0],
                    "webp_zensim": results["webp"][1],
                    "avif_q": results["avif"][0],
                    "avif_zensim": results["avif"][1],
                    "score_spread": score_spread,
                    "dist_from_target": dist_from_target,
                    "pair_butter_max_mean": mean_max,
                    "pair_butter_max_worst": max_max,
                    "pair_butter_pnorm3_mean": mean_p3,
                    "pair_jpeg_webp_max": next(
                        (p[2] for p in pair_bmax if p[0] == "jpeg" and p[1] == "webp"),
                        float("nan"),
                    ),
                    "pair_jpeg_avif_max": next(
                        (p[2] for p in pair_bmax if p[0] == "jpeg" and p[1] == "avif"),
                        float("nan"),
                    ),
                    "pair_webp_avif_max": next(
                        (p[2] for p in pair_bmax if p[0] == "webp" and p[1] == "avif"),
                        float("nan"),
                    ),
                }
                rows_for_pt.append(row)
                all_rows.append(row)
                print(
                    f"  {image_id}: jpeg(q={row['jpeg_q']:>2} s={row['jpeg_zensim']:>6.2f}) "
                    f"webp(q={row['webp_q']:>2} s={row['webp_zensim']:>6.2f}) "
                    f"avif(q={row['avif_q']:>2} s={row['avif_zensim']:>6.2f}) "
                    f"butter_max(mean={mean_max:.3f} worst={max_max:.3f}) "
                    f"score_spread={score_spread:.2f}",
                    flush=True,
                )

            # Per-(profile, target) summary
            valid_butter = [r["pair_butter_max_mean"] for r in rows_for_pt if r["pair_butter_max_mean"] == r["pair_butter_max_mean"]]
            valid_p3 = [r["pair_butter_pnorm3_mean"] for r in rows_for_pt if r["pair_butter_pnorm3_mean"] == r["pair_butter_pnorm3_mean"]]
            valid_spread = [r["score_spread"] for r in rows_for_pt]
            valid_dist = [r["dist_from_target"] for r in rows_for_pt]
            summary[(profile, target)] = {
                "mean_butter_max": sum(valid_butter) / len(valid_butter) if valid_butter else float("nan"),
                "mean_butter_pnorm3": sum(valid_p3) / len(valid_p3) if valid_p3 else float("nan"),
                "mean_score_spread": sum(valid_spread) / len(valid_spread),
                "mean_dist_from_target": sum(valid_dist) / len(valid_dist),
                "n": len(rows_for_pt),
            }

    # Write raw TSV
    raw_tsv = out_dir / "cross_codec_raw_2026-05-19.tsv"
    if all_rows:
        cols = list(all_rows[0].keys())
        with open(raw_tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in all_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"\nraw TSV: {raw_tsv}", flush=True)

    # Write summary JSON + Markdown
    summary_json = out_dir / "cross_codec_summary_2026-05-19.json"
    out_summary_serializable = {
        f"{k[0]}@T{k[1]}": v for k, v in summary.items()
    }
    with open(summary_json, "w") as f:
        json.dump(out_summary_serializable, f, indent=2)
    print(f"summary JSON: {summary_json}", flush=True)

    # Print summary table
    print("\n## Summary table — pair_butter_max_mean (lower = more consistent)")
    print("| Profile | " + " | ".join(f"T={t}" for t in TARGETS) + " |")
    print("|---|" + "|".join(["---:"] * len(TARGETS)) + "|")
    for p in PROFILES:
        cells = []
        for t in TARGETS:
            v = summary.get((p, t), {}).get("mean_butter_max", float("nan"))
            cells.append(f"{v:.2f}" if v == v else "n/a")
        print(f"| {p} | " + " | ".join(cells) + " |")

    print("\n## Summary table — mean_score_spread (variance of achieved scores)")
    print("| Profile | " + " | ".join(f"T={t}" for t in TARGETS) + " |")
    print("|---|" + "|".join(["---:"] * len(TARGETS)) + "|")
    for p in PROFILES:
        cells = []
        for t in TARGETS:
            v = summary.get((p, t), {}).get("mean_score_spread", float("nan"))
            cells.append(f"{v:.2f}" if v == v else "n/a")
        print(f"| {p} | " + " | ".join(cells) + " |")

    print("\n## Summary table — mean_dist_from_target (closer = reachable)")
    print("| Profile | " + " | ".join(f"T={t}" for t in TARGETS) + " |")
    print("|---|" + "|".join(["---:"] * len(TARGETS)) + "|")
    for p in PROFILES:
        cells = []
        for t in TARGETS:
            v = summary.get((p, t), {}).get("mean_dist_from_target", float("nan"))
            cells.append(f"{v:.2f}" if v == v else "n/a")
        print(f"| {p} | " + " | ".join(cells) + " |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
