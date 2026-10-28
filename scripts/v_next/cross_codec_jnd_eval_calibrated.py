#!/usr/bin/env python3
"""Cross-codec JND consistency eval with per-codec calibration (2026-05-19).

Re-runs the cross_codec_jnd_eval.py methodology against
`PreviewV0_5Tuner` with the new `--per-codec-calibration on` flag.
Compares pre-calibration (raw Tuner output) vs post-calibration mean
pairwise butteraugli at each target score.

Reuses the existing encode/decode cache at
`/mnt/v/output/zensim/cross_codec_consistency_2026-05-19/work/`.
Pairwise butteraugli is re-computed on the post-calibration q picks
(the q choice may differ when calibration changes which q lands
closest to T).
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
WORK_DIR_PRECOMPUTED = Path("/mnt/v/output/zensim/cross_codec_consistency_2026-05-19/work")
OUT_DIR = Path("/mnt/v/output/zensim/per_codec_calibration_2026-05-19/eval")

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


def zensim_score(ref: Path, dist: Path, codec: str, calibrated: bool) -> float:
    """Score (ref, dist) under v0_5_tuner. When calibrated=True, pass the codec name."""
    cmd = [str(ZENSIM_BIN), "v0_5_tuner", str(ref), str(dist)]
    if calibrated:
        cmd += ["--codec", codec, "--per-codec-calibration", "on"]
    else:
        cmd += ["--per-codec-calibration", "off"]
    out = subprocess.check_output(cmd, text=True)
    return float(out.strip())


def butteraugli_pair(a: Path, b: Path):
    out = subprocess.check_output(
        [
            str(ZEN_METRICS),
            "score",
            "--metric",
            "butteraugli",
            "--reference", str(a),
            "--distorted", str(b),
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


def find_q_closest(scores_by_q, target):
    best_q = None
    best_s = None
    for q, s in scores_by_q.items():
        if best_q is None or abs(s - target) < abs(best_s - target):
            best_q = q
            best_s = s
    return best_q, best_s


def score_curve(ref_path, image_id, codec, calibrated):
    """Score every q for one (image, codec) combination. Returns dict q -> score."""
    scores = {}
    for q in Q_GRID:
        dec = WORK_DIR_PRECOMPUTED / f"{image_id}__{codec}_q{q:03d}.png"
        if not dec.exists():
            print(f"  WARN: missing {dec}", flush=True)
            continue
        scores[q] = zensim_score(Path(ref_path), dec, codec, calibrated)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = SELECTED_IMAGES[: args.n_images]

    print(f"# cross-codec calibrated eval start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    print(f"# {len(images)} images × {len(TARGETS)} targets × {len(CODECS)} codecs × 2 modes (raw + calibrated)", flush=True)

    # Phase 1: score every (image, codec, q) under both modes once
    # (curve cache).
    curves = {}  # (image_id, codec, mode) -> {q: score}
    for img_path in images:
        image_id = Path(img_path).stem
        for codec in CODECS:
            for mode in ("raw", "calibrated"):
                cal = (mode == "calibrated")
                if (image_id, codec, mode) in curves:
                    continue
                print(f"  scoring {image_id}/{codec}/{mode}...", flush=True)
                curves[(image_id, codec, mode)] = score_curve(img_path, image_id, codec, cal)

    # Phase 2: for each target T, mode, image: find q closest for each codec.
    rows = []
    summary = {}  # (mode, target) -> list of mean_butter, etc.
    for mode in ("raw", "calibrated"):
        for target in TARGETS:
            rows_for_pt = []
            for img_path in images:
                image_id = Path(img_path).stem
                picked = {}  # codec -> (q, score, dec_path)
                for codec in CODECS:
                    sc = curves[(image_id, codec, mode)]
                    q, s = find_q_closest(sc, target)
                    dec = WORK_DIR_PRECOMPUTED / f"{image_id}__{codec}_q{q:03d}.png"
                    picked[codec] = (q, s, dec)
                # Pairwise butteraugli
                pair_bmax = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        ci, cj = CODECS[i], CODECS[j]
                        bmax, _bp3 = butteraugli_pair(picked[ci][2], picked[cj][2])
                        if bmax is not None:
                            pair_bmax.append(bmax)
                mean_max = sum(pair_bmax) / len(pair_bmax) if pair_bmax else float("nan")
                max_max = max(pair_bmax) if pair_bmax else float("nan")
                achieved = [picked[c][1] for c in CODECS]
                score_spread = max(achieved) - min(achieved)
                dist_from_target = sum(abs(a - target) for a in achieved) / len(achieved)
                row = {
                    "mode": mode,
                    "target": target,
                    "image_id": image_id,
                    **{f"{c}_q": picked[c][0] for c in CODECS},
                    **{f"{c}_score": picked[c][1] for c in CODECS},
                    "score_spread": score_spread,
                    "dist_from_target": dist_from_target,
                    "pair_butter_max_mean": mean_max,
                    "pair_butter_max_worst": max_max,
                }
                rows.append(row)
                rows_for_pt.append(row)
                print(
                    f"  {mode}/{image_id}/T={target}: "
                    f"jpeg(q={picked['jpeg'][0]:>2} s={picked['jpeg'][1]:>6.2f}) "
                    f"webp(q={picked['webp'][0]:>2} s={picked['webp'][1]:>6.2f}) "
                    f"avif(q={picked['avif'][0]:>2} s={picked['avif'][1]:>6.2f}) "
                    f"butter_max(mean={mean_max:.3f} worst={max_max:.3f}) "
                    f"spread={score_spread:.2f}",
                    flush=True,
                )
            valid_b = [r["pair_butter_max_mean"] for r in rows_for_pt if r["pair_butter_max_mean"] == r["pair_butter_max_mean"]]
            valid_sp = [r["score_spread"] for r in rows_for_pt]
            valid_d = [r["dist_from_target"] for r in rows_for_pt]
            summary[(mode, target)] = {
                "mean_butter_max": sum(valid_b)/len(valid_b) if valid_b else float("nan"),
                "mean_score_spread": sum(valid_sp)/len(valid_sp),
                "mean_dist_from_target": sum(valid_d)/len(valid_d),
                "n": len(rows_for_pt),
            }

    # Write raw TSV
    raw_tsv = out_dir / "raw_2026-05-19.tsv"
    if rows:
        cols = list(rows[0].keys())
        with open(raw_tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"raw TSV: {raw_tsv}")

    summary_json = out_dir / "summary_2026-05-19.json"
    out_summary = {f"{k[0]}@T{k[1]}": v for k, v in summary.items()}
    with open(summary_json, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"summary JSON: {summary_json}")

    print("\n## mean pair_butter_max — Raw Tuner vs Calibrated Tuner (lower = more JND-consistent)\n")
    print("| Mode | " + " | ".join(f"T={t}" for t in TARGETS) + " |")
    print("|---|" + "|".join(["---:"] * len(TARGETS)) + "|")
    for mode in ("raw", "calibrated"):
        cells = []
        for t in TARGETS:
            v = summary.get((mode, t), {}).get("mean_butter_max", float("nan"))
            cells.append(f"{v:.2f}" if v == v else "n/a")
        print(f"| {mode} | " + " | ".join(cells) + " |")

    print("\n## mean score_spread (codec spread of achieved scores)\n")
    print("| Mode | " + " | ".join(f"T={t}" for t in TARGETS) + " |")
    print("|---|" + "|".join(["---:"] * len(TARGETS)) + "|")
    for mode in ("raw", "calibrated"):
        cells = []
        for t in TARGETS:
            v = summary.get((mode, t), {}).get("mean_score_spread", float("nan"))
            cells.append(f"{v:.2f}" if v == v else "n/a")
        print(f"| {mode} | " + " | ".join(cells) + " |")

    print("\n## mean dist_from_target (closer = reachable)\n")
    print("| Mode | " + " | ".join(f"T={t}" for t in TARGETS) + " |")
    print("|---|" + "|".join(["---:"] * len(TARGETS)) + "|")
    for mode in ("raw", "calibrated"):
        cells = []
        for t in TARGETS:
            v = summary.get((mode, t), {}).get("mean_dist_from_target", float("nan"))
            cells.append(f"{v:.2f}" if v == v else "n/a")
        print(f"| {mode} | " + " | ".join(cells) + " |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
