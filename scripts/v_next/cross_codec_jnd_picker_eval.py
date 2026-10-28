#!/usr/bin/env python3
"""Cross-codec JND consistency eval using per-codec picker MLPs (2026-05-19).

For each (image, target T): the picker MLP predicts q given zenanalyze
features + T. We then look up the pre-encoded decoded PNG and compute
pairwise butteraugli between the 3 codecs' picked outputs.

Compares:
  - Raw Tuner (binary-searched q from the existing baseline eval)
  - Tuner + per-codec affine calibration (the calibrated baseline)
  - Per-codec picker MLPs (this work)

The pre-existing encode cache at
`/mnt/v/output/zensim/cross_codec_consistency_2026-05-19/work/`
holds 10 images × 3 codecs × 19 q values. We restrict to the subset
of images whose source has zenanalyze features in the canonical
features parquet at
`/mnt/v/zen/picker-training/2026-05-19/sources_zenanalyze_features.parquet`.

Output:
  - Picker-driven pairwise butter per (image, T) at OUT_DIR/picker_raw.tsv
  - Aggregated per-T table in OUT_DIR/picker_summary.json
  - Combined comparison table in OUT_DIR/comparison.md
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/lilith/work/zen/zensimx")
ZEN_METRICS = Path("/home/lilith/work/zen/zenmetrics/target/release/zenmetrics")
WORK_DIR = Path("/mnt/v/output/zensim/cross_codec_consistency_2026-05-19/work")
FEATURES_PARQUET = Path("/mnt/v/zen/picker-training/2026-05-19/sources_zenanalyze_features.parquet")
PICKER_INFER = ROOT / "target/release/zensim_picker_infer"
BAKES = {
    "jpeg": ROOT / "zensim-experimental/weights/picker_zenjpeg_2026-05-19.bin",
    "webp": ROOT / "zensim-experimental/weights/picker_zenwebp_2026-05-19.bin",
    "avif": ROOT / "zensim-experimental/weights/picker_zenavif_2026-05-19.bin",
}
OUT_DIR = Path("/mnt/v/output/zensim/picker_cross_codec_2026-05-19")
BASELINE_SUMMARY = Path("/mnt/v/output/zensim/per_codec_calibration_2026-05-19/eval/summary_2026-05-19.json")

# Eval image basenames (matching the existing encode cache subdirectory
# scheme: <basename>__<codec>_q<NNN>.png). We must restrict to images
# whose source has zenanalyze features in the canonical features
# parquet — see EVAL_IMAGES_WITH_FEATURES below.
EVAL_IMAGES_FULL = [
    "00b13be94a4867dd_1022x818_512sq",
    "0543403b53d39228_512sq",
    "1248582_512sq",
    "2866385_512sq",
    "3d4f8bdb5b3733d3_512sq",
    "8f0cf3087dd4497f_512sq",
    "c20c6d8b7bdd7059_512sq",
    "gen-chart__00028_s4059c457_512sq",
    "gen-chart__00059_s1bd0eb8a_512sq",
    "gen-chart__00125_s02bb2eae_512sq",
]

# Mapping from encode-cache basename → features parquet ref_basename.
# Several encode-cache images are size-suffix variants of source images
# in the features parquet (e.g. _1022x818_512sq → _512sq). We use the
# closest available variant.
ENCODE_TO_FEATURES_MAP = {
    "00b13be94a4867dd_1022x818_512sq": None,        # no exact match in features parquet
    "0543403b53d39228_512sq": "0543403b53d39228_512sq",
    "1248582_512sq": "1248582_512sq",
    "2866385_512sq": "2866385_512sq",
    "3d4f8bdb5b3733d3_512sq": "3d4f8bdb5b3733d3_512sq",
    "8f0cf3087dd4497f_512sq": "8f0cf3087dd4497f_512sq",
    "c20c6d8b7bdd7059_512sq": "c20c6d8b7bdd7059_512sq",
    "gen-chart__00028_s4059c457_512sq": None,       # gen-chart not in safesyn corpus features parquet
    "gen-chart__00059_s1bd0eb8a_512sq": None,
    "gen-chart__00125_s02bb2eae_512sq": None,
}

TARGETS = [30, 50, 63, 70, 80, 90]
CODECS = ["jpeg", "webp", "avif"]
Q_GRID = list(range(5, 100, 5))


def butter_pair(a: Path, b: Path):
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


def run_picker_inference(codec: str, refs_to_include: list, out_tsv: Path):
    """Invoke zensim_picker_infer for the given codec → predictions TSV."""
    # Write filter file.
    filter_path = OUT_DIR / f"refs_{codec}.txt"
    filter_path.write_text("\n".join(refs_to_include) + "\n")
    cmd = [
        str(PICKER_INFER),
        "--bake", str(BAKES[codec]),
        "--features", str(FEATURES_PARQUET),
        "--t-values", ",".join(str(t) for t in TARGETS),
        "--ref-basenames", str(filter_path),
        "--out", str(out_tsv),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:", r.stdout, file=sys.stderr)
        print("STDERR:", r.stderr, file=sys.stderr)
        raise RuntimeError(f"picker infer failed for {codec}")


def load_predictions(tsv: Path):
    """Return {(ref_basename, T): q_rounded}."""
    out = {}
    with open(tsv) as f:
        header = f.readline()  # skip
        for line in f:
            parts = line.rstrip().split("\t")
            ref = parts[0]
            t = float(parts[1])
            q_pred = float(parts[2])
            q_round = int(parts[3])
            out[(ref, int(t))] = q_round
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restrict eval to images that have features.
    eval_images = [
        (encode_name, feat_name)
        for encode_name, feat_name in ENCODE_TO_FEATURES_MAP.items()
        if feat_name is not None
    ]
    print(f"Eval images ({len(eval_images)}/{len(EVAL_IMAGES_FULL)} have features):")
    for e, f in eval_images:
        print(f"  encode={e}  features={f}")

    refs_needed = [feat_name + ".png" for _, feat_name in eval_images]

    # 1. Run picker inference per codec.
    preds = {}
    for codec in CODECS:
        codec_bake_name = "jpeg" if codec == "jpeg" else codec  # 1:1 map
        tsv = out_dir / f"picker_{codec}_preds.tsv"
        print(f"\n# running picker inference for {codec} ...", flush=True)
        run_picker_inference(codec, refs_needed, tsv)
        preds[codec] = load_predictions(tsv)
        print(f"  {codec}: {len(preds[codec])} predictions")

    # 2. For each (image, T): for each codec, get the picked q (rounded
    #    to Q_GRID), look up the decoded PNG, compute pairwise butter.
    rows = []
    summary = {}
    for target in TARGETS:
        print(f"\n# target T={target}", flush=True)
        per_t_rows = []
        for encode_name, feat_name in eval_images:
            results = {}  # codec -> (q, decoded_path)
            for codec in CODECS:
                q = preds[codec].get((feat_name + ".png", target))
                if q is None:
                    print(f"  WARN: no prediction for {feat_name} T={target} codec={codec}")
                    continue
                if q < 5:
                    q = 5
                if q > 95:
                    q = 95
                dec_path = WORK_DIR / f"{encode_name}__{codec}_q{q:03d}.png"
                if not dec_path.exists():
                    print(f"  WARN: missing decoded {dec_path}")
                    continue
                results[codec] = (q, dec_path)
            if len(results) != 3:
                print(f"  skip {encode_name}: only {len(results)}/3 codecs available")
                continue

            # Pairwise butter.
            pair_bmax = []
            pair_bp3 = []
            for i in range(len(CODECS)):
                for j in range(i + 1, len(CODECS)):
                    ci, cj = CODECS[i], CODECS[j]
                    bmax, bp3 = butter_pair(results[ci][1], results[cj][1])
                    pair_bmax.append((ci, cj, bmax))
                    if bp3 is not None:
                        pair_bp3.append(bp3)
            mean_bmax = sum(p[2] for p in pair_bmax) / len(pair_bmax)
            max_bmax = max(p[2] for p in pair_bmax)
            mean_bp3 = sum(pair_bp3) / len(pair_bp3) if pair_bp3 else float("nan")
            row = {
                "target": target,
                "image": encode_name,
                "ref_features": feat_name,
                "jpeg_q": results["jpeg"][0],
                "webp_q": results["webp"][0],
                "avif_q": results["avif"][0],
                "pair_butter_max_mean": mean_bmax,
                "pair_butter_max_worst": max_bmax,
                "pair_butter_pnorm3_mean": mean_bp3,
            }
            per_t_rows.append(row)
            rows.append(row)
            print(
                f"  {encode_name}: jpeg(q={row['jpeg_q']:>2}) webp(q={row['webp_q']:>2}) avif(q={row['avif_q']:>2}) "
                f"butter_max(mean={mean_bmax:.3f} worst={max_bmax:.3f})",
                flush=True,
            )
        # Per-T summary
        if per_t_rows:
            summary[f"picker@T{target}"] = {
                "mean_butter_max": sum(r["pair_butter_max_mean"] for r in per_t_rows) / len(per_t_rows),
                "mean_butter_pnorm3": sum(r["pair_butter_pnorm3_mean"] for r in per_t_rows if r["pair_butter_pnorm3_mean"] == r["pair_butter_pnorm3_mean"]) / max(1, sum(1 for r in per_t_rows if r["pair_butter_pnorm3_mean"] == r["pair_butter_pnorm3_mean"])),
                "n": len(per_t_rows),
            }

    # Raw TSV
    raw_tsv = out_dir / "picker_raw.tsv"
    if rows:
        cols = list(rows[0].keys())
        with open(raw_tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"\nraw TSV: {raw_tsv}", flush=True)

    # Summary JSON
    summary_json = out_dir / "picker_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary JSON: {summary_json}", flush=True)

    # Comparison table vs baseline (also re-aggregate baseline on the
    # 6-image subset to be apples-to-apples).
    baseline = json.loads(BASELINE_SUMMARY.read_text())

    # Re-aggregate the BASELINE on the same 6-image subset by reading
    # the existing cross_codec raw TSV.
    cross_raw_tsv = Path("/mnt/v/output/zensim/per_codec_calibration_2026-05-19/eval/raw_2026-05-19.tsv")
    baseline_subset = {}  # ("raw"/"calibrated", T) -> mean_butter_max on subset
    if cross_raw_tsv.exists():
        subset_features = set(e for e, f in eval_images)
        with open(cross_raw_tsv) as f:
            hdr = f.readline().rstrip().split("\t")
            for line in f:
                parts = line.rstrip().split("\t")
                d = dict(zip(hdr, parts))
                img = d.get("image_id", "")
                if img not in subset_features:
                    continue
                mode = d.get("mode", "")  # raw/calibrated
                t = int(d.get("target", "0"))
                key = (mode, t)
                if key not in baseline_subset:
                    baseline_subset[key] = []
                try:
                    baseline_subset[key].append(float(d["pair_butter_max_mean"]))
                except Exception:
                    pass
        baseline_subset = {k: (sum(v) / len(v) if v else float("nan")) for k, v in baseline_subset.items()}

    comparison_md = out_dir / "comparison.md"
    with open(comparison_md, "w") as f:
        f.write("# Cross-codec JND consistency: picker MLPs vs Tuner baselines\n\n")
        f.write(f"Eval images ({len(eval_images)} of {len(EVAL_IMAGES_FULL)} have features):\n\n")
        for e, fn in eval_images:
            f.write(f"- `{e}` (features: `{fn}`)\n")
        f.write("\n## Mean pairwise butter_max (lower = more consistent across codecs)\n\n")
        f.write("Baseline numbers in this table aggregate the SAME image subset as the picker work,\n")
        f.write("so the comparison is apples-to-apples.\n\n")
        f.write("| Target | Tuner raw (full 10-img) | Tuner +affine (full 10-img) | Tuner raw (6-img subset) | Tuner +affine (6-img subset) | Picker MLPs (6-img) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for t in TARGETS:
            raw_full = baseline.get(f"raw@T{t}", {}).get("mean_butter_max", float("nan"))
            cal_full = baseline.get(f"calibrated@T{t}", {}).get("mean_butter_max", float("nan"))
            raw_sub = baseline_subset.get(("raw", t), float("nan"))
            cal_sub = baseline_subset.get(("calibrated", t), float("nan"))
            pkr = summary.get(f"picker@T{t}", {}).get("mean_butter_max", float("nan"))
            f.write(
                f"| T={t} | {raw_full:.2f} | {cal_full:.2f} | {raw_sub:.2f} | {cal_sub:.2f} | **{pkr:.2f}** |\n"
            )
    print(f"comparison MD: {comparison_md}", flush=True)
    print("\n## Final comparison table\n")
    with open(comparison_md) as f:
        print(f.read())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
