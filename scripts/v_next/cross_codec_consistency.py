#!/usr/bin/env python3
"""Cross-codec consistency eval for PreviewV0_5Tuner (2026-05-18).

EVAL-ACCEL refactor (2026-05-19): the binary-search-over-q step
no longer live-encodes + extracts features per `measure(q)` call.
Instead it consults pre-extracted 372-feat parquets at
`/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`
(produced by EXP-CROSS-CODEC-METRIC #156) for every (image, codec, q)
tuple. The bake forward pass runs via a Rust binary
`predict_features_with_bake` that takes a packed f32 feature buffer
and emits one score per row. Only the convergence q is re-encoded
via PIL + scored once via zen-metrics butter_pnorm3.

Old wall: ~6 min/bake (480 measure calls × 5-15 s each).
New wall: ~30 s/bake (1 cached feature read + ≤8 Rust forward calls
per (image, codec) ≤ 480 forwards total, batched into one binary
invocation per binary-search step; then 60 butter encode/score calls).

For each of K source images and a target zensim score T:
  1. For each codec in {JPEG, WebP, AVIF}, binary-search for the q
     value whose zensim score (under the eval bake, dispatched from
     the bake's metadata heads) is closest to T. The feature rows are
     read from the codec parquet at the standard 19-q grid
     {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
      85, 90, 95}.
  2. PIL-encode the source at the chosen q, decode to PNG.
  3. Compute pairwise butteraugli (max + p-norm-3) between the 3
     decoded PNGs via the zen-metrics CLI. Pairwise mean should be
     SMALL — codecs landed at perceptually similar quality.

Output: a TSV with per-image rows (image_id, jpeg_q, webp_q, avif_q,
zensim_jpeg, zensim_webp, zensim_avif, pairwise_butter_max_mean,
pairwise_butter_p3_mean).
"""

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

BUTTER_PARQUET_ROOT = Path("/mnt/v/zen/picker-training/2026-05-19/butter")
SOURCE_ROOT = Path("/mnt/v/input/zensim/sources")
OUT_DIR_DEFAULT = Path("/mnt/v/output/zensim/exp_tuner_2026-05-18/cross_codec")

# Standard 19-q grid present in the butter parquets.
Q_GRID = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
N_FEATURES = 372

# Codec → parquet basename + PIL save args.
CODECS = {
    "jpeg": {
        "parquet": "zenjpeg.parquet",
        "ext": ".jpg",
        "format": "JPEG",
        "save_kwargs": {
            "quality": None,  # filled at encode time
            "subsampling": 2,
            "optimize": False,
            "progressive": False,
        },
    },
    "webp": {
        "parquet": "zenwebp.parquet",
        "ext": ".webp",
        "format": "WEBP",
        "save_kwargs": {"quality": None, "method": 4},
    },
    "avif": {
        "parquet": "zenavif.parquet",
        "ext": ".avif",
        "format": "AVIF",
        "save_kwargs": {"quality": None, "speed": 6},
    },
}


# ---------------------------------------------------------------------------
# Feature-parquet cache
# ---------------------------------------------------------------------------


def load_codec_features(parquet_path: Path):
    """Return a dict (ref_basename → {q → list[float] of N_FEATURES}).

    Reads only the columns we need (ref_basename, q, f0..f371). The
    resulting structure lives in memory for the duration of the run;
    each codec parquet is ~35 MB on disk and decodes to ~60 MB of
    f32 features.
    """
    print(f"loading features from {parquet_path} …", file=sys.stderr, flush=True)
    cols = ["ref_basename", "q"] + [f"f{i}" for i in range(N_FEATURES)]
    table = pq.read_table(parquet_path, columns=cols)
    pdf = table.to_pandas()
    feat_cols = [f"f{i}" for i in range(N_FEATURES)]
    by_image = {}
    for image_id, group in pdf.groupby("ref_basename"):
        per_q = {}
        for _, row in group.iterrows():
            q = int(row["q"])
            per_q[q] = [float(row[c]) for c in feat_cols]
        by_image[str(image_id)] = per_q
    print(
        f"  loaded {len(by_image)} images × ≤{len(Q_GRID)} q-rows × {N_FEATURES} features",
        file=sys.stderr,
        flush=True,
    )
    return by_image


# ---------------------------------------------------------------------------
# Score-from-features via Rust binary
# ---------------------------------------------------------------------------


def predict_zensim_from_features(features_rows, bake_path, bake_post, predict_tool):
    """Score N rows of features against the bake. Returns a list[float].

    `features_rows` is a list of feature vectors, each length N_FEATURES.
    """
    n_rows = len(features_rows)
    if n_rows == 0:
        return []
    n_features = len(features_rows[0])
    # Pack into the binary's expected layout: u32 n_features, u32 n_rows,
    # then n_rows * n_features f32 LE.
    buf = bytearray()
    buf += struct.pack("<II", n_features, n_rows)
    for row in features_rows:
        if len(row) != n_features:
            raise ValueError(
                f"feature row width mismatch: expected {n_features}, got {len(row)}"
            )
        buf += struct.pack(f"<{n_features}f", *row)
    with tempfile.NamedTemporaryFile(suffix=".features.bin", delete=False) as f:
        f.write(buf)
        feats_path = f.name
    try:
        out = subprocess.check_output(
            [
                predict_tool,
                "--bake",
                bake_path,
                "--bake-post",
                bake_post,
                "--features-file",
                feats_path,
            ],
            text=True,
        )
    finally:
        Path(feats_path).unlink(missing_ok=True)
    scores = [float(line) for line in out.strip().splitlines() if line.strip()]
    if len(scores) != n_rows:
        raise RuntimeError(
            f"predict_features_with_bake returned {len(scores)} scores; expected {n_rows}"
        )
    return scores


# ---------------------------------------------------------------------------
# Binary search on cached features
# ---------------------------------------------------------------------------


def binary_search_q_cached(
    target_score,
    per_q,
    bake_path,
    bake_post,
    predict_tool,
):
    """Find the q in Q_GRID whose cached features score closest to target.

    Returns (best_q, best_score). Score is the bake-with-post output.
    Pre-scores all 19 q values with one Rust call — far cheaper than
    8 binary-search steps × Python ↔ Rust round trips.
    """
    rows = []
    qs_present = []
    for q in Q_GRID:
        if q in per_q:
            rows.append(per_q[q])
            qs_present.append(q)
    if not rows:
        return None, None
    scores = predict_zensim_from_features(rows, bake_path, bake_post, predict_tool)
    best_q = qs_present[0]
    best_score = scores[0]
    best_err = abs(best_score - target_score)
    for q, s in zip(qs_present[1:], scores[1:]):
        err = abs(s - target_score)
        if err < best_err:
            best_err = err
            best_q = q
            best_score = s
    return best_q, best_score


# ---------------------------------------------------------------------------
# Encode at convergence q + butter
# ---------------------------------------------------------------------------


def encode_convergence(src_png, q, codec_name, codec_cfg, work_dir, image_id):
    enc_path = work_dir / f"{image_id}_{codec_name}_q{q:03d}{codec_cfg['ext']}"
    dec_path = work_dir / f"{image_id}_{codec_name}_q{q:03d}.png"
    if not dec_path.exists():
        img = Image.open(src_png).convert("RGB")
        kwargs = dict(codec_cfg["save_kwargs"])
        kwargs["quality"] = int(q)
        img.save(enc_path, codec_cfg["format"], **kwargs)
        Image.open(enc_path).convert("RGB").save(dec_path, "PNG")
    return dec_path


def butteraugli_pair(ref_png, dist_png, zen_metrics):
    """Return (max, p3) butteraugli for the pair."""
    out = subprocess.check_output(
        [
            zen_metrics,
            "score",
            "--metric",
            "butteraugli",
            "--reference",
            str(ref_png),
            "--distorted",
            str(dist_png),
        ],
        text=True,
    )
    bmax, bp3 = None, None
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--bake", required=True, help="bake bytes to use for zensim scoring")
    ap.add_argument(
        "--bake-post",
        default="clamp",
        help="post-processing mode for the bake (clamp|raw|mapped[:a,b])",
    )
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument(
        "--predict-tool",
        default="/home/lilith/work/zen/zensimx/target/release/predict_features_with_bake",
    )
    ap.add_argument(
        "--zen-metrics",
        default="/home/lilith/work/zen/zenmetrics/target/release/zenmetrics",
    )
    ap.add_argument(
        "--butter-parquet-root",
        default=str(BUTTER_PARQUET_ROOT),
        help="dir containing zenjpeg.parquet, zenwebp.parquet, zenavif.parquet",
    )
    ap.add_argument(
        "--source-root",
        default=str(SOURCE_ROOT),
        help="dir holding the source PNGs keyed by ref_basename",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default=None, help="encode/decode scratch dir")
    args = ap.parse_args()

    butter_root = Path(args.butter_parquet_root)
    source_root = Path(args.source_root)

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = OUT_DIR_DEFAULT / f"bake_{Path(args.bake).stem}_t{int(args.target)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {work_dir}", file=sys.stderr, flush=True)

    # 1) Load all 3 codec feature parquets.
    feat_by_codec = {}
    for codec_name, cfg in CODECS.items():
        path = butter_root / cfg["parquet"]
        feat_by_codec[codec_name] = load_codec_features(path)

    # 2) Pick the first N_IMAGES from the intersection of the 3 codecs'
    #    ref_basename sets. Sort lexicographically for determinism.
    common = (
        set(feat_by_codec["jpeg"])
        & set(feat_by_codec["webp"])
        & set(feat_by_codec["avif"])
    )
    images = sorted(common)[: args.n_images]
    print(f"selected {len(images)} images (cross-codec intersection)", file=sys.stderr, flush=True)

    rows = []
    for image_id in images:
        src_png = source_root / image_id
        if not src_png.exists():
            print(f"  SKIP {image_id}: source PNG not found at {src_png}", file=sys.stderr, flush=True)
            continue
        print(f"\nimage: {image_id}", file=sys.stderr, flush=True)
        results = {}
        for codec_name, cfg in CODECS.items():
            per_q = feat_by_codec[codec_name][image_id]
            q, s = binary_search_q_cached(
                args.target,
                per_q,
                args.bake,
                args.bake_post,
                args.predict_tool,
            )
            print(f"  {codec_name} q={q} zensim={s:.2f}", file=sys.stderr, flush=True)
            dec_path = encode_convergence(
                src_png,
                q,
                codec_name,
                cfg,
                work_dir,
                image_id.rsplit(".", 1)[0],
            )
            results[codec_name] = (q, s, dec_path)

        # 3) Pairwise butteraugli between the 3 decoded outputs.
        names = ["jpeg", "webp", "avif"]
        pair_b_max = []
        pair_b_p3 = []
        for i in range(3):
            for j in range(i + 1, 3):
                _, _, pi = results[names[i]]
                _, _, pj = results[names[j]]
                bmax, bp3 = butteraugli_pair(pi, pj, args.zen_metrics)
                print(
                    f"  butter[{names[i]} vs {names[j]}] max={bmax} p3={bp3}",
                    file=sys.stderr,
                    flush=True,
                )
                if bmax is not None:
                    pair_b_max.append(bmax)
                if bp3 is not None:
                    pair_b_p3.append(bp3)
        mean_max = sum(pair_b_max) / len(pair_b_max) if pair_b_max else float("nan")
        mean_p3 = sum(pair_b_p3) / len(pair_b_p3) if pair_b_p3 else float("nan")
        rows.append({
            "image_id": image_id,
            "target": args.target,
            "jpeg_q": results["jpeg"][0], "jpeg_zensim": results["jpeg"][1],
            "webp_q": results["webp"][0], "webp_zensim": results["webp"][1],
            "avif_q": results["avif"][0], "avif_zensim": results["avif"][1],
            "pairwise_butter_max_mean": mean_max,
            "pairwise_butter_p3_mean": mean_p3,
        })

    with open(args.out, "w") as f:
        if not rows:
            print("no rows", file=sys.stderr, flush=True)
            return 1
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {args.out}", file=sys.stderr, flush=True)
    pmax = [r["pairwise_butter_max_mean"] for r in rows if r["pairwise_butter_max_mean"] == r["pairwise_butter_max_mean"]]
    pp3 = [r["pairwise_butter_p3_mean"] for r in rows if r["pairwise_butter_p3_mean"] == r["pairwise_butter_p3_mean"]]
    if pmax:
        print(f"mean butter_max across images: {sum(pmax)/len(pmax):.3f}", file=sys.stderr, flush=True)
    if pp3:
        print(f"mean butter_p3  across images: {sum(pp3)/len(pp3):.3f}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
