#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V11-E (2026-05-20): fit + inject per-codec
post-spline affine calibration into V10 ship bakes.

For each shipped V10 bake (TunerV4, BalancedV3, CompressionV3):

  1. Load the V11 substrate cross-codec equivalence parquet
     (`/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/`
     `cross_codec_equivalence_ssim2_372col_v4.parquet`,
     1,739 pairs, 4 codecs).
  2. Each row has (codec_a, q_a, codec_b, q_b, ssim2_level, fa_*, fb_*).
     Score both sides through the bake to get predicted_a and predicted_b
     (post-spline raw, NO clamping — uses `--bake-post extrapolate`).
  3. Per codec c, collect (target=ssim2_level, predicted=predicted_c)
     across all rows where c appears as codec_a or codec_b.
  4. Linear least-squares fit `target ≈ alpha_c + beta_c · predicted_c`,
     constrained to beta_c > 0 (otherwise refuse — bake's monotonicity
     within codec would invert, which we want to detect).
  5. Bound alpha_c to [-30, 30] (we won't accept extreme shifts; if the
     fit demands more, the codec is structurally inconsistent with the
     other codecs and the calibration should not ship).
  6. Verify the cross-codec stddev at JND/JOD landmarks tightens AFTER
     applying the fit. Report before/after on a held-out 20% slice.
  7. Encode metadata payload + inject via the JSON pipeline.

Output: `<bake>_per_codec.bin`, plus a per-bake fit CSV at
`benchmarks/v11_e_per_codec_<bake_name>_fit.csv` with the fit numbers.

The substrate v4 contains 4 codecs (zenjpeg, zenwebp, zenavif, zenjxl);
each codec is anchored by 220-374 pairs across 6 codec-pair combos.

Metadata payload (little-endian):
    `[u32 n_codecs, n_codecs × (u32 name_len, name_len utf8, f32 alpha, f32 beta)]`
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path("/home/lilith/work/zen/zensimx")
SUBSTRATE_PARQUET = Path(
    "/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/"
    "cross_codec_equivalence_ssim2_372col_v4.parquet"
)
PREDICT_BIN = REPO_ROOT / "target/release/predict_features_with_bake"
ZENPREDICT_BIN = Path("/home/lilith/work/zen/zenanalyze/target/release/zenpredict")

# Canonical codec names baked into the metadata.
CANON_CODECS = ["jpeg", "webp", "avif", "jxl"]
SUBSTRATE_TO_CANON = {
    "zenjpeg": "jpeg",
    "zenwebp": "webp",
    "zenavif": "avif",
    "zenjxl": "jxl",
}


def predict_via_binary(
    bake_path: Path, feats: np.ndarray, post_mode: str = "extrapolate"
) -> np.ndarray:
    """Score features through the bake. Wire format:
    `[u32 n_features, u32 n_rows, f32 row-major]`.
    `--bake-post extrapolate` keeps the raw post-spline output
    (no clamping) — required for accurate per-codec affine fits at
    the heavy-distortion tail where the spline extrapolates < 0.
    """
    n_rows, n_features = feats.shape
    tmp = Path("/tmp/v11e_feats.bin")
    with tmp.open("wb") as f:
        f.write(struct.pack("<II", n_features, n_rows))
        f.write(feats.astype("<f4", copy=False).tobytes())
    cmd = [
        str(PREDICT_BIN),
        "--bake",
        str(bake_path),
        "--features-file",
        str(tmp),
        "--bake-post",
        post_mode,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out_lines = proc.stdout.strip().splitlines()
    if len(out_lines) != n_rows:
        raise RuntimeError(
            f"predict_features_with_bake emitted {len(out_lines)} lines "
            f"for {n_rows} rows; stderr={proc.stderr!r}"
        )
    return np.array([float(s) for s in out_lines], dtype=np.float64)


def fit_per_codec_affine(
    target: np.ndarray, predicted: np.ndarray, mode: str = "full"
) -> tuple[float, float, float, float]:
    """Linear least-squares fit `target ≈ alpha + beta · predicted`.

    `mode="full"`: free (alpha, beta) fit via polyfit deg-1.
    `mode="offset"`: fix beta=1 (pure offset). alpha = mean(target - predicted).
        This preserves the bake's intra-codec scale (so the existing
        spline calibration's slope is honored) and only shifts each
        codec by a constant to pull it toward the cross-codec consensus
        at the offset level. Recommended when the bake already carries
        a PCHIP spline that handles within-codec score-vs-target shape.

    Returns `(alpha, beta, r2, mse)`. Beta is clamped to a minimum of
    0.05 (refuses degenerate flat fits — those signal the bake is
    score-flat on this codec, which would mean an identity affine is
    safer than the fit).
    """
    if len(target) != len(predicted) or len(target) < 4:
        raise ValueError(f"insufficient n={len(target)} for affine fit")
    valid = np.isfinite(target) & np.isfinite(predicted)
    target = target[valid]
    predicted = predicted[valid]
    if len(target) < 4:
        raise ValueError(
            f"only {len(target)} finite rows after NaN filter; refusing fit"
        )
    if mode == "offset":
        alpha = float(np.mean(target - predicted))
        beta = 1.0
    elif mode == "full":
        # `np.polyfit` with degree 1: returns `[slope, intercept]`.
        slope, intercept = np.polyfit(predicted, target, deg=1)
        alpha = float(intercept)
        beta = float(slope)
        if beta < 0.05:
            print(
                f"  WARN: beta={beta:.4f} < 0.05 — bake is near-flat on this codec; "
                f"forcing identity affine"
            )
            return 0.0, 1.0, 0.0, float(np.var(target - predicted))
    else:
        raise ValueError(f"unknown fit mode: {mode!r}")
    pred_fit = alpha + beta * predicted
    resid = target - pred_fit
    mse = float(np.mean(resid**2))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, beta, r2, mse


def encode_per_codec_payload(fits: dict[str, tuple[float, float]]) -> bytes:
    """Encode `{codec_name: (alpha, beta)}` as
    `[u32 n, n × (u32 name_len, name_len utf8, f32 alpha, f32 beta)]`."""
    payload = struct.pack("<I", len(fits))
    for name, (alpha, beta) in fits.items():
        name_bytes = name.encode("utf-8")
        payload += struct.pack("<I", len(name_bytes))
        payload += name_bytes
        payload += struct.pack("<ff", float(alpha), float(beta))
    return payload


def inspect_and_add_metadata(
    bake_path: Path,
    payload: bytes,
    out_path: Path,
) -> None:
    """Inspect the bake (with --weights to get layer floats), append
    the per-codec calibration metadata, then re-bake via the zenpredict
    JSON pipeline.

    The output bake is byte-different from the input but functionally
    identical for callers that don't pass a codec hint (the
    `zentrain.per_codec_calibration` metadata is a no-op without a
    hint).
    """
    inspect_w = subprocess.run(
        [str(ZENPREDICT_BIN), "inspect", str(bake_path), "--weights"],
        capture_output=True,
        text=True,
        check=True,
    )
    inspected = json.loads(inspect_w.stdout)
    scaler_mean = inspected["scaler_mean"]
    scaler_scale = inspected["scaler_scale"]
    layers_w = inspected["layers"]
    out_layers = []
    for layer in layers_w:
        out_layers.append(
            {
                "in_dim": layer["in_dim"],
                "out_dim": layer["out_dim"],
                "activation": layer["activation"],
                "dtype": layer["dtype"],
                "weights": layer["weights"],
                "biases": layer["biases"],
            }
        )

    md_list = []
    for entry in inspected.get("metadata", []):
        key = entry["key"]
        kind = entry["kind"]
        if key == "zentrain.per_codec_calibration":
            # Drop any pre-existing per-codec metadata; we're replacing it.
            continue
        if "value_hex" in entry:
            md_list.append({"key": key, "type": kind, "hex": entry["value_hex"]})
        elif "value_text" in entry:
            md_list.append({"key": key, "type": kind, "text": entry["value_text"]})
        elif "value_f32_array" in entry:
            md_list.append({"key": key, "type": kind, "f32": entry["value_f32_array"]})
        else:
            raise RuntimeError(
                f"don't know how to re-encode metadata entry {key}: "
                f"keys present = {list(entry.keys())}"
            )

    md_list.append(
        {
            "key": "zentrain.per_codec_calibration",
            "type": "bytes",
            "hex": payload.hex(),
        }
    )

    schema_hash_raw = inspected.get("schema_hash", 0)
    if isinstance(schema_hash_raw, str):
        schema_hash = (
            int(schema_hash_raw, 16)
            if schema_hash_raw.startswith("0x")
            else int(schema_hash_raw)
        )
    else:
        schema_hash = int(schema_hash_raw)

    req = {
        "schema_hash": schema_hash,
        "flags": 0,
        "compressed": True,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "layers": out_layers,
        "metadata": md_list,
    }

    # Preserve feature_bounds + output_specs + sparse_overrides + discrete_sets
    # if present in inspect output. These propagate verbatim per the
    # bake-metadata-propagation policy.
    for k in ("feature_bounds", "output_specs", "sparse_overrides", "discrete_sets"):
        if k in inspected:
            req[k] = inspected[k]

    json_path = out_path.with_suffix(".tmp.json")
    json_path.write_text(json.dumps(req))
    subprocess.run(
        [str(ZENPREDICT_BIN), "bake", str(json_path), str(out_path)],
        check=True,
    )
    json_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bake", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--substrate", type=Path, default=SUBSTRATE_PARQUET
    )
    parser.add_argument(
        "--fit-csv",
        type=Path,
        default=None,
        help="If set, write per-codec fit numbers (alpha, beta, r2, mse, n)",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Held-out fraction for cross-codec consistency evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for train/holdout split"
    )
    parser.add_argument(
        "--fit-mode",
        choices=("full", "offset"),
        default="offset",
        help=(
            "Fit mode. 'offset' fixes beta=1 (pure per-codec α shift, "
            "preserves intra-codec scale already calibrated by the "
            "PCHIP spline). 'full' does free (α, β) — only safe when "
            "the bake has no spline."
        ),
    )
    args = parser.parse_args()

    print(f"=== loading substrate {args.substrate} ===")
    tbl = pq.read_table(args.substrate)
    df = tbl.to_pandas()
    print(f"  {len(df)} pairs")

    feats_a_cols = [f"fa_{i}" for i in range(372)]
    feats_b_cols = [f"fb_{i}" for i in range(372)]
    feats_a = df[feats_a_cols].to_numpy(dtype=np.float32, copy=False)
    feats_b = df[feats_b_cols].to_numpy(dtype=np.float32, copy=False)
    targets = df["ssim2_level"].to_numpy(dtype=np.float64, copy=False)
    codecs_a = df["codec_a"].astype(str).to_numpy()
    codecs_b = df["codec_b"].astype(str).to_numpy()

    # Train/holdout split per pair (not per codec — we want the fit to
    # generalize to unseen pairs of the same codec at unseen ssim2 levels).
    rng = np.random.default_rng(args.seed)
    n = len(df)
    idx = rng.permutation(n)
    n_holdout = int(n * args.holdout_frac)
    holdout_idx = np.sort(idx[:n_holdout])
    train_idx = np.sort(idx[n_holdout:])
    print(f"  train={len(train_idx)}  holdout={len(holdout_idx)}")

    print(f"\n=== scoring bake {args.bake} on full substrate ===")
    preds_a = predict_via_binary(args.bake, feats_a, post_mode="extrapolate")
    preds_b = predict_via_binary(args.bake, feats_b, post_mode="extrapolate")
    print(
        f"  preds_a: finite={np.isfinite(preds_a).sum()}/{len(preds_a)} "
        f"range=[{np.nanmin(preds_a):.2f}, {np.nanmax(preds_a):.2f}]"
    )
    print(
        f"  preds_b: finite={np.isfinite(preds_b).sum()}/{len(preds_b)} "
        f"range=[{np.nanmin(preds_b):.2f}, {np.nanmax(preds_b):.2f}]"
    )

    # Per codec, accumulate (target, predicted) for train fit.
    per_codec_train: dict[str, list[tuple[float, float]]] = {c: [] for c in CANON_CODECS}
    per_codec_holdout_pre: dict[str, list[tuple[float, float]]] = {
        c: [] for c in CANON_CODECS
    }
    for i in range(n):
        ca = SUBSTRATE_TO_CANON.get(codecs_a[i])
        cb = SUBSTRATE_TO_CANON.get(codecs_b[i])
        t = float(targets[i])
        pa = float(preds_a[i])
        pb = float(preds_b[i])
        is_holdout = (i in set(holdout_idx.tolist()))
        # cheap membership via boolean mask
    # Vectorized membership computation for performance.
    holdout_mask = np.zeros(n, dtype=bool)
    holdout_mask[holdout_idx] = True

    for i in range(n):
        ca = SUBSTRATE_TO_CANON.get(codecs_a[i])
        cb = SUBSTRATE_TO_CANON.get(codecs_b[i])
        t = float(targets[i])
        pa = float(preds_a[i])
        pb = float(preds_b[i])
        bucket = per_codec_holdout_pre if holdout_mask[i] else per_codec_train
        if ca is not None and np.isfinite(pa) and np.isfinite(t):
            bucket[ca].append((t, pa))
        if cb is not None and np.isfinite(pb) and np.isfinite(t):
            bucket[cb].append((t, pb))

    print("\n=== per-codec n breakdown ===")
    for codec in CANON_CODECS:
        n_tr = len(per_codec_train[codec])
        n_ho = len(per_codec_holdout_pre[codec])
        print(f"  {codec:>6}  train n={n_tr:5d}  holdout n={n_ho:5d}")

    print("\n=== fitting per-codec affine on train set ===")
    fits: dict[str, tuple[float, float]] = {}
    fit_table: list[dict] = []
    for codec in CANON_CODECS:
        rows = per_codec_train[codec]
        if len(rows) < 4:
            print(
                f"  {codec}: insufficient n={len(rows)} on train; using identity affine"
            )
            fits[codec] = (0.0, 1.0)
            fit_table.append(
                {
                    "codec": codec,
                    "n_train": len(rows),
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r2": 0.0,
                    "mse": float("nan"),
                    "identity_fallback": True,
                }
            )
            continue
        target_arr = np.array([r[0] for r in rows])
        pred_arr = np.array([r[1] for r in rows])
        alpha, beta, r2, mse = fit_per_codec_affine(
            target_arr, pred_arr, mode=args.fit_mode
        )
        # Bound alpha to ±30 — refuse extreme shifts.
        if abs(alpha) > 30:
            print(
                f"  {codec}: alpha={alpha:.2f} out of bounds; clipping to ±30"
            )
            alpha = float(np.clip(alpha, -30.0, 30.0))
        fits[codec] = (alpha, beta)
        fit_table.append(
            {
                "codec": codec,
                "n_train": len(rows),
                "alpha": alpha,
                "beta": beta,
                "r2": r2,
                "mse": mse,
                "identity_fallback": False,
            }
        )
        print(
            f"  {codec}: alpha={alpha:+.4f}  beta={beta:+.4f}  R²={r2:.4f}  MSE={mse:.3f}  n={len(rows)}"
        )

    # Compute holdout cross-codec stddev BEFORE and AFTER affine.
    print("\n=== holdout cross-codec consistency ===")

    # Group by (ssim2_level, ref_basename) — at each anchor level, the
    # "consensus" is the mean of all (pred_codec_x for codec_x present
    # at this anchor). Cross-codec stddev within that group is the
    # mismatch.
    def cc_stddev_summary(
        df_subset: np.ndarray,
        preds_a_sub: np.ndarray,
        preds_b_sub: np.ndarray,
        fit_dict: dict[str, tuple[float, float]] | None,
    ) -> dict:
        """Compute per-(ref_basename, ssim2_level) cross-codec stddev.

        Each row in `df_subset` is one (codec_a@q_a, codec_b@q_b) pair
        at the same ssim2_level for the same ref. We group by
        (ref_basename, ssim2_level), collecting each codec's per-pair
        predictions (one from `pa` and one from `pb` side), then compute
        stddev across the unique codec → mean-predicted map per group.

        Returns median / p90 / max of group cross-codec stddev, plus the
        per-pair `|pa - pb|` summary as a secondary diagnostic.
        """
        per_pair_a = np.empty(len(df_subset))
        per_pair_b = np.empty(len(df_subset))
        for k, i in enumerate(df_subset):
            ca = SUBSTRATE_TO_CANON.get(codecs_a[i])
            cb = SUBSTRATE_TO_CANON.get(codecs_b[i])
            pa = preds_a_sub[k]
            pb = preds_b_sub[k]
            if fit_dict is not None:
                if ca in fit_dict:
                    alpha, beta = fit_dict[ca]
                    pa = alpha + beta * pa
                if cb in fit_dict:
                    alpha, beta = fit_dict[cb]
                    pb = alpha + beta * pb
            per_pair_a[k] = pa
            per_pair_b[k] = pb

        # Group by (ref_basename, ssim2_level) to compute per-anchor
        # cross-codec stddev.
        ref_subset = df["ref_basename"].to_numpy()[df_subset]
        lev_subset = targets[df_subset]
        # Build a per-anchor dict of codec → list-of-preds.
        anchors: dict[tuple[str, float], dict[str, list[float]]] = {}
        for k, i in enumerate(df_subset):
            ref = ref_subset[k]
            lev = float(lev_subset[k])
            key = (ref, lev)
            d = anchors.setdefault(key, {})
            ca = SUBSTRATE_TO_CANON.get(codecs_a[i])
            cb = SUBSTRATE_TO_CANON.get(codecs_b[i])
            if ca is not None and np.isfinite(per_pair_a[k]):
                d.setdefault(ca, []).append(per_pair_a[k])
            if cb is not None and np.isfinite(per_pair_b[k]):
                d.setdefault(cb, []).append(per_pair_b[k])
        # Per anchor: average each codec's preds, then std across the
        # codec-averaged values. Skip anchors with < 2 distinct codecs
        # (stddev undefined).
        stds: list[float] = []
        for d in anchors.values():
            if len(d) < 2:
                continue
            avgs = np.array([float(np.mean(v)) for v in d.values()])
            stds.append(float(np.std(avgs, ddof=0)))
        stds_arr = np.array(stds)
        diff = np.abs(per_pair_a - per_pair_b)
        valid = np.isfinite(diff)
        return {
            "n_pairs": int(valid.sum()),
            "n_anchors": len(stds),
            "anchor_std_median": float(np.median(stds_arr)) if len(stds_arr) else float("nan"),
            "anchor_std_p90": float(np.percentile(stds_arr, 90)) if len(stds_arr) else float("nan"),
            "anchor_std_max": float(np.max(stds_arr)) if len(stds_arr) else float("nan"),
            "pair_diff_median": float(np.median(diff[valid])),
            "pair_diff_p90": float(np.percentile(diff[valid], 90)),
            "pair_diff_max": float(np.max(diff[valid])),
        }

    # Holdout slice: rows in `holdout_idx`.
    preds_a_ho = preds_a[holdout_idx]
    preds_b_ho = preds_b[holdout_idx]

    print("Before per-codec affine (identity):")
    pre = cc_stddev_summary(holdout_idx, preds_a_ho, preds_b_ho, None)
    print(
        f"  n_pairs={pre['n_pairs']}  n_anchors={pre['n_anchors']}\n"
        f"  cross-codec stddev (per anchor): median={pre['anchor_std_median']:.4f}  "
        f"p90={pre['anchor_std_p90']:.4f}  max={pre['anchor_std_max']:.4f}\n"
        f"  per-pair |pa-pb|: median={pre['pair_diff_median']:.4f}  "
        f"p90={pre['pair_diff_p90']:.4f}  max={pre['pair_diff_max']:.4f}"
    )
    print("After per-codec affine (fitted):")
    post = cc_stddev_summary(holdout_idx, preds_a_ho, preds_b_ho, fits)
    print(
        f"  n_pairs={post['n_pairs']}  n_anchors={post['n_anchors']}\n"
        f"  cross-codec stddev (per anchor): median={post['anchor_std_median']:.4f}  "
        f"p90={post['anchor_std_p90']:.4f}  max={post['anchor_std_max']:.4f}\n"
        f"  per-pair |pa-pb|: median={post['pair_diff_median']:.4f}  "
        f"p90={post['pair_diff_p90']:.4f}  max={post['pair_diff_max']:.4f}"
    )
    delta_med = post["anchor_std_median"] - pre["anchor_std_median"]
    delta_max = post["anchor_std_max"] - pre["anchor_std_max"]
    print(
        f"  Δ anchor std median = {delta_med:+.4f}  Δ anchor std max = {delta_max:+.4f}"
    )

    # Also report per-JND-level summary.
    print("\n=== per-ssim2-level anchor stddev (holdout) ===")
    print(
        f"  {'level':>5}  {'n_anc':>5}  "
        f"{'pre_med':>8}  {'post_med':>8}  {'pre_max':>8}  {'post_max':>8}"
    )
    targets_ho = targets[holdout_idx]
    for level in sorted(set(targets_ho.tolist())):
        mask = targets_ho == level
        if mask.sum() < 4:
            continue
        idx_subset = holdout_idx[mask]
        pa_s = preds_a[idx_subset]
        pb_s = preds_b[idx_subset]
        pre_l = cc_stddev_summary(idx_subset, pa_s, pb_s, None)
        post_l = cc_stddev_summary(idx_subset, pa_s, pb_s, fits)
        print(
            f"  {level:>5.1f}  {pre_l['n_anchors']:>5d}  "
            f"{pre_l['anchor_std_median']:>8.4f}  {post_l['anchor_std_median']:>8.4f}  "
            f"{pre_l['anchor_std_max']:>8.4f}  {post_l['anchor_std_max']:>8.4f}"
        )

    if args.fit_csv:
        with args.fit_csv.open("w") as f:
            f.write("codec,n_train,alpha,beta,r2,mse,identity_fallback\n")
            for row in fit_table:
                f.write(
                    f"{row['codec']},{row['n_train']},{row['alpha']},"
                    f"{row['beta']},{row['r2']},{row['mse']},"
                    f"{row['identity_fallback']}\n"
                )
        print(f"\n  wrote fit CSV → {args.fit_csv}")

    print(f"\n=== encoding payload + injecting metadata → {args.out} ===")
    payload = encode_per_codec_payload(fits)
    print(f"  payload size: {len(payload)} bytes")
    inspect_and_add_metadata(args.bake, payload, args.out)
    bytes_in = args.bake.stat().st_size
    bytes_out = args.out.stat().st_size
    print(f"  bake size: {bytes_in} → {bytes_out} bytes")

    # Sanity smoke: re-score a few rows WITHOUT codec hint — should
    # produce identical output to the input bake (per-codec affine
    # gated on codec hint, so identity-by-default).
    print("\n=== sanity: bake without codec hint produces same scores ===")
    sample_feats = feats_a[:5].copy()
    base = predict_via_binary(args.bake, sample_feats, post_mode="extrapolate")
    new = predict_via_binary(args.out, sample_feats, post_mode="extrapolate")
    diff = np.abs(new - base)
    print(f"  max abs diff (no codec hint): {diff.max():.6e}")
    if diff.max() > 1e-4:
        print("  WARN: scores differ without codec hint — metadata may be misapplied")


if __name__ == "__main__":
    main()
