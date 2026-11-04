#!/usr/bin/env python3
"""V_20 input-shaping greedy screen.

For each feature (228 / 300 / 372 columns depending on schema) × 7
FeatureTransform candidate families, computes |Pearson(transform(feat),
target)| and ranks by lift vs the Identity baseline. Yeo-Johnson was
added 2026-05-25 (task #214) as the modern "auto-fit the power
transform" default.

Per the V_20 design doc (`benchmarks/v0_20_v0_21_design_2026-05-14.md`),
training MLP variants for all 1600 (feature, transform) cells costs
~640 GPU-hours. This screen filters to ~50-200 candidates by Pearson
gap > 0.005, taking ~minutes of CPU.

## Inputs

Tabular features CSV (or parquet) with columns `ref_basename`,
`human_score` (or a `--target-column` override), `f0..f<N-1>`.
Defaults to the 218k+e1=340k clean safe-synthetic features at the
2026-05-07 path.

## Output

TSV with one row per (feature, best_transform) candidate where the
lift clears `--min-lift` (default 0.005). Sorted by lift desc.

Columns:
- feat_idx
- best_transform_token
- params_csv          (empty for non-parameterized variants)
- baseline_pearson    (|Pearson(feat, mos)|)
- transformed_pearson (best across the variant's param sweep)
- lift                (transformed − baseline)
- baseline_spearman   (rank-correlation sanity check)
- n_samples           (count of finite feat values)

## Param sweeps per variant

| Variant | Param schedule |
|---|---|
| clip_then_log1p | ε = [p5, p10, p25, p50, p75] of the feature distribution |
| winsor_p99 | pairs: (p1, p99), (p5, p95), (p10, p90), (p25, p75) |
| quantile_bins | one 8-bin config from feature distribution percentiles |
| yeo_johnson | MLE-fit λ ∈ [-2, 2] per feature on the marginal distribution |

Non-parameterized variants (identity / log / log1p / signed_log1p /
signed_sqrt / signed_cbrt) have a single config each.

Identity baseline is always included so per-feature winners can stay
Identity if no transform helps.

## Yeo-Johnson (task #214)

YJ is parameterized by a single λ fit per-feature via maximum-likelihood
on the marginal training distribution. Defined on the full real line
(handles negative AND positive inputs); λ=1 is identity-shifted,
λ=0 is `ln(1+x)` for positive inputs, λ=2 is `-ln(1−x)` for negative.

The screen fits λ in-process using a 1D golden-section search over
λ ∈ [-2, 2] — matches scipy's `yeojohnson` to ~5 decimal places.
For more control or out-of-band fitting, use the
`fit_yeo_johnson` Rust binary at
`~/work/zen/zenanalyze/target/release/fit_yeo_johnson`:

  cargo build --release -p zenpredict-bake --features fit-yj --bin fit_yeo_johnson

## Usage

  python3 scripts/v_next/v0_20_feature_transform_greedy_screen.py \\
    --features-parquet /mnt/v/zen/zensim-training/2026-05-07/v06-features/safe_synth_ssim2_features.parquet \\
    --out benchmarks/v0_20_greedy_screen_2026-05-15.tsv \\
    --min-lift 0.005

  # Against canonical safesyn with the cvvdp×iwssim mix target:
  python3 scripts/v_next/v0_20_feature_transform_greedy_screen.py \\
    --features-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet \\
    --target-column mix_cv40_iw60 \\
    --out benchmarks/yeo_johnson_screen_2026-05-25/screen_results.tsv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


# -- transforms -----------------------------------------------------------------


def t_identity(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return x


def t_log(x: np.ndarray, _params: list[float]) -> np.ndarray:
    # ln(x). Only valid for strictly positive; NaN below 0.
    out = np.full_like(x, np.nan)
    pos = x > 0
    out[pos] = np.log(x[pos])
    return out


def t_log1p(x: np.ndarray, _params: list[float]) -> np.ndarray:
    out = np.full_like(x, np.nan)
    valid = x >= 0
    out[valid] = np.log1p(x[valid])
    return out


def t_signed_log1p(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def t_signed_sqrt(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


def t_signed_cbrt(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.cbrt(np.abs(x))


def t_clip_then_log1p(x: np.ndarray, params: list[float]) -> np.ndarray:
    eps = params[0] if params else 0.0
    return np.log1p(np.maximum(0.0, x - eps))


def t_winsor_p99(x: np.ndarray, params: list[float]) -> np.ndarray:
    if len(params) < 2:
        return x
    return np.clip(x, params[0], params[1])


def t_quantile_bins(x: np.ndarray, params: list[float]) -> np.ndarray:
    if len(params) < 2:
        return x
    # Number of edges = N; outputs are bin_idx/N
    edges = np.asarray(params, dtype=np.float64)
    idx = np.zeros_like(x, dtype=np.float64)
    for edge in edges:
        idx += (x >= edge).astype(np.float64)
    return idx / len(edges)


def t_yeo_johnson(x: np.ndarray, params: list[float]) -> np.ndarray:
    """Yeo-Johnson transform with given λ. Mirrors the Rust runtime in
    `zenpredict::FeatureTransform::YeoJohnson` exactly (smooth at λ→0
    and λ→2 limits, full-real-line domain).

    Empty params reduces to Identity to match the runtime fallback.
    """
    if not params:
        return x
    lmb = float(params[0])
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    eps = 1e-9
    if abs(lmb) < eps:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lmb) - 1.0) / lmb
    if abs(lmb - 2.0) < eps:
        out[neg] = -np.log1p(-x[neg])
    else:
        e = 2.0 - lmb
        out[neg] = -(np.power(-x[neg] + 1.0, e) - 1.0) / e
    return out


# Module-level YJ λ search bounds. main() overrides via --grid-min / --grid-max.
# Defaults match scipy's documented [-2, 2] for backwards-compatibility with
# the 2026-05-25 initial screen.
_YJ_GRID_LO: float = -2.0
_YJ_GRID_HI: float = 2.0


def fit_yj_lambda(col: np.ndarray, lo: float | None = None, hi: float | None = None,
                  tol: float = 1e-5) -> float:
    """Golden-section search for the MLE Yeo-Johnson λ on `col`.

    Mirrors the `fit_yeo_johnson` Rust binary at
    `~/work/zen/zenanalyze/target/release/fit_yeo_johnson` —
    bound-clamped [-2, 2] is scipy's documented default. Converges
    to ~1e-5 in ~50 evaluations; cheap per-feature.

    `lo` / `hi` default to module-level `_YJ_GRID_LO` / `_YJ_GRID_HI`,
    which `main()` configures from `--grid-min` / `--grid-max`.
    """
    if lo is None:
        lo = _YJ_GRID_LO
    if hi is None:
        hi = _YJ_GRID_HI
    finite = col[np.isfinite(col)]
    if finite.size < 2:
        return 1.0  # YJ identity at λ=1
    var = float(finite.var())
    if var < 1e-10:
        return 1.0

    def loglik(lmb: float) -> float:
        y = t_yeo_johnson(finite, [lmb])
        if not np.all(np.isfinite(y)):
            return -np.inf
        # Log-Jacobian: (λ - 1) * Σ sign(x) * log(|x| + 1)
        log_jac = (lmb - 1.0) * float(np.sum(np.sign(finite) * np.log1p(np.abs(finite))))
        # Gaussian fit term: -(n/2) * log(var(y))
        v = max(float(y.var()), 1e-30)
        return log_jac - 0.5 * finite.size * math.log(v)

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = loglik(c)
    fd = loglik(d)
    iters = 0
    while (b - a) > tol and iters < 200:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loglik(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loglik(d)
        iters += 1
    return (a + b) / 2.0


def t_soft_clip(x: np.ndarray, params: list[float]) -> np.ndarray:
    knee, soft = params
    a = np.abs(x)
    # Maximum avoids evaluating an invalid log in the unselected core.
    tail = knee + soft * np.log1p(np.maximum(a-knee, 0.)/soft)
    return np.where(a <= knee, x, np.sign(x)*tail)


def t_soft_sign(x: np.ndarray, params: list[float]) -> np.ndarray:
    return x/(params[0]+np.abs(x))


R14_VARIANTS = ("identity", "log1p", "signed_log1p", "yeo_johnson", "soft_clip", "soft_sign")


TRANSFORMS: dict[str, Callable[[np.ndarray, list[float]], np.ndarray]] = {
    "identity": t_identity,
    "soft_clip": t_soft_clip,
    "soft_sign": t_soft_sign,
    "log": t_log,
    "log1p": t_log1p,
    "signed_log1p": t_signed_log1p,
    "signed_sqrt": t_signed_sqrt,
    "signed_cbrt": t_signed_cbrt,
    "clip_then_log1p": t_clip_then_log1p,
    "winsor_p99": t_winsor_p99,
    "quantile_bins": t_quantile_bins,
    "yeo_johnson": t_yeo_johnson,
}


# -- param sweeps ---------------------------------------------------------------


def sweep_for(name: str, col: np.ndarray) -> list[list[float]]:
    """List of param-vector candidates for a given transform name +
    feature column. Single empty list for non-parameterized variants."""
    valid = col[np.isfinite(col)]
    if valid.size == 0:
        return [[]]
    if name == "soft_clip":
        return [[float(k), float(k)] for k in np.percentile(valid, [25,50,75,90]) if k > 0]
    if name == "soft_sign":
        return [[float(k)] for k in np.percentile(valid,[50,75,90]) if k > 0]
    if name == "clip_then_log1p":
        # ε = percentiles of the feature's positive distribution. We
        # want ε at the noise-floor — try p5..p75 spread to find where
        # subtracting noise yields the cleanest log-shape.
        pcts = [5, 10, 25, 50, 75]
        return [[float(np.percentile(valid, p))] for p in pcts]
    if name == "winsor_p99":
        # (lo, hi) clamp pairs from feature percentiles.
        bounds = [(1, 99), (5, 95), (10, 90), (25, 75)]
        return [
            [float(np.percentile(valid, lo)), float(np.percentile(valid, hi))]
            for (lo, hi) in bounds
        ]
    if name == "quantile_bins":
        # 8-bin edges from percentiles 12.5, 25, 37.5, ..., 87.5
        edges = [float(np.percentile(valid, p)) for p in [12.5, 25, 37.5, 50, 62.5, 75, 87.5]]
        return [edges]
    if name == "yeo_johnson":
        # MLE-fit λ on the marginal feature distribution. λ ∈ [-2, 2]
        # is scipy's documented default and matches the Rust runtime's
        # `fit_yeo_johnson` binary. One config per feature — no sweep
        # since λ is data-driven.
        lmb = fit_yj_lambda(valid)
        return [[lmb]]
    # Non-parameterized
    return [[]]


# -- I/O ------------------------------------------------------------------------


def load_features_parquet(path: Path, target_column: str = "human_score"):
    """Load a parquet file with columns `<target_column>, f0..f<N-1>`.

    Returns (mos: np.ndarray[n], features: np.ndarray[n, N], feat_names).
    Rows where `target_column` is null are dropped (canonical parquets
    use explicit nulls for non-applicable targets).
    """
    try:
        import pyarrow.parquet as pq  # noqa: F401
        import pyarrow as pa  # noqa: F401
    except ImportError:
        print("ERROR: pyarrow not installed. pip install pyarrow", file=sys.stderr)
        sys.exit(2)
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    tbl = pq.read_table(str(path))
    n = tbl.num_rows
    cols = tbl.column_names
    if target_column not in cols:
        raise RuntimeError(
            f"missing target column '{target_column}' in {path}; "
            f"available columns: {sorted(c for c in cols if not c.startswith('f'))}"
        )
    feat_cols = [c for c in cols if c.startswith("f") and c[1:].isdigit()]
    feat_cols.sort(key=lambda c: int(c[1:]))
    mos_arr = tbl[target_column].to_numpy(zero_copy_only=False).astype(np.float64)
    arrs = []
    for c in feat_cols:
        arrs.append(tbl[c].to_numpy(zero_copy_only=False).astype(np.float64))
    features = np.stack(arrs, axis=1)
    # Drop rows where the target is null/NaN.
    mask = np.isfinite(mos_arr)
    n_dropped = int((~mask).sum())
    if n_dropped:
        mos_arr = mos_arr[mask]
        features = features[mask, :]
        print(
            f"  dropped {n_dropped} rows with null/NaN '{target_column}'",
            file=sys.stderr,
        )
    print(
        f"loaded {path.name}: n={len(mos_arr)} rows × {len(feat_cols)} features "
        f"(target='{target_column}') in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return mos_arr, features, feat_cols


def load_features_csv(path: Path, target_column: str = "human_score"):
    """Slower CSV path. Reads everything into memory."""
    t0 = time.perf_counter()
    feat_cols = None
    mos_list = []
    feat_rows = []
    with path.open() as f:
        r = csv.reader(f)
        header = next(r)
        feat_cols = [c for c in header if c.startswith("f") and c[1:].isdigit()]
        feat_idx = [header.index(c) for c in feat_cols]
        if target_column not in header:
            raise RuntimeError(
                f"missing target column '{target_column}' in {path}"
            )
        mos_idx = header.index(target_column)
        for row in r:
            try:
                mos = float(row[mos_idx])
                feats = [float(row[i]) for i in feat_idx]
            except (ValueError, IndexError):
                continue
            mos_list.append(mos)
            feat_rows.append(feats)
    mos = np.array(mos_list, dtype=np.float64)
    features = np.array(feat_rows, dtype=np.float64)
    print(
        f"loaded {path.name}: n={len(mos)} rows × {len(feat_cols)} features "
        f"(target='{target_column}') in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return mos, features, feat_cols


# -- core analysis --------------------------------------------------------------


def safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """|Pearson(a, b)| dropping non-finite rows. Returns (corr, n)."""
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3:
        return (float("nan"), n)
    aa = a[mask]
    bb = b[mask]
    sa = aa.std()
    sb = bb.std()
    if sa < 1e-12 or sb < 1e-12:
        return (0.0, n)
    return (float(abs(np.corrcoef(aa, bb)[0, 1])), n)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """|Spearman| via ranks then Pearson. Drops non-finite rows."""
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    # ranks via argsort-argsort
    ra = np.argsort(np.argsort(aa)).astype(np.float64)
    rb = np.argsort(np.argsort(bb)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(ra, rb)[0, 1]))


def screen_one_feature(
    i: int,
    feat_col: np.ndarray,
    mos: np.ndarray,
) -> dict:
    """For one feature column, sweep all transforms × param configs and
    return the best transform's (lift, transform_token, params, etc.).

    Training-safety gates (added 2026-05-15 after first run hit NaN
    loss): some transforms produce NaN on real training data when the
    Pearson screen accepts them via its NaN-dropping rule. The gates:

    - `log` requires `min(feat) > 0` (strictly positive). Mathematically
      `log(x ≤ 0) = NaN`, and the trainer's apply path doesn't drop
      bad rows.
    - `log1p` requires `min(feat) > -1` (anything > -1, since
      `log1p(0) = 0`).

    Other transforms (`signed_*`, `clip_then_log1p`, `winsor_p99`,
    `quantile_bins`, `yeo_johnson`) are safe across the full real line."""
    finite = feat_col[np.isfinite(feat_col)]
    feat_min = float(finite.min()) if finite.size else 0.0
    baseline, n = safe_pearson(feat_col, mos)
    baseline_spearman = safe_spearman(feat_col, mos)
    best_lift = 0.0
    best_token = "identity"
    best_params: list[float] = []
    best_corr = baseline
    for token, fn in TRANSFORMS.items():
        if token in ("soft_clip", "soft_sign"):
            continue
        # Training-safety gates
        if token == "log" and feat_min <= 0.0:
            continue
        if token == "log1p" and feat_min <= -1.0:
            continue
        for params in sweep_for(token, feat_col):
            tx = fn(feat_col, params)
            corr, _ = safe_pearson(tx, mos)
            if math.isnan(corr):
                continue
            lift = corr - baseline
            if lift > best_lift:
                best_lift = lift
                best_token = token
                best_params = params
                best_corr = corr
    return {
        "feat_idx": i,
        "best_transform": best_token,
        "params_csv": ",".join(f"{v:.6g}" for v in best_params) if best_params else "",
        "baseline_pearson": baseline,
        "transformed_pearson": best_corr,
        "lift": best_lift,
        "baseline_spearman": baseline_spearman,
        "n_samples": n,
    }


def screen_one_feature_all_transforms(
    i: int,
    feat_col: np.ndarray,
    mos: np.ndarray,
    variants: tuple[str, ...] | None = None,
    yj_lambda: float | None = None,
) -> tuple[dict, dict[str, tuple[float, float, list[float]]]]:
    """Like `screen_one_feature` but ALSO returns per-transform results
    (token → (pearson, lift, params)) so the caller can attribute
    Yeo-Johnson lift specifically without re-running the full screen.

    Returns (best_record, per_transform_pearson).
    """
    finite = feat_col[np.isfinite(feat_col)]
    feat_min = float(finite.min()) if finite.size else 0.0
    baseline, n = safe_pearson(feat_col, mos)
    baseline_spearman = safe_spearman(feat_col, mos)
    best_lift = 0.0
    best_token = "identity"
    best_params: list[float] = []
    best_corr = baseline
    per_transform: dict[str, tuple[float, float, list[float]]] = {
        "identity": (baseline, 0.0, []),
    }
    allowed = variants if variants is not None else tuple(t for t in TRANSFORMS if t not in ("soft_clip", "soft_sign"))
    if any(t not in TRANSFORMS for t in allowed):
        raise ValueError("unknown screen variant")
    for token, fn in TRANSFORMS.items():
        if token not in allowed:
            continue
        if token == "identity":
            continue
        if token == "log" and feat_min <= 0.0:
            per_transform[token] = (float("nan"), float("nan"), [])
            continue
        if token == "log1p" and feat_min <= -1.0:
            per_transform[token] = (float("nan"), float("nan"), [])
            continue
        best_corr_this = float("-inf")
        best_params_this: list[float] = []
        candidates = [[yj_lambda]] if token == "yeo_johnson" and yj_lambda is not None else sweep_for(token, feat_col)
        for params in candidates:
            tx = fn(feat_col, params)
            corr, _ = safe_pearson(tx, mos)
            if math.isnan(corr):
                continue
            if corr > best_corr_this:
                best_corr_this = corr
                best_params_this = params
        if math.isfinite(best_corr_this):
            lift_this = best_corr_this - baseline
            per_transform[token] = (best_corr_this, lift_this, best_params_this)
            if lift_this > best_lift:
                best_lift = lift_this
                best_token = token
                best_params = best_params_this
                best_corr = best_corr_this
        else:
            per_transform[token] = (float("nan"), float("nan"), [])
    rec = {
        "feat_idx": i,
        "best_transform": best_token,
        "params_csv": ",".join(format(v, ".9g" if variants is not None else ".6g") for v in best_params) if best_params else "",
        "baseline_pearson": baseline,
        "transformed_pearson": best_corr,
        "lift": best_lift,
        "baseline_spearman": baseline_spearman,
        "n_samples": n,
    }
    return rec, per_transform


def _block_for_feat_idx(idx: int) -> str:
    """Per the canonical 372-feature schema documented in
    zensim/CLAUDE.md "Canonical training/validation corpora":

    - f0..f155     basic features (13/ch × 3ch × 4 scales)
    - f156..f227   peak features (6/ch × 3ch × 4 scales)
    - f228..f299   masked features (6/ch × 3ch × 4 scales)
    - f300..f347   psychovisual features (4/ch × 3ch × 4 scales) — note older
                   doc cites 48 here. The newer canonical schema reuses
                   f300..f371 as the IW-pool block (72 features).
    - f348..f371   trailing slots — pooling/projection
    """
    if idx < 156:
        return "basic"
    if idx < 228:
        return "peak"
    if idx < 300:
        return "masked"
    return "iw_pool"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", type=Path)
    ap.add_argument("--features-csv", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--target-column",
        type=str,
        default="human_score",
        help=(
            "Name of the target column to correlate against. Default 'human_score'. "
            "Set to e.g. 'mix_cv40_iw60' or 'cvvdp_score' to screen against a "
            "different anchor (the canonical parquets carry several)."
        ),
    )
    ap.add_argument(
        "--min-lift",
        type=float,
        default=0.005,
        help="emit rows where transformed_pearson - baseline_pearson >= this. Default 0.005.",
    )
    ap.add_argument(
        "--per-transform-out",
        type=Path,
        default=None,
        help=(
            "Optional path for a long-form TSV with one row per "
            "(feature, transform) pair. Columns: feat_idx, transform, "
            "pearson, lift, params_csv. Useful for analyzing "
            "candidate-level winners (e.g. YJ-specific lift)."
        ),
    )
    ap.add_argument(
        "--grid-min",
        type=float,
        default=-2.0,
        help=(
            "Lower bound for Yeo-Johnson λ search (golden-section). "
            "Default -2.0 matches scipy's documented default. Widen "
            "(e.g. -5) when many features pin at the boundary — heavy-"
            "positive-tailed features can want λ ≈ -2.7 or beyond."
        ),
    )
    ap.add_argument(
        "--grid-max",
        type=float,
        default=2.0,
        help=(
            "Upper bound for Yeo-Johnson λ search. Default 2.0 matches "
            "scipy. Widen (e.g. 5) if features want positive λ above "
            "the boundary."
        ),
    )
    ap.add_argument("--variants", help="comma-separated transform token filter; historical defaults when absent")
    args = ap.parse_args()
    variants = tuple(args.variants.split(",")) if args.variants else None
    if variants and any(t not in TRANSFORMS for t in variants):
        ap.error("unknown transform variant")
    if args.grid_min >= args.grid_max:
        print(
            f"ERROR: --grid-min ({args.grid_min}) must be < --grid-max ({args.grid_max})",
            file=sys.stderr,
        )
        return 2
    global _YJ_GRID_LO, _YJ_GRID_HI
    _YJ_GRID_LO = float(args.grid_min)
    _YJ_GRID_HI = float(args.grid_max)
    print(
        f"YJ λ grid: [{_YJ_GRID_LO}, {_YJ_GRID_HI}]",
        file=sys.stderr,
    )
    if not args.features_parquet and not args.features_csv:
        print("ERROR: pass --features-parquet or --features-csv", file=sys.stderr)
        return 2

    if args.features_parquet:
        mos, features, feat_names = load_features_parquet(
            args.features_parquet, target_column=args.target_column
        )
    else:
        mos, features, feat_names = load_features_csv(
            args.features_csv, target_column=args.target_column
        )

    n_features = features.shape[1]
    rows: list[dict] = []
    per_transform_rows: list[tuple[int, str, float, float, str]] = []
    t0 = time.perf_counter()
    for i in range(n_features):
        rec, per_tx = screen_one_feature_all_transforms(i, features[:, i], mos, variants=variants)
        rows.append(rec)
        for tok, (corr, lift, params) in per_tx.items():
            params_csv = ",".join(f"{v:.6g}" for v in params) if params else ""
            per_transform_rows.append(
                (
                    i,
                    tok,
                    corr if math.isfinite(corr) else float("nan"),
                    lift if math.isfinite(lift) else float("nan"),
                    params_csv,
                )
            )
    print(
        f"screened {n_features} features in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )

    # Sort by lift desc
    rows.sort(key=lambda r: -r["lift"])

    # Emit all rows (informational) and a filtered "winners" subset.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "feat_idx",
                "best_transform",
                "params_csv",
                "baseline_pearson",
                "transformed_pearson",
                "lift",
                "baseline_spearman",
                "n_samples",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["feat_idx"],
                    r["best_transform"],
                    r["params_csv"],
                    f"{r['baseline_pearson']:.6f}",
                    f"{r['transformed_pearson']:.6f}",
                    f"{r['lift']:.6f}",
                    f"{r['baseline_spearman']:.6f}",
                    r["n_samples"],
                ]
            )
    if args.per_transform_out:
        args.per_transform_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_transform_out.open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["feat_idx", "transform", "pearson", "lift", "params_csv"])
            for (idx, tok, corr, lift, params_csv) in per_transform_rows:
                w.writerow([
                    idx, tok,
                    f"{corr:.6f}" if math.isfinite(corr) else "nan",
                    f"{lift:.6f}" if math.isfinite(lift) else "nan",
                    params_csv,
                ])
        print(
            f"wrote {len(per_transform_rows)} per-transform rows to {args.per_transform_out}",
            file=sys.stderr,
        )
    winners = [r for r in rows if r["lift"] >= args.min_lift]
    print(
        f"wrote {len(rows)} rows to {args.out}; "
        f"{len(winners)} features clear --min-lift {args.min_lift}",
        file=sys.stderr,
    )

    # Per-block summary (block = basic / peak / masked / iw_pool).
    print("\nper-block: feature counts where best_transform != identity:", file=sys.stderr)
    block_totals: dict[str, int] = {}
    block_wins: dict[str, int] = {}
    block_yj_wins: dict[str, int] = {}
    for r in rows:
        b = _block_for_feat_idx(r["feat_idx"])
        block_totals[b] = block_totals.get(b, 0) + 1
        if r["best_transform"] != "identity":
            block_wins[b] = block_wins.get(b, 0) + 1
        if r["best_transform"] == "yeo_johnson":
            block_yj_wins[b] = block_yj_wins.get(b, 0) + 1
    print("block     | total | any_transform_wins | yj_wins", file=sys.stderr)
    for b in ("basic", "peak", "masked", "iw_pool"):
        tot = block_totals.get(b, 0)
        any_w = block_wins.get(b, 0)
        yj_w = block_yj_wins.get(b, 0)
        print(f"{b:<9} | {tot:>5} | {any_w:>18} | {yj_w:>7}", file=sys.stderr)

    # YJ-specific summary across all features.
    yj_lifts: list[float] = []
    for (_idx, tok, _corr, lift, _params) in per_transform_rows:
        if tok == "yeo_johnson" and math.isfinite(lift):
            yj_lifts.append(lift)
    yj_arr = np.array(yj_lifts, dtype=np.float64)
    if yj_arr.size:
        print(
            f"\nyeo_johnson lift across {yj_arr.size} features: "
            f"mean={yj_arr.mean():.4f}, median={float(np.median(yj_arr)):.4f}, "
            f"max={yj_arr.max():.4f}, "
            f"n>=0.005={(yj_arr >= 0.005).sum()}, "
            f"n>=0.02={(yj_arr >= 0.02).sum()}, "
            f"n>=0.05={(yj_arr >= 0.05).sum()}",
            file=sys.stderr,
        )

    # Print top 20 to stderr for quick eyeball
    print("\ntop 20 by lift:", file=sys.stderr)
    print("feat | transform        | params                       | base_p | tx_p   | lift", file=sys.stderr)
    for r in rows[:20]:
        print(
            f"f{r['feat_idx']:<3} | {r['best_transform']:<16} | {r['params_csv']:<28} | "
            f"{r['baseline_pearson']:.4f} | {r['transformed_pearson']:.4f} | {r['lift']:+.4f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
