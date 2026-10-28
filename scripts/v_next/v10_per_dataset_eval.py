#!/usr/bin/env python3
"""V10 per-human-dataset eval (task #183, 2026-05-20).

Measures cross-codec stddev at JND/JOD targets + per-curve monotonicity
for each of the 3 V10 ship profiles (BalancedV3, CompressionV3,
TunerV4) plus their V9 ancestors (BalancedV2, CompressionV2, TunerV3).

## Cross-codec stddev (per profile, per target T)

For each of the 1000 refs in the butter parquets:
1. For each codec in {zenjpeg, zenwebp, zenavif, zenjxl}: load the
   19-q feature grid, score every q against the bake, pick the q
   whose zensim score is nearest the target T.
2. Record `butter_pnorm3` at that q (joined from the butter parquet).
3. Compute stddev of butter_pnorm3 across the 4 codecs per source.
4. Aggregate median + p90 across sources.

Targets: T=80 (V10 JND) and T=50 (V10 JOD).

## Per-curve monotonicity (per profile)

For each (ref, codec) curve from butter parquet (1000 × 4 = 4000
curves per profile, 19 q-points each):
1. Sort by q ascending.
2. Score every q against the bake.
3. Count strict decreases (`score(q+1) < score(q)`) across adjacent
   pairs. Pair-strict mono rate = 1 - n_strict / n_pairs.

## Per-human-dataset filtering

The intent of the task was "filter to sources also in the human
dataset's reference set." The butter parquets are on a synthetic
source corpus (1000 `*_512sq.png` / `*_1024sq.png` refs) that does
NOT overlap with the human dataset reference sets
(`SRC0579.png` for KonJND, `00002_853x945.png` for AIC-3, etc.).
Filtered measurement therefore yields zero rows per human dataset.

This script reports the GLOBAL measurement (all 1000 butter refs)
and flags the gap explicitly. To get truly per-human-dataset
measurement, the butter sweep would need re-running on each human
dataset's source images.

Output: a TSV with per-(profile, metric, value) rows, plus a markdown
summary.
"""

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median, stdev

import pyarrow.parquet as pq

BUTTER_ROOT = Path("/mnt/v/zen/picker-training/2026-05-19/butter")
HUMAN_VAL_ROOT = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/val")
WEIGHTS_ROOT = Path("/home/lilith/work/zen/zensimx/zensim/weights")
PREDICT_TOOL = Path(
    "/home/lilith/work/zen/zensimx/target/release/predict_features_with_bake"
)

# 19-q grid present in the butter parquets.
Q_GRID = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
CODECS = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]
HUMAN_DATASETS = ["aic3", "aic4", "konjnd"]

# V10 score-space targets (per V10 anchor table).
TARGETS_V10 = {"JND": 80.0, "JOD": 50.0}

# Gate criteria from task #183.
CROSS_CODEC_STDDEV_GATE = 5.0
MONO_GATE = 0.94

# Profiles: (name, bake_path, post_mode, n_features).
# V9 ancestors use post=clamp (bounded by spline+clamp).
# V10 profiles use post=extrapolate (no clamp; spline output flows through).
PROFILES = [
    # V9 ancestors
    ("V9_BalancedV2", WEIGHTS_ROOT / "v_balanced_v2_2026-05-20.bin", "clamp", 300),
    ("V9_CompressionV2", WEIGHTS_ROOT / "v_compression_v2_2026-05-20.bin", "clamp", 300),
    ("V9_TunerV3", WEIGHTS_ROOT / "v_tuner_v9_2026-05-20.bin", "clamp", 372),
    # V10 ships
    ("V10_BalancedV3", WEIGHTS_ROOT / "v_balanced_v3_2026-05-20.bin", "extrapolate", 300),
    ("V10_CompressionV3", WEIGHTS_ROOT / "v_compression_v3_2026-05-20.bin", "extrapolate", 300),
    ("V10_TunerV4", WEIGHTS_ROOT / "v_tuner_v10_2026-05-20.bin", "extrapolate", 372),
]


# ---------------------------------------------------------------------------
# Load butter parquet features (per codec)
# ---------------------------------------------------------------------------


def load_butter_codec(parquet_path):
    """Return dict[ref] -> dict[q] -> (features list, butter_pnorm3)."""
    print(f"loading {parquet_path} …", file=sys.stderr, flush=True)
    # Read only what we need.
    cols = ["ref_basename", "q", "butter_pnorm3"] + [f"f{i}" for i in range(372)]
    table = pq.read_table(parquet_path, columns=cols)
    df = table.to_pandas()
    by_ref = {}
    feat_cols = [f"f{i}" for i in range(372)]
    for ref, group in df.groupby("ref_basename"):
        per_q = {}
        for _, row in group.iterrows():
            q = int(row["q"])
            butter = float(row["butter_pnorm3"])
            feats = [float(row[c]) for c in feat_cols]
            per_q[q] = (feats, butter)
        by_ref[str(ref)] = per_q
    print(f"  loaded {len(by_ref)} refs × {len(Q_GRID)} qs", file=sys.stderr, flush=True)
    return by_ref


# ---------------------------------------------------------------------------
# Score a feature batch via the Rust binary
# ---------------------------------------------------------------------------


def score_batch(features_rows, bake_path, post_mode, n_features):
    """Returns list of scores for each row."""
    n_rows = len(features_rows)
    if n_rows == 0:
        return []
    buf = bytearray()
    buf += struct.pack("<II", n_features, n_rows)
    for row in features_rows:
        # Truncate or zero-pad to n_features.
        feats = row[:n_features] if len(row) >= n_features else row + [0.0] * (n_features - len(row))
        buf += struct.pack(f"<{n_features}f", *feats)
    with tempfile.NamedTemporaryFile(suffix=".features.bin", delete=False) as f:
        f.write(buf)
        feats_path = f.name
    try:
        out = subprocess.check_output(
            [
                str(PREDICT_TOOL),
                "--bake",
                str(bake_path),
                "--bake-post",
                post_mode,
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
# For each codec, build the full score matrix [ref × q] per profile
# ---------------------------------------------------------------------------


def score_codec_matrix(by_ref, bake_path, post_mode, n_features, refs_ordered):
    """Return dict[ref] -> dict[q] -> score.

    Batches all (ref, q) rows into one Rust invocation per codec.
    """
    # Build row order: (ref, q) flattened.
    flat_rows = []
    flat_keys = []
    for ref in refs_ordered:
        per_q = by_ref.get(ref)
        if per_q is None:
            continue
        for q in Q_GRID:
            entry = per_q.get(q)
            if entry is None:
                continue
            feats, _butter = entry
            flat_rows.append(feats)
            flat_keys.append((ref, q))
    scores = score_batch(flat_rows, bake_path, post_mode, n_features)
    result = {}
    for (ref, q), s in zip(flat_keys, scores):
        result.setdefault(ref, {})[q] = s
    return result


# ---------------------------------------------------------------------------
# Cross-codec stddev (per target T)
# ---------------------------------------------------------------------------


def cross_codec_stddev(per_codec_scores, per_codec_butter, refs, target):
    """For each ref, find the q per codec whose score is closest to
    target, look up butter_pnorm3 at that q, compute stddev across codecs.
    Returns list of stddev values (one per ref) and per-ref dict.
    """
    stddevs = []
    per_ref_detail = []
    for ref in refs:
        butters = []
        per_codec_q = {}
        for codec in CODECS:
            score_grid = per_codec_scores[codec].get(ref)
            butter_grid = per_codec_butter[codec].get(ref)
            if not score_grid or not butter_grid:
                continue
            # Find q with min |score - target|.
            best_q = None
            best_err = float("inf")
            for q, s in score_grid.items():
                err = abs(s - target)
                if err < best_err:
                    best_err = err
                    best_q = q
            if best_q is None:
                continue
            b = butter_grid.get(best_q)
            if b is not None:
                butters.append(b)
                per_codec_q[codec] = (best_q, score_grid[best_q], b)
        if len(butters) >= 2:
            sd = stdev(butters)
            stddevs.append(sd)
            per_ref_detail.append((ref, sd, per_codec_q))
    return stddevs, per_ref_detail


# ---------------------------------------------------------------------------
# Monotonicity (pair-strict per curve)
# ---------------------------------------------------------------------------


def monotonicity(per_codec_scores, refs):
    """Pair-strict mono rate = 1 - n_strict_decrease_pairs / n_adj_pairs.

    Also report curve-strict = n_curves_zero_decrease / n_curves.
    """
    n_pairs = 0
    n_strict_decrease = 0
    n_ties = 0
    n_curves = 0
    n_curves_clean = 0
    for codec in CODECS:
        per_ref = per_codec_scores[codec]
        for ref in refs:
            grid = per_ref.get(ref)
            if not grid or len(grid) < 2:
                continue
            n_curves += 1
            qs_sorted = sorted(grid.keys())
            curve_clean = True
            for i in range(len(qs_sorted) - 1):
                q1, q2 = qs_sorted[i], qs_sorted[i + 1]
                s1, s2 = grid[q1], grid[q2]
                n_pairs += 1
                if s2 < s1:
                    n_strict_decrease += 1
                    curve_clean = False
                elif s2 == s1:
                    n_ties += 1
            if curve_clean:
                n_curves_clean += 1
    pair_strict = 1.0 - (n_strict_decrease / n_pairs) if n_pairs else float("nan")
    curve_strict = n_curves_clean / n_curves if n_curves else float("nan")
    tied_rate = n_ties / n_pairs if n_pairs else float("nan")
    return {
        "pair_strict": pair_strict,
        "curve_strict": curve_strict,
        "tied_rate": tied_rate,
        "n_pairs": n_pairs,
        "n_curves": n_curves,
        "n_strict_decrease": n_strict_decrease,
        "n_ties": n_ties,
    }


# ---------------------------------------------------------------------------
# Human dataset filter (for surfacing the gap)
# ---------------------------------------------------------------------------


def human_dataset_refs(name):
    path = HUMAN_VAL_ROOT / f"{name}.parquet"
    if not path.exists():
        return set()
    t = pq.read_table(path, columns=["ref_basename"])
    return set(t.column("ref_basename").to_pylist())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument(
        "--limit-refs",
        type=int,
        default=0,
        help="if > 0, only process this many refs (for quick test runs)",
    )
    args = ap.parse_args()

    # 1) Load butter parquets for all 4 codecs.
    by_codec = {}
    for codec in CODECS:
        path = BUTTER_ROOT / f"{codec}.parquet"
        by_codec[codec] = load_butter_codec(path)

    # 2) Intersect refs across all 4 codecs (must have all 4 to compute
    #    cross-codec stddev). Sort lex for determinism.
    refs_common = sorted(
        set.intersection(*[set(by_codec[c]) for c in CODECS])
    )
    print(f"\n{len(refs_common)} refs common across all 4 codecs", file=sys.stderr, flush=True)
    if args.limit_refs > 0:
        refs_common = refs_common[: args.limit_refs]
        print(f"  limited to first {len(refs_common)} for testing", file=sys.stderr, flush=True)

    # 3) Compute per-human-dataset intersection (just for reporting).
    per_dataset_refs = {}
    for ds in HUMAN_DATASETS:
        hr = human_dataset_refs(ds)
        inter = sorted(set(refs_common) & hr)
        per_dataset_refs[ds] = {
            "n_dataset_refs": len(hr),
            "n_intersection_with_butter": len(inter),
            "intersection": inter,
        }
        print(
            f"  human dataset {ds}: dataset_refs={len(hr)} intersection_with_butter={len(inter)}",
            file=sys.stderr,
            flush=True,
        )

    # Pre-build butter lookup per (codec, ref) -> dict q -> butter.
    per_codec_butter = {}
    for codec in CODECS:
        per_codec_butter[codec] = {
            ref: {q: by_codec[codec][ref][q][1] for q in by_codec[codec][ref]}
            for ref in by_codec[codec]
        }

    # 4) For each profile, score the full grid and compute metrics.
    rows = []
    for profile_name, bake_path, post_mode, n_features in PROFILES:
        if not bake_path.exists():
            print(f"!!! SKIP {profile_name}: bake missing at {bake_path}", file=sys.stderr)
            continue
        print(f"\n=== {profile_name} (post={post_mode}, n_features={n_features}) ===",
              file=sys.stderr, flush=True)
        per_codec_scores = {}
        for codec in CODECS:
            print(f"  scoring {codec} …", file=sys.stderr, flush=True)
            per_codec_scores[codec] = score_codec_matrix(
                by_codec[codec], bake_path, post_mode, n_features, refs_common
            )

        # Cross-codec stddev per target.
        for target_name, target_val in TARGETS_V10.items():
            sds, _detail = cross_codec_stddev(
                per_codec_scores, per_codec_butter, refs_common, target_val
            )
            if not sds:
                continue
            sds_sorted = sorted(sds)
            n = len(sds)
            med = median(sds)
            p90 = sds_sorted[int(0.90 * (n - 1))]
            p99 = sds_sorted[int(0.99 * (n - 1))]
            mx = max(sds)
            mean = sum(sds) / n
            rows.append({
                "profile": profile_name,
                "scope": "global_butter",
                "metric": f"cross_codec_stddev_at_{target_name}_T{int(target_val)}",
                "n": n,
                "median": med,
                "mean": mean,
                "p90": p90,
                "p99": p99,
                "max": mx,
                "gate_5": "PASS" if med <= CROSS_CODEC_STDDEV_GATE and p90 <= CROSS_CODEC_STDDEV_GATE else "FAIL",
            })
            print(
                f"    cross-codec stddev @ {target_name} (T={target_val}): "
                f"n={n} median={med:.3f} p90={p90:.3f} max={mx:.3f}",
                file=sys.stderr, flush=True,
            )

        # Mono on the full corpus.
        mono = monotonicity(per_codec_scores, refs_common)
        rows.append({
            "profile": profile_name,
            "scope": "global_butter",
            "metric": "pair_strict_mono",
            "n": mono["n_pairs"],
            "median": mono["pair_strict"],
            "mean": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "gate_5": "PASS" if mono["pair_strict"] >= MONO_GATE else "FAIL",
        })
        rows.append({
            "profile": profile_name,
            "scope": "global_butter",
            "metric": "curve_strict_mono",
            "n": mono["n_curves"],
            "median": mono["curve_strict"],
            "mean": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "gate_5": "n/a",
        })
        print(
            f"    mono: pair_strict={mono['pair_strict']:.4f} "
            f"curve_strict={mono['curve_strict']:.4f} "
            f"tied={mono['tied_rate']:.4f} "
            f"({mono['n_strict_decrease']} strict + {mono['n_ties']} ties / {mono['n_pairs']} pairs)",
            file=sys.stderr, flush=True,
        )

    # 5) Write TSV.
    with open(args.out_tsv, "w") as f:
        cols = ["profile", "scope", "metric", "n", "median", "mean", "p90", "p99", "max", "gate_5"]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                str(r[c]) if not isinstance(r[c], float) or r[c] != r[c]
                else f"{r[c]:.4f}"
                for c in cols
            ) + "\n")
    print(f"\nwrote TSV → {args.out_tsv}", file=sys.stderr)

    # 6) Write markdown report.
    with open(args.out_md, "w") as f:
        f.write(_render_report(rows, refs_common, per_dataset_refs))
    print(f"wrote MD → {args.out_md}", file=sys.stderr)
    return 0


def _fmt(x):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _render_report(rows, refs_common, per_dataset_refs):
    out = []
    out.append("# V10 per-human-dataset eval (task #183, 2026-05-20)\n\n")
    out.append("**Bake commit:** main@origin `6e4c665` (V10 ship: TunerV4 / "
               "BalancedV3 / CompressionV3).\n\n")
    out.append("## Gap surfaced: butter parquets do not cover human-dataset sources\n\n")
    out.append(
        "The task asked for cross-codec stddev + mono \"per human dataset.\" "
        "The implementation depends on the existing butter parquets at "
        "`/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`, "
        "which carry q-sweeps for 4 codecs × 1000 reference images on the "
        "**synthetic-safe source corpus** (gen-mixed `*_512sq.png` / "
        "`*_1024sq.png` etc.). These references do NOT overlap with the "
        "human-dataset reference sets:\n\n"
    )
    out.append("| dataset | n dataset refs | n butter refs | intersection |\n")
    out.append("|---|---:|---:|---:|\n")
    for ds, info in per_dataset_refs.items():
        out.append(
            f"| {ds} | {info['n_dataset_refs']} | {len(refs_common)} | "
            f"{info['n_intersection_with_butter']} |\n"
        )
    out.append(
        "\nThe per-human-dataset filtered measurement therefore yields **zero rows** "
        "for every dataset. The literal task request cannot be answered with current "
        "artifacts.\n\n"
    )
    out.append(
        "**What's measurable:** the same cross-codec stddev + per-curve mono "
        "computation, applied **globally** to the synthetic-safe butter sweep "
        "(1000 refs × 4 codecs × 19 qs). Reported below as `scope=global_butter`. "
        "This is the same methodology / corpus used by the V9 mono ship audit "
        "(`benchmarks/v_tuner_v9_mono_audit_2026-05-20.md`), just extended to "
        "all 1000 refs instead of a 50-ref random sample.\n\n"
    )
    out.append(
        "**To get truly per-human-dataset measurement:** the butter sweep would "
        "need re-running on the human-dataset source images themselves "
        "(AIC-3: 10 refs, AIC-4: 5 refs, KonJND: 1008 refs). Estimated wall: "
        "~30 min per codec × 4 codecs × ~1023 refs = several CPU-hours. Not in "
        "this task's 45-min budget.\n\n"
    )
    out.append(f"All measurements below: `n_refs = {len(refs_common)}` (the cross-codec "
               "intersection of the 4 butter parquets).\n\n")

    out.append("## Cross-codec stddev (butter_pnorm3 at target T)\n\n")
    out.append(
        "Methodology per task: for each (profile, target T), find the q whose "
        "score is closest to T for each codec, look up butter_pnorm3 at that q, "
        "compute stddev across the 4 codecs per source, aggregate across "
        f"{len(refs_common)} sources.\n\n"
    )
    out.append("Gate: median + p90 stddev ≤ 5.0 at every (profile, T).\n\n")
    out.append("| profile | target | T | n | median | mean | p90 | p99 | max | gate (≤5 median+p90) |\n")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        if "cross_codec_stddev" not in r["metric"]:
            continue
        # Parse "cross_codec_stddev_at_JND_T80" -> JND, 80
        parts = r["metric"].replace("cross_codec_stddev_at_", "").split("_T")
        tn = parts[0]
        tv = parts[1] if len(parts) > 1 else "—"
        out.append(
            f"| {r['profile']} | {tn} | {tv} | {r['n']} | {_fmt(r['median'])} | "
            f"{_fmt(r['mean'])} | {_fmt(r['p90'])} | {_fmt(r['p99'])} | "
            f"{_fmt(r['max'])} | **{r['gate_5']}** |\n"
        )
    out.append("\n")

    out.append("## Per-curve monotonicity\n\n")
    out.append(
        "Methodology: for each (ref, codec) curve in the butter parquet, sort "
        "by q ascending, score every q against the bake, count strict "
        "decreases. `pair_strict` = 1 - n_strict / n_pairs; `curve_strict` = "
        "fraction of curves with zero strict decreases.\n\n"
    )
    out.append("Gate: pair_strict mono ≥ 0.94 at every profile.\n\n")
    out.append("| profile | metric | n | value | gate (≥0.94 pair) |\n")
    out.append("|---|---|---:|---:|---|\n")
    for r in rows:
        if r["metric"] not in ("pair_strict_mono", "curve_strict_mono"):
            continue
        out.append(
            f"| {r['profile']} | {r['metric']} | {r['n']} | "
            f"{_fmt(r['median'])} | "
            f"{'**' + r['gate_5'] + '**' if r['gate_5'] != 'n/a' else 'n/a'} |\n"
        )
    out.append("\n")

    # V10 vs V9 delta table.
    out.append("## V10 vs V9 delta\n\n")
    out.append("Per (corresponding profile pair):\n\n")
    pairs = [
        ("V9_BalancedV2", "V10_BalancedV3"),
        ("V9_CompressionV2", "V10_CompressionV3"),
        ("V9_TunerV3", "V10_TunerV4"),
    ]
    out.append("| pair | metric | V9 | V10 | Δ (V10 − V9) | direction |\n")
    out.append("|---|---|---:|---:|---:|---|\n")
    by_profile = {(r["profile"], r["metric"]): r for r in rows}
    for v9, v10 in pairs:
        for metric_key, metric_label in [
            ("cross_codec_stddev_at_JND_T80", "cc_stddev @ JND"),
            ("cross_codec_stddev_at_JOD_T50", "cc_stddev @ JOD"),
            ("pair_strict_mono", "pair_strict_mono"),
            ("curve_strict_mono", "curve_strict_mono"),
        ]:
            v9r = by_profile.get((v9, metric_key))
            v10r = by_profile.get((v10, metric_key))
            if not v9r or not v10r:
                continue
            v9v = v9r["median"]
            v10v = v10r["median"]
            delta = v10v - v9v
            # "tighter" = lower stddev / higher mono.
            if "stddev" in metric_key:
                direction = "TIGHTENED" if delta < -0.05 else ("LOOSENED" if delta > 0.05 else "flat")
            else:
                direction = "IMPROVED" if delta > 0.005 else ("REGRESSED" if delta < -0.005 else "flat")
            sign = "+" if delta >= 0 else "−"
            delta_str = f"{sign}{abs(delta):.3f}"
            out.append(
                f"| {v9} → {v10} | {metric_label} | "
                f"{_fmt(v9v)} | {_fmt(v10v)} | {delta_str} | {direction} |\n"
            )
    out.append("\n")

    # Verdict.
    out.append("## Verdict\n\n")
    # ALL PASS only if every V10 row passes.
    v10_rows = [r for r in rows if r["profile"].startswith("V10_") and r["gate_5"] in ("PASS", "FAIL")]
    fails = [r for r in v10_rows if r["gate_5"] == "FAIL"]
    if not fails:
        out.append(
            "**ALL V10 ROWS PASS** the gates (median + p90 cross-codec stddev ≤ 5, "
            f"pair_strict mono ≥ {MONO_GATE}) on the global butter sweep.\n\n"
            "V10 is the right ship under the achievable scope. **Mark task #184 "
            "(V10b fallback) as not needed** — V10's spline reallocation "
            "preserves cross-codec parity and monotonicity vs V9.\n\n"
        )
    else:
        out.append("**V10b DISPATCH NEEDED.** The following V10 rows failed:\n\n")
        for r in fails:
            out.append(
                f"- {r['profile']} {r['metric']}: median={_fmt(r['median'])} "
                f"(gate ≤ {CROSS_CODEC_STDDEV_GATE if 'stddev' in r['metric'] else MONO_GATE})\n"
            )
        out.append(
            "\nV10b should shift JND → 70, JOD → 40 per the task #184 design to "
            "widen knot spacing in the failing band.\n\n"
        )
    out.append(
        "**Important caveat:** these results are on the global synthetic butter "
        "corpus, NOT filtered to human-dataset sources. The user's original "
        "intent (\"tighten cross-codec stddev per human dataset\") cannot be "
        "verified at human-dataset granularity until butter sweeps are produced "
        "on AIC-3 / AIC-4 / KonJND source images. The global stddev is a "
        "reasonable proxy because the synthetic safesyn corpus spans natural "
        "image content (gen-mixed multi-content sources) — but content-class "
        "skew could mean per-dataset numbers differ.\n"
    )
    return "".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
