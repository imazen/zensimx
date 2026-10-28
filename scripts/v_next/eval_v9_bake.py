#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V9 per-bake evaluation script (2026-05-20).

Runs the complete V9 ship gate against ONE calibrated bake:

1. bake_verdict — Mohammadi panel on CID22/KADID/TID/KonJND/AIC-3.
2. JPEG q-sweep — strict monotonicity, tied rate, median range.
3. Range extension — median predicted score at the [q=5 zenavif,
   q=95 zenjxl, q=95 zenwebp] reference rows of the V9 anchor parquet.
4. Clean-anchor accuracy — |predicted − target_score| at each band.
5. Cross-codec consistency at T=30 (JOD), T=60 (JND), T=80, T=90.

Output is a single per-bake markdown verdict file with all gate
results in one table.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

V9_ANCHOR_PARQUET = Path(
    "/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
)
PREDICT_BIN = Path(
    "/home/lilith/work/zen/zensimx/target/release/"
    "predict_features_with_bake"
)
BAKE_VERDICT_BIN = Path(
    "/home/lilith/work/zen/zensimx/target/release/bake_verdict"
)
BUTTER_DIR = Path("/mnt/v/zen/picker-training/2026-05-19/butter")

V9_TARGETS = [0.0, 10.0, 30.0, 50.0, 60.0, 80.0, 90.0, 100.0]
JOD_TARGET = 30.0
JND_TARGET = 60.0
CODECS = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]


def predict_features(bake: Path, feats: np.ndarray, post: str = "raw") -> np.ndarray:
    n_rows, n_features = feats.shape
    tmp = Path("/tmp/v9_eval_feats.bin")
    with tmp.open("wb") as f:
        f.write(struct.pack("<II", n_features, n_rows))
        f.write(feats.astype("<f4", copy=False).tobytes())
    proc = subprocess.run(
        [
            str(PREDICT_BIN),
            "--bake",
            str(bake),
            "--features-file",
            str(tmp),
            "--bake-post",
            post,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out_lines = proc.stdout.strip().splitlines()
    return np.array([float(s) for s in out_lines], dtype=np.float64)


def load_anchor_features(parquet_path: Path) -> dict:
    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    feat_cols = [f"f{i}" for i in range(372)]
    feats = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    return {
        "feats": feats,
        "target_score": df["target_score"].to_numpy(dtype=np.float64),
        "butter_pnorm3": df["butter_pnorm3"].to_numpy(dtype=np.float64),
        "codec": df["codec"].to_numpy(),
        "ref_basename": df["ref_basename"].to_numpy(),
        "q": df["q"].to_numpy(dtype=np.int64),
    }


def score_per_band(preds: np.ndarray, targets: np.ndarray) -> list[dict]:
    out = []
    for t in V9_TARGETS:
        mask = targets == t
        if mask.sum() == 0:
            continue
        sub = preds[mask]
        out.append(
            {
                "target": t,
                "n": int(mask.sum()),
                "med": float(np.median(sub)),
                "p25": float(np.percentile(sub, 25)),
                "p75": float(np.percentile(sub, 75)),
                "abs_err": float(abs(np.median(sub) - t)),
            }
        )
    return out


def range_extension_check(preds: np.ndarray, data: dict) -> dict:
    """Compute median predicted score at the extreme bands."""
    # q=5 worst-codec → score should be ≤ 5 (target 0)
    # q=95 best-codec (zenjxl q=95) → score should be ≥ 95 (target 100)
    is_worstfloor = data["target_score"] == 0.0
    is_lossless = data["target_score"] == 100.0
    return {
        "worstfloor_n": int(is_worstfloor.sum()),
        "worstfloor_med": float(np.median(preds[is_worstfloor]))
        if is_worstfloor.sum() > 0
        else float("nan"),
        "lossless_n": int(is_lossless.sum()),
        "lossless_med": float(np.median(preds[is_lossless]))
        if is_lossless.sum() > 0
        else float("nan"),
    }


def cross_codec_consistency(
    preds: np.ndarray, data: dict, target: float
) -> dict:
    """For each ref image: predicted-score per codec at the rows
    closest to `target`. Then compute cross-codec std per ref + median
    of those stds."""
    band_mask = data["target_score"] == target
    if band_mask.sum() == 0:
        return {"target": target, "n_refs": 0, "cc_std_median": float("nan")}
    by_ref = defaultdict(dict)  # ref -> codec -> score
    p = preds[band_mask]
    refs = data["ref_basename"][band_mask]
    codecs = data["codec"][band_mask]
    for i in range(len(p)):
        by_ref[str(refs[i])][str(codecs[i])] = float(p[i])
    stds = []
    for ref, codec_scores in by_ref.items():
        if len(codec_scores) >= 2:
            scores = list(codec_scores.values())
            stds.append(float(np.std(scores)))
    if not stds:
        return {"target": target, "n_refs": 0, "cc_std_median": float("nan")}
    return {
        "target": target,
        "n_refs": len(stds),
        "cc_std_median": float(np.median(stds)),
        "cc_std_max": float(max(stds)),
        "cc_std_p95": float(np.percentile(stds, 95)),
    }


def run_qsweep_mono(bake: Path, log_dir: Path) -> dict:
    """Use the butter parquets as a stand-in q-sweep corpus for
    monotonicity / range / tied checks. We score each codec's full
    19-q sweep (q=5..95) across 50 random images and compute strict
    mono / tied / range per the V6 gate definitions."""
    # Sample 50 sources from zenjpeg butter parquet
    df = pq.read_table(BUTTER_DIR / "zenjpeg.parquet").to_pandas()
    refs_all = sorted(df["ref_basename"].unique())
    rng = np.random.default_rng(seed=42)
    refs_sample = rng.choice(refs_all, size=min(50, len(refs_all)), replace=False)
    sub = df[df["ref_basename"].isin(refs_sample)].copy()
    sub = sub.sort_values(["ref_basename", "q"])
    feat_cols = [f"f{i}" for i in range(372)]
    feats = sub[feat_cols].to_numpy(dtype=np.float32, copy=False)
    preds = predict_features(bake, feats, post="raw")
    sub["pred"] = preds

    mono_curves = 0
    total_curves = 0
    tied_pairs = 0
    total_pairs = 0
    medRanges = []
    for ref, g in sub.groupby("ref_basename"):
        g = g.sort_values("q")
        scores = g["pred"].to_numpy()
        if len(scores) < 2:
            continue
        total_curves += 1
        # "Strict monotonicity" per V6 convention: non-decreasing on
        # the (low_q, high_q) order. Ties count as monotone (the
        # `tied_pct` metric tracks them separately). Matches V6 ship's
        # mono=0.9522 convention from `benchmarks/v_tuner_v6_methodology_2026-05-19.md`.
        diffs = np.diff(scores)
        if (diffs >= 0).all():
            mono_curves += 1
        # Tied pairs
        total_pairs += len(diffs)
        tied_pairs += int((diffs == 0).sum())
        medRanges.append(float(scores.max() - scores.min()))

    return {
        "n_curves": total_curves,
        "mono_pct": mono_curves / total_curves if total_curves else 0.0,
        "tied_pct": tied_pairs / total_pairs if total_pairs else 0.0,
        "medRange": float(np.median(medRanges)) if medRanges else float("nan"),
        "maxRange": float(np.max(medRanges)) if medRanges else float("nan"),
    }


def run_bake_verdict(bake: Path, out_md: Path) -> dict:
    """Run bake_verdict on canonical val corpora; parse SROCC per corpus.

    Note: bake_verdict expects files named
    `<corpus>_features_372col_2026-05-15.parquet` under the
    features-root, which only lives at the legacy 2026-05-15 path.
    """
    cmd = [
        str(BAKE_VERDICT_BIN),
        "--bake",
        str(bake),
        "--corpora",
        "cid22,kadid,tid,konjnd,aic3",
        "--features-root",
        "/mnt/v/zen/zensim-training/2026-05-15-full-features",
        "--output",
        str(out_md),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"WARN: bake_verdict failed: {proc.stderr[:500]}")
    # Parse the SROCC per corpus from the md file
    md_text = out_md.read_text() if out_md.exists() else ""
    srocc = {}
    cur = None
    for ln in md_text.splitlines():
        if ln.startswith("## ") and "(" in ln:
            # Format: "## CID22 (n=4292)" — extract corpus
            nm = ln[3:].split(" (")[0].strip()
            cur = nm
        if "| V_X bake |" in ln and cur:
            # Format: | V_X bake | <SROCC> | ...
            parts = [p.strip() for p in ln.split("|")]
            try:
                srocc[cur] = float(parts[2])
            except (IndexError, ValueError):
                pass
    return srocc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bake", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--verdict-md", type=Path, default=None)
    args = parser.parse_args()

    bake = args.bake
    out_md = args.out_md
    verdict_md = args.verdict_md or out_md.with_suffix(".panel.md")

    print(f"=== eval bake {bake} ===")
    data = load_anchor_features(V9_ANCHOR_PARQUET)
    preds = predict_features(bake, data["feats"], post="raw")

    print("=== per-band achievement ===")
    bands = score_per_band(preds, data["target_score"])
    for b in bands:
        print(
            f"  target={b['target']:6.1f} n={b['n']:5d} med={b['med']:7.3f} "
            f"abs_err={b['abs_err']:7.3f}"
        )

    print("=== range extension ===")
    range_ext = range_extension_check(preds, data)
    print(f"  worstfloor (target=0):  med={range_ext['worstfloor_med']:7.3f}")
    print(f"  lossless   (target=100): med={range_ext['lossless_med']:7.3f}")

    print("=== cross-codec consistency ===")
    cc = {}
    for t in [JOD_TARGET, JND_TARGET, 80.0, 90.0]:
        cc[t] = cross_codec_consistency(preds, data, t)
        print(
            f"  T={t:5.1f} n_refs={cc[t]['n_refs']:4d} "
            f"cc_std_median={cc[t]['cc_std_median']:6.3f}"
        )

    print("=== q-sweep monotonicity ===")
    qsweep = run_qsweep_mono(bake, out_md.parent)
    print(
        f"  mono={qsweep['mono_pct']:.4f} tied={qsweep['tied_pct']:.4f} "
        f"medRange={qsweep['medRange']:.2f}"
    )

    print("=== Mohammadi panel ===")
    srocc = run_bake_verdict(bake, verdict_md)
    print(f"  {srocc}")

    # Compose verdict markdown
    lines = []
    lines.append(f"# V9 verdict: {bake.name}\n")
    lines.append("## Per-band anchor achievement\n")
    lines.append("| target_score | n | median_pred | abs_err |")
    lines.append("|---:|---:|---:|---:|")
    for b in bands:
        lines.append(
            f"| {b['target']:.1f} | {b['n']} | {b['med']:.3f} | "
            f"{b['abs_err']:.3f} |"
        )
    lines.append("")
    lines.append("## Range extension\n")
    lines.append(
        f"- worstfloor (target=0):  n={range_ext['worstfloor_n']} "
        f"median_pred={range_ext['worstfloor_med']:.3f}"
    )
    lines.append(
        f"- lossless   (target=100): n={range_ext['lossless_n']} "
        f"median_pred={range_ext['lossless_med']:.3f}\n"
    )
    lines.append("## Cross-codec consistency\n")
    lines.append("| target | n_refs | cc_std_median | cc_std_max | cc_std_p95 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for t in [JOD_TARGET, JND_TARGET, 80.0, 90.0]:
        v = cc[t]
        lines.append(
            f"| {t:.1f} | {v['n_refs']} | {v['cc_std_median']:.3f} | "
            f"{v.get('cc_std_max', float('nan')):.3f} | "
            f"{v.get('cc_std_p95', float('nan')):.3f} |"
        )
    lines.append("")
    lines.append("## Q-sweep monotonicity (50 imgs × 19 q on zenjpeg butter parquet)\n")
    lines.append(f"- n_curves: {qsweep['n_curves']}")
    lines.append(f"- strict_mono: {qsweep['mono_pct']:.4f}")
    lines.append(f"- tied: {qsweep['tied_pct']:.4f}")
    lines.append(f"- medRange: {qsweep['medRange']:.2f}")
    lines.append(f"- maxRange: {qsweep['maxRange']:.2f}\n")
    lines.append("## Mohammadi panel SROCC\n")
    lines.append("| corpus | SROCC |")
    lines.append("|---|---:|")
    for k, v in srocc.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    # Gate verdicts
    lines.append("## Gate verdicts\n")
    lines.append("| gate | observed | gate value | verdict |")
    lines.append("|---|---:|---|:-:|")
    gates = [
        ("mono ≥ 0.9378", qsweep["mono_pct"], "≥ 0.9378",
         qsweep["mono_pct"] >= 0.9378),
        ("tied ≤ 5%", qsweep["tied_pct"], "≤ 0.05",
         qsweep["tied_pct"] <= 0.05),
        ("medRange ≥ 60", qsweep["medRange"], "≥ 60",
         qsweep["medRange"] >= 60.0),
        ("worstfloor med ≤ 5", range_ext["worstfloor_med"], "≤ 5",
         range_ext["worstfloor_med"] <= 5.0),
        ("lossless med ≥ 95", range_ext["lossless_med"], "≥ 95",
         range_ext["lossless_med"] >= 95.0),
        ("JND abs_err ≤ 2", next(b["abs_err"] for b in bands if b["target"] == JND_TARGET), "≤ 2",
         next(b["abs_err"] for b in bands if b["target"] == JND_TARGET) <= 2.0),
        ("JOD abs_err ≤ 2", next(b["abs_err"] for b in bands if b["target"] == JOD_TARGET), "≤ 2",
         next(b["abs_err"] for b in bands if b["target"] == JOD_TARGET) <= 2.0),
        ("PJND cc_std_median ≤ 5", cc[JND_TARGET]["cc_std_median"], "≤ 5",
         cc[JND_TARGET]["cc_std_median"] <= 5.0),
        ("JOD cc_std_median ≤ 5", cc[JOD_TARGET]["cc_std_median"], "≤ 5",
         cc[JOD_TARGET]["cc_std_median"] <= 5.0),
        ("T80 cc_std_median ≤ 5", cc[80.0]["cc_std_median"], "≤ 5",
         cc[80.0]["cc_std_median"] <= 5.0),
        ("T90 cc_std_median ≤ 5", cc[90.0]["cc_std_median"], "≤ 5",
         cc[90.0]["cc_std_median"] <= 5.0),
    ]
    n_pass = 0
    for name, observed, gate_str, passed in gates:
        v = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        lines.append(f"| {name} | {observed:.4f} | {gate_str} | **{v}** |")
    lines.append("")
    lines.append(f"**Total: {n_pass}/{len(gates)} gates pass**\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")
    print(f"GATES: {n_pass}/{len(gates)} pass")


if __name__ == "__main__":
    main()
