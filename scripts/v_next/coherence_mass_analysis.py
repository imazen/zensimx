#!/usr/bin/env python3
"""Coherence-mechanism test: is M3a determined by WHERE a bake's contribution mass sits?

Registered in `benchmarks/sota944_campaign_2026-08-03.md` appendix D (written
before any number here was computed). Reads:

  - `benchmarks/slot_decomposability_2026-08-04.tsv` — the source-derived
    E/A/N classification (the ONLY place the classification lives).
  - per-bake `bake_contrib` TSVs (the contribution owner; this script never
    re-derives a contribution).
  - `/mnt/v/output/zensim/reports/fulleval/*.fulleval.json` — `m3a_coherence`
    and the confound annotations.

Stats come from `scripts/lib/zen_stats` (the canonical owner, which shells the
`panel` binary) — nothing statistical is computed here.

Emits the per-bake mass table (TSV) and the correlation summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib import zen_stats  # noqa: E402

N_SCALES = 4
N_CH = 3
BASIC_PER_CH = 13
V2_PER_CH = 29
APPEND_PER_CH = 17
APPEND2_PER_SCALE = 5

CH_Y = 1
CH_B = 2


def load_classification(path: str):
    """Return {block: {local_key: class}} plus the whole-block rows."""
    basic: dict[int, str] = {}
    v2: dict[int, str] = {}
    append_default: dict[int, str] = {}
    append_ch: dict[tuple[int, int], str] = {}  # (local, ch) -> class override
    append2: dict[int, str] = {}
    v1pool_class = None
    append_b_scale0 = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "block":
                continue
            block, local, _name, cls, _pool, _integ, channels = parts[:7]
            if block == "basic":
                basic[int(local)] = cls
            elif block == "v1pool":
                v1pool_class = cls
            elif block == "v2":
                v2[int(local)] = cls
            elif block == "append":
                if local == "ALL":
                    append_b_scale0 = cls
                    continue
                li = int(local)
                if channels == "all":
                    append_default[li] = cls
                else:
                    for c in channels.split(","):
                        ci = {"X": 0, "Y": 1, "B": 2}[c]
                        append_ch[(li, ci)] = cls
            elif block == "append2":
                append2[int(local)] = cls
    assert v1pool_class == "N", v1pool_class
    assert append_b_scale0 == "N", append_b_scale0
    assert len(basic) == BASIC_PER_CH, len(basic)
    assert len(v2) == V2_PER_CH, len(v2)
    assert len(append2) == APPEND2_PER_SCALE, len(append2)
    return basic, v1pool_class, v2, append_default, append_ch, append2


def class_array(width: int, tables) -> list[str]:
    """Per-column decomposability class for a given bake input width."""
    basic, v1pool_class, v2, append_default, append_ch, append2 = tables
    cls = ["?"] * width
    # basic f0-155: scale*39 + ch*13 + local
    for s in range(N_SCALES):
        for ch in range(N_CH):
            for lo in range(BASIC_PER_CH):
                i = s * (N_CH * BASIC_PER_CH) + ch * BASIC_PER_CH + lo
                if i < width:
                    cls[i] = basic[lo]
    # v1 pools f156-371 (whole block)
    for i in range(156, min(372, width)):
        cls[i] = v1pool_class
    # v2 f372-719: 372 + scale*87 + ch*29 + local
    for s in range(N_SCALES):
        for ch in range(N_CH):
            for lo in range(V2_PER_CH):
                i = 372 + s * (N_CH * V2_PER_CH) + ch * V2_PER_CH + lo
                if i < width:
                    cls[i] = v2[lo]
    # append f720-923: 720 + scale*51 + ch*17 + local
    for s in range(N_SCALES):
        for ch in range(N_CH):
            for lo in range(APPEND_PER_CH):
                i = 720 + s * (N_CH * APPEND_PER_CH) + ch * APPEND_PER_CH + lo
                if i >= width:
                    continue
                if ch == CH_B and s == 0:
                    cls[i] = "N"  # APPEND_SKIP_B_SCALE0
                elif (lo, ch) in append_ch:
                    cls[i] = append_ch[(lo, ch)]
                else:
                    cls[i] = append_default[lo]
    # append2 f924-943: 924 + scale*5 + local (Y-only, no channel axis)
    for s in range(N_SCALES):
        for lo in range(APPEND2_PER_SCALE):
            i = 924 + s * APPEND2_PER_SCALE + lo
            if i < width:
                cls[i] = append2[lo]
    assert "?" not in cls, f"unmapped column at width {width}: {cls.index('?')}"
    return cls


def scale_array(width: int) -> list[int]:
    """Per-column pyramid scale (0 = finest). POST-HOC diagnostic axis.

    Layouts (verified in source): basic `scale*39 + ch*13 + slot`; the v1
    peak/masked/IW blocks each `base + scale*18 + ch*6 + slot` with bases
    156/228/300 (`metric.rs:4346,4409,4443`); v2 `372 + scale*87 + ...`;
    append `720 + scale*51 + ...`; append2 `924 + scale*5 + local`.
    """
    sc = [-1] * width
    for i in range(width):
        if i < 156:
            sc[i] = i // (N_CH * BASIC_PER_CH)
        elif i < 372:
            base = 156 if i < 228 else (228 if i < 300 else 300)
            sc[i] = (i - base) // (N_CH * 6)
        elif i < 720:
            sc[i] = (i - 372) // (N_CH * V2_PER_CH)
        elif i < 924:
            sc[i] = (i - 720) // (N_CH * APPEND_PER_CH)
        else:
            sc[i] = (i - 924) // APPEND2_PER_SCALE
    assert min(sc) == 0 and max(sc) < N_SCALES, (min(sc), max(sc))
    return sc


def read_contrib(path: str) -> tuple[list[int], list[float]]:
    idx: list[int] = []
    mass: list[float] = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_idx = header.index("idx")
        i_mass = header.index("mean_abs")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            idx.append(int(p[i_idx]))
            mass.append(float(p[i_mass]))
    return idx, mass


def coarse_decay_of(bake_path: str, fulleval: dict) -> str:
    """'yes' / 'no' / 'unknown' from the embedded repro argv, else spec.json."""
    argv = None
    rep = fulleval.get("repro")
    if isinstance(rep, dict):
        argv = rep.get("argv")
    if argv is None:
        spec = bake_path + ".spec.json"
        if os.path.exists(spec):
            try:
                sj = json.load(open(spec))
            except Exception:
                sj = {}
            argv = sj.get("argv") or (sj.get("repro") or {}).get("argv")
    if argv is None:
        return "unknown"
    text = " ".join(argv) if isinstance(argv, list) else str(argv)
    return "yes" if "--coarse-decay" in text else "no"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", required=True)
    ap.add_argument("--contrib-dir", required=True)
    ap.add_argument("--fulleval-dir", default="/mnt/v/output/zensim/reports/fulleval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    tables = load_classification(args.classification)
    cls_cache: dict[int, list[str]] = {}
    scale_cache: dict[int, list[int]] = {}

    rows = []
    for p in sorted(glob.glob(os.path.join(args.fulleval_dir, "*.fulleval.json"))):
        d = json.load(open(p))
        if d.get("m3a_coherence") is None:
            continue
        name = os.path.basename(p).replace(".fulleval.json", "")
        tsv = os.path.join(args.contrib_dir, name + ".tsv")
        if not os.path.exists(tsv):
            print(f"MISSING contrib TSV for {name}", file=sys.stderr)
            continue
        model = d.get("model") or {}
        width = d.get("n_inputs") or model.get("n_inputs")
        if width not in cls_cache:
            cls_cache[width] = class_array(width, tables)
        cls = cls_cache[width]
        idx, mass = read_contrib(tsv)
        assert len(idx) == width, (name, len(idx), width)
        tot = {"E": 0.0, "A": 0.0, "N": 0.0}
        for i, m in zip(idx, mass):
            tot[cls[i]] += m
        total = tot["E"] + tot["A"] + tot["N"]
        if total <= 0:
            print(f"ZERO total mass for {name}", file=sys.stderr)
            continue
        # POST-HOC diagnostic axes (NOT registered; exploratory, labelled as such)
        if width not in scale_cache:
            scale_cache[width] = scale_array(width)
        sc = scale_cache[width]
        by_scale = [0.0] * N_SCALES
        for i, m in zip(idx, mass):
            by_scale[sc[i]] += m
        hhi = sum((m / total) ** 2 for m in mass)
        n_live = sum(1 for m in mass if m > 0)
        bake = d.get("bake") or ""
        rows.append(
            dict(
                name=name,
                m3a=d["m3a_coherence"],
                m3=d.get("m3_coherence"),
                exact_frac=tot["E"] / total,
                approx_frac=tot["A"] / total,
                nondecomp_frac=tot["N"] / total,
                decomp_frac=(tot["E"] + tot["A"]) / total,
                fine_frac=(by_scale[0] + by_scale[1]) / total,
                s0_frac=by_scale[0] / total,
                s3_frac=by_scale[3] / total,
                mean_scale=sum(s * by_scale[s] for s in range(N_SCALES)) / total,
                eff_n=1.0 / hhi,
                n_live=n_live,
                n_inputs=width,
                n_layers=model.get("n_layers"),
                depth="linear" if model.get("n_layers") == 1 else "mlp",
                coarse_decay=coarse_decay_of(bake, d),
                mass_total=total,
                bake=bake,
            )
        )

    rows.sort(key=lambda r: -r["m3a"])
    cols = [
        "name", "m3a", "m3", "exact_frac", "approx_frac", "nondecomp_frac",
        "decomp_frac", "fine_frac", "s0_frac", "s3_frac", "mean_scale", "eff_n",
        "n_live", "n_inputs", "n_layers", "depth", "coarse_decay",
        "mass_total", "bake",
    ]
    git = subprocess.run(
        ["git", "-C", "/home/lilith/work/zen/zensimx", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    with open(args.out, "w") as fh:
        fh.write("# coherence mass-placement table (registered appendix D)\n")
        fh.write(f"# git: {git}\n")
        fh.write(f"# classification: {args.classification}\n")
        fh.write(f"# contrib dir: {args.contrib_dir}\n")
        fh.write("# mass = Sum mean|Delta| (bake_contrib exact standardized-zero mean-ablation),\n")
        fh.write("#        corpus = regime-native cid22val, 4292 rows, no target\n")
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(
                f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols
            ) + "\n")

    # --- correlations (canonical stats owner only; ONE batched panel call) ---
    m3a = [r["m3a"] for r in rows]
    jobs = []
    for key in ("exact_frac", "decomp_frac", "nondecomp_frac", "approx_frac",
                "fine_frac", "s0_frac", "s3_frac", "mean_scale", "eff_n", "m3"):
        jobs.append((f"all::{key}", [r[key] for r in rows], m3a))
    groups_all: dict[tuple[str, object], list] = {}
    for gkey in ("n_inputs", "depth", "coarse_decay"):
        for r in rows:
            groups_all.setdefault((gkey, r[gkey]), []).append(r)
    for (gkey, g), rs in sorted(groups_all.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        if len(rs) < 4:
            continue
        ys = [r["m3a"] for r in rs]
        jobs.append((f"{gkey}={g}::exact_frac", [r["exact_frac"] for r in rs], ys))
        jobs.append((f"{gkey}={g}::decomp_frac", [r["decomp_frac"] for r in rs], ys))
    res = {j["label"]: j for j in zen_stats.panel_batch(jobs)}

    labels = {
        "exact_frac": "PRIMARY exact_mass_fraction",
        "decomp_frac": "SECONDARY decomposable_mass_fraction",
        "nondecomp_frac": "nondecomposable_fraction",
        "approx_frac": "approx_fraction",
        "fine_frac": "[post-hoc] fine_mass_fraction (scale 0+1)",
        "s0_frac": "[post-hoc] scale-0 mass fraction",
        "s3_frac": "[post-hoc] scale-3 (coarsest) mass fraction",
        "mean_scale": "[post-hoc] mass-weighted mean pyramid scale",
        "eff_n": "[post-hoc] effective #features (1/HHI of mass)",
        "m3": "[post-hoc] M3 (signal-fold coherence)",
    }
    out = []
    out.append(f"n = {len(rows)} bakes with measured m3a_coherence")
    for key, label in labels.items():
        r0 = res[f"all::{key}"]
        out.append(
            f"{label:44s} SROCC {r0['srocc_signed']:+.4f} (|{r0['srocc']:.4f}|)"
            f"   PLCC {r0['plcc_raw']:+.4f}   n {r0['n']}"
        )
    out.append("")
    out.append("within-group (DESCRIPTIVE ONLY, small n):")
    for (gkey, g), rs in sorted(groups_all.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        ys = [r["m3a"] for r in rs]
        if len(rs) < 4:
            out.append(f"  {gkey}={str(g):<8} n={len(rs):<3} (n<4, not correlated)"
                       f"  m3a range [{min(ys):.3f},{max(ys):.3f}]")
            continue
        a = res[f"{gkey}={g}::exact_frac"]
        b = res[f"{gkey}={g}::decomp_frac"]
        out.append(
            f"  {gkey}={str(g):<8} n={len(rs):<3} SROCC(exact) {a['srocc_signed']:+.4f}  "
            f"SROCC(decomp) {b['srocc_signed']:+.4f}  m3a range [{min(ys):.3f},{max(ys):.3f}]"
        )
    text = "\n".join(out)
    with open(args.summary, "w") as fh:
        fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
