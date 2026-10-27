"""Formula revision 7: how far each feature of revision 6's arithmetic ("current") and of revision 7's ("stable")
sits from the exact f64 reference, on near-identical pairs.

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Formula revision 7"). Inputs are the dumps of
`tier_audit_features` (`ZENSIM_FEATCANON_DUMP`, one tier): `prod_<tier>_<pair>.f64bin` at each revision
(`ZENSIM_FORMULA_REV`), and `exact_<tier>_<pair>.f64bin` (`ZENSIM_FEATCANON=exact`, the f64 oracle bodies, built with
`--features featcanon-oracle`), plus `names_<pair>.tsv`. `--exact-stable` gives the second arm its own exact dumps
(revision 7's); without it both arms are measured against `--exact`. The first measurement (2026-10-24, before
revision 7 existed) compared two builds of one revision, the second with the forms behind a since-removed cfg.
Without `--stable` it measures the one arm `--current` against `--exact` (the native HDR precision probe).
`--predict BIN --bake NAME=BAKE ...` also scores each arm's dumps and the exact dumps with every bake through
`predict_features_with_bake --f64-wire --production` (identity layout, dimensions 0x0) and reports the score
differences.

  python3 rev7_forms.py --current DIR [--stable DIR] --exact DIR [--exact-stable DIR] [--tier v4x] [--out OUT.json]
      [--predict BIN --bake NAME=BAKE ...]

Per feature: the median over pairs of |x - exact| / max(|exact|, floor), floor = 1e-12; per family: the worst
feature's median and the share of the family's features whose median error exceeds 1e-3.
"""

import argparse
import json
import os
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(d: Path, mode: str, tier: str, pair: str):
    p = d / f"{mode}_{tier}_{pair}.f64bin"
    return np.fromfile(p, dtype="<f8") if p.exists() else None


def scores(predict, bake, rows):
    """Score identity-layout f64 rows with one bake through the native feature-row owner."""
    with tempfile.NamedTemporaryFile(dir=os.environ.get("TMPDIR", Path.home() / "tmp"), suffix=".bin") as f:
        f.write(struct.pack("<II", rows.shape[1], rows.shape[0]) + rows.astype("<f8").tobytes())
        f.flush()
        out = subprocess.run([predict, "--bake", bake, "--features-file", f.name, "--f64-wire", "--production"],
                             capture_output=True, text=True, check=True).stdout.split()
    if len(out) != len(rows):
        raise ValueError(f"{bake}: {len(out)} scores for {len(rows)} rows")
    return np.array([float(v) for v in out])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", required=True)
    ap.add_argument("--stable", help="second arm (default: measure --current alone)")
    ap.add_argument("--exact", required=True)
    ap.add_argument("--exact-stable", help="exact dumps for the --stable arm (default: --exact)")
    ap.add_argument("--tier", default="v4x")
    ap.add_argument("--out")
    ap.add_argument("--predict", help="predict_features_with_bake binary for the score comparison")
    ap.add_argument("--bake", action="append", default=[], help="NAME=BAKE to score (with --predict)")
    a = ap.parse_args()
    if bool(a.predict) != bool(a.bake):
        ap.error("--predict and --bake go together")
    single = a.stable is None
    cur, exa = Path(a.current), Path(a.exact)
    sta = cur if single else Path(a.stable)
    exs = Path(a.exact_stable) if a.exact_stable else exa
    pairs = sorted(p.name[len("names_"):-len(".tsv")] for p in cur.glob("names_*.tsv"))
    names = None
    rel_c, rel_s = defaultdict(list), defaultdict(list)
    vectors, kept = defaultdict(list), []
    for pair in pairs:
        if names is None:
            names = [line.split("\t") for line in (cur / f"names_{pair}.tsv").read_text().splitlines()]
        c, s = load(cur, "prod", a.tier, pair), load(sta, "prod", a.tier, pair)
        e, es = load(exa, "exact", a.tier, pair), load(exs, "exact", a.tier, pair)
        if c is None or s is None or e is None or es is None:
            continue
        n = min(len(c), len(s), len(e), len(es))
        kept.append(pair)
        for arm, v in (("current", c), ("stable", s), ("exact", e), ("exact_stable", es)):
            vectors[arm].append(v[:n])
        floor = np.maximum(np.abs(e[:n]), 1e-12)
        floor_s = np.maximum(np.abs(es[:n]), 1e-12)
        for i in range(n):
            if not (np.isfinite(e[i]) and np.isfinite(es[i]) and np.isfinite(c[i]) and np.isfinite(s[i])):
                continue
            rel_c[i].append(abs(c[i] - e[i]) / floor[i])
            rel_s[i].append(abs(s[i] - es[i]) / floor_s[i])
    fam = defaultdict(list)
    per_feature = []
    for i in sorted(rel_c):
        fid, family, name = names[i][0], names[i][1], names[i][2] if len(names[i]) > 2 else ""
        mc, ms = float(np.median(rel_c[i])), float(np.median(rel_s[i]))
        fam[family].append((mc, ms, fid, name))
        per_feature.append({"id": fid, "family": family, "name": name, "current": mc, "stable": ms,
                            "current_p90": float(np.quantile(rel_c[i], 0.9)), "stable_p90": float(np.quantile(rel_s[i], 0.9))})
    print(f"{len(pairs)} pairs, {len(per_feature)} features")
    allc = np.array([f["current"] for f in per_feature])
    alls = np.array([f["stable"] for f in per_feature])
    for label, v in (("current", allc),) if single else (("current", allc), ("stable", alls)):
        print(f"{label:8s} per-feature median error: p50 {np.quantile(v, 0.5):.2e} p90 {np.quantile(v, 0.9):.2e} "
              f"p99 {np.quantile(v, 0.99):.2e} max {v.max():.2e}")
    if not single:
        print(f"stable better on {int((alls < allc).sum())}, worse on {int((alls > allc).sum())}, "
              f"tied on {int((alls == allc).sum())} features")
    stb_head = "" if single else f" {'worst stb':>11s}"
    stb_share = "" if single else f" {'stb>1e-3':>9s}"
    print(f"{'family':24s} {'n':>5s} {'worst cur':>11s}{stb_head} {'cur>1e-3':>9s}{stb_share}  worst feature (current)")
    summary = {}
    for family, rows in sorted(fam.items()):
        wc = max(rows, key=lambda r: r[0])
        ws = max(r[1] for r in rows)
        oc = sum(r[0] > 1e-3 for r in rows) / len(rows)
        os_ = sum(r[1] > 1e-3 for r in rows) / len(rows)
        summary[family] = {"n": len(rows), "worst_current": wc[0], "worst_stable": ws, "share_current_over_1e-3": oc,
                           "share_stable_over_1e-3": os_, "worst_feature": f"f{wc[2]} {wc[3]}"}
        stb = "" if single else f" {ws:11.3e}"
        stb_s = "" if single else f" {os_:9.1%}"
        print(f"{family:24s} {len(rows):5d} {wc[0]:11.3e}{stb} {oc:9.1%}{stb_s}  f{wc[2]} {wc[3]}")
    score_report = {}
    for spec in a.bake:
        name, bake = spec.split("=", 1)
        width = min(len(r) for r in vectors["current"])
        arm_scores = {arm: scores(a.predict, bake, np.array([r[:width] for r in vectors[arm]]))
                      for arm in (("current", "exact") if single else ("current", "stable", "exact", "exact_stable"))}
        report = {}
        for arm, ref in (("current", "exact"),) if single else (("current", "exact"), ("stable", "exact_stable")):
            d = np.abs(arm_scores[arm] - arm_scores[ref])
            report[arm] = {"max_abs": float(d.max()), "median_abs": float(np.median(d)),
                           "worst_pair": kept[int(np.argmax(d))]}
            print(f"{name} {arm}: |score - exact score| max {d.max():.3e} median {np.median(d):.3e} "
                  f"(worst {report[arm]['worst_pair']})")
        score_report[name] = {"bake": bake, "pairs": kept, "scores": {k: v.tolist() for k, v in arm_scores.items()},
                              **report}
    if a.out:
        Path(a.out).write_text(json.dumps({"pairs": pairs, "tier": a.tier, "families": summary,
                                           "features": per_feature, "scores": score_report}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
