"""Formula revision 7: how far each feature of revision 6's arithmetic ("current") and of revision 7's ("stable")
sits from the exact f64 reference, on near-identical pairs.

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Formula revision 7"). Inputs are the dumps of
`tier_audit_features` (`ZENSIM_FEATCANON_DUMP`, one tier): `prod_<tier>_<pair>.f64bin` at each revision
(`ZENSIM_FORMULA_REV`), and `exact_<tier>_<pair>.f64bin` (`ZENSIM_FEATCANON=exact`, the f64 oracle bodies, built with
`--features featcanon-oracle`), plus `names_<pair>.tsv`. `--exact-stable` gives the second arm its own exact dumps
(revision 7's); without it both arms are measured against `--exact`. The first measurement (2026-10-24, before
revision 7 existed) compared two builds of one revision, the second with the forms behind a since-removed cfg.

  python3 rev7_forms.py --current DIR --stable DIR --exact DIR [--exact-stable DIR] [--tier v4x] [--out OUT.json]

Per feature: the median over pairs of |x - exact| / max(|exact|, floor), floor = 1e-12; per family: the worst
feature's median and the share of the family's features whose median error exceeds 1e-3.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(d: Path, mode: str, tier: str, pair: str):
    p = d / f"{mode}_{tier}_{pair}.f64bin"
    return np.fromfile(p, dtype="<f8") if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", required=True)
    ap.add_argument("--stable", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--exact-stable", help="exact dumps for the --stable arm (default: --exact)")
    ap.add_argument("--tier", default="v4x")
    ap.add_argument("--out")
    a = ap.parse_args()
    cur, sta, exa = Path(a.current), Path(a.stable), Path(a.exact)
    exs = Path(a.exact_stable) if a.exact_stable else exa
    pairs = sorted(p.name[len("names_"):-len(".tsv")] for p in cur.glob("names_*.tsv"))
    names = None
    rel_c, rel_s = defaultdict(list), defaultdict(list)
    for pair in pairs:
        if names is None:
            names = [line.split("\t") for line in (cur / f"names_{pair}.tsv").read_text().splitlines()]
        c, s = load(cur, "prod", a.tier, pair), load(sta, "prod", a.tier, pair)
        e, es = load(exa, "exact", a.tier, pair), load(exs, "exact", a.tier, pair)
        if c is None or s is None or e is None or es is None:
            continue
        n = min(len(c), len(s), len(e), len(es))
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
    for label, v in (("current", allc), ("stable", alls)):
        print(f"{label:8s} per-feature median error: p50 {np.quantile(v, 0.5):.2e} p90 {np.quantile(v, 0.9):.2e} "
              f"p99 {np.quantile(v, 0.99):.2e} max {v.max():.2e}")
    print(f"stable better on {int((alls < allc).sum())}, worse on {int((alls > allc).sum())}, "
          f"tied on {int((alls == allc).sum())} features")
    print(f"{'family':24s} {'n':>5s} {'worst cur':>11s} {'worst stb':>11s} {'cur>1e-3':>9s} {'stb>1e-3':>9s}  worst feature (current)")
    summary = {}
    for family, rows in sorted(fam.items()):
        wc = max(rows, key=lambda r: r[0])
        ws = max(r[1] for r in rows)
        oc = sum(r[0] > 1e-3 for r in rows) / len(rows)
        os_ = sum(r[1] > 1e-3 for r in rows) / len(rows)
        summary[family] = {"n": len(rows), "worst_current": wc[0], "worst_stable": ws, "share_current_over_1e-3": oc,
                           "share_stable_over_1e-3": os_, "worst_feature": f"f{wc[2]} {wc[3]}"}
        print(f"{family:24s} {len(rows):5d} {wc[0]:11.3e} {ws:11.3e} {oc:9.1%} {os_:9.1%}  f{wc[2]} {wc[3]}")
    if a.out:
        Path(a.out).write_text(json.dumps({"pairs": pairs, "tier": a.tier, "families": summary,
                                           "features": per_feature}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
