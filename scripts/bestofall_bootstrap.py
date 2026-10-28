#!/usr/bin/env python3
"""Paired bootstrap of the wave's candidates against SHIPPED D and the 944
leaders, over the per-pair vectors both scorers already produced.

STAT OWNER: `panel` (via `scripts/lib/zen_stats.panel_batch_indexed`, the
index-set resample shape). Nothing here computes a correlation — a
scipy-in-a-bootstrap-loop is the banned pattern this replaces, and the batch
form declares each base vector ONCE and sends ~n integers per resample.

REFERENCE-CLUSTERED resampling where a ref id is available: CID22's 4,292 pairs
come from 49 references, so an i.i.d. pair bootstrap would badly understate the
interval. Absent ref ids the fallback is an i.i.d. pair bootstrap, and it SAYS
SO in the output rather than pretending otherwise.

Usage:
  bestofall_bootstrap.py --a <cand.fulleval.json> --b <ref.fulleval.json>
                         [--corpora cid22,konjnd,aic3,...] [-B 2000] [--seed N]
"""
import argparse
import json
import os
import random
import sys

# Repo-relative, like every other driver in this lane. The first draft
# hardcoded `~/work/zen/zensimx/scripts` — the PRIMARY checkout — from a script
# that runs in a sibling workspace, so a lane-local change to the stat owner
# would have been invisible to the lane's own bootstrap. (Review 2026-09-06.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from lib import zen_stats  # noqa: E402

DEFAULT_CORPORA = "cid22,konjnd,aic3,tid,kadid,csiq,live,hfnlproxy"


# The target key is per-corpus: `mos` on the MOS corpora, `jnd` on the
# threshold ones, `ssim2`/`target` elsewhere. Take whichever non-`pred` list the
# block carries rather than guessing — and refuse if there is more than one.
TARGET_KEYS = ("mos", "jnd", "ssim2", "target", "human_score", "score")


def vectors(d, corpus):
    pp = (d.get("per_pair") or {}).get(corpus)
    if not isinstance(pp, dict):
        return None, None
    cand = [k for k in TARGET_KEYS if isinstance(pp.get(k), list)]
    if len(cand) != 1 or not isinstance(pp.get("pred"), list):
        return None, None
    return pp["pred"], pp[cand[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="candidate fulleval")
    ap.add_argument("--b", required=True, help="reference fulleval")
    ap.add_argument("--corpora", default=DEFAULT_CORPORA)
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    a = ap.parse_args()

    da, db = json.load(open(a.a)), json.load(open(a.b))
    la = a.label_a or da.get("name") or os.path.basename(a.a)
    lb = a.label_b or db.get("name") or os.path.basename(a.b)
    rng = random.Random(a.seed)

    print(f"paired bootstrap  B={a.B}  seed={a.seed}")
    print(f"  A = {la}\n  B = {lb}")
    print(f"{'corpus':<12} {'n':>6} {'A':>10} {'B':>10} {'delta':>10} "
          f"{'95% CI':>22}  verdict")
    for corpus in a.corpora.split(","):
        pa, ta = vectors(da, corpus)
        pb, tb = vectors(db, corpus)
        if not pa or not pb:
            print(f"{corpus:<12} {'—':>6} {'—':>10} {'—':>10} {'—':>10} "
                  f"{'NOT MEASURED':>22}  (one side has no per-pair vector)")
            continue
        if len(pa) != len(pb):
            print(f"{corpus:<12} {len(pa):>6} — REFUSED: A has {len(pa)} pairs, "
                  f"B has {len(pb)}; the rows are not the same rows")
            continue
        n = len(pa)
        # Targets must agree row-for-row or the vectors are not aligned.
        worst = max(abs(x - y) for x, y in zip(ta, tb))
        if worst > 1e-9:
            print(f"{corpus:<12} {n:>6} — REFUSED: targets differ by up to "
                  f"{worst:.3e}; the two fullevals are not row-aligned")
            continue
        bases = {"pa": pa, "pb": pb, "t": ta}
        jobs = [("point_a", "pa", "t", None), ("point_b", "pb", "t", None)]
        for b in range(a.B):
            idx = [rng.randrange(n) for _ in range(n)]
            jobs.append((f"ra{b}", "pa", "t", idx))
            jobs.append((f"rb{b}", "pb", "t", idx))
        rows = zen_stats.panel_batch_indexed(bases, jobs, stats="srocc")
        by = {r["label"]: r for r in rows}
        sgn = -1.0 if corpus == "konjnd" else 1.0
        pa_pt = sgn * by["point_a"].get("srocc_signed", by["point_a"]["srocc"])
        pb_pt = sgn * by["point_b"].get("srocc_signed", by["point_b"]["srocc"])
        deltas = sorted(
            (sgn * by[f"ra{b}"].get("srocc_signed", by[f"ra{b}"]["srocc"]))
            - (sgn * by[f"rb{b}"].get("srocc_signed", by[f"rb{b}"]["srocc"]))
            for b in range(a.B)
        )
        lo = deltas[int(0.025 * len(deltas))]
        hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
        verdict = "A WINS" if lo > 0 else ("A LOSES" if hi < 0 else "tie")
        print(f"{corpus:<12} {n:>6} {pa_pt:>10.5f} {pb_pt:>10.5f} "
              f"{pa_pt - pb_pt:>+10.5f} [{lo:>+9.5f},{hi:>+9.5f}]  {verdict}")
    print("\ni.i.d. PAIR resample (fullevals carry no per-pair reference id), so "
          "these intervals are NARROWER than a reference-clustered one would be. "
          "Read them as a floor on the uncertainty, not a ceiling.")


if __name__ == "__main__":
    main()
