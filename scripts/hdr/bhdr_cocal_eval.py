#!/usr/bin/env python3
"""Evaluate a BHdr dial candidate vs the shipped bake on the deployment regime.

Three questions, per the co-cal task + user constraints:
  1. NEGATIVES preserved? (lower-bound probe: real UPIQ HDR heavy pairs + the
     corruption grid reach the low/negative region — "test completely different
     pairs to find the lower score bounds").
  2. RANK-INVARIANT? (per-stratum UPIQ SROCC must be IDENTICAL to shipped — a
     dial re-fit changes calibration, never rank).
  3. What is the score DISTRIBUTION (range, negatives, median) on real HDR?

  usage: bhdr_cocal_eval.py <cand.bin> [--label NAME]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

# parse OUR args first, then neutralize sys.argv before exec-ing the instrument
# head (it builds its own argparse at module scope).
ap = argparse.ArgumentParser()
ap.add_argument("cand")
ap.add_argument("--label", default="candidate")
a = ap.parse_args()
sys.argv = ["xdi"]

REPO = os.path.expanduser("~/work/zen/zensimx")
PROBE = "/mnt/v/output/zensim-multicodec-probe"
SHIPPED = f"{REPO}/zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin"
HDR_FEATS = f"{PROBE}/upiq_features_372_pulinear.parquet"
JOD_CSV = "/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv"

_spec = importlib.util.spec_from_file_location(
    "xdi", f"{REPO}/scripts/hdr/upiq_crossdomain_instrument.py")
_src = open(f"{REPO}/scripts/hdr/upiq_crossdomain_instrument.py").read()
_ns = {}
exec(compile(_src.split("# ---- SDR half")[0], "xdi_head", "exec"), _ns)
score_bake = _ns["score_bake"]

t = pq.read_table(HDR_FEATS)
n = t.num_rows
# UPIQ HDR strata: rows 0..139 narwaria, 140..379 korshunov (positional, §8.1)
strat = np.array(["narwaria"] * 140 + ["korshunov"] * (n - 140))
jod = None
if os.path.exists(JOD_CSV):
    import csv
    jr = list(csv.DictReader(open(JOD_CSV)))
    if len(jr) == n:
        col = "JOD" if "JOD" in jr[0] else list(jr[0])[-1]
        jod = np.array([float(r[col]) for r in jr])

sh = score_bake(SHIPPED, t)
cd = score_bake(a.cand, t)


def dist(name, s):
    neg = int((s < 0).sum())
    print(f"  {name:10s}: range [{s.min():7.1f}, {s.max():6.1f}]  median {np.median(s):6.1f}  "
          f"negatives {neg:3d}/{n}  p5 {np.percentile(s,5):6.1f}")


print(f"\n=== score distribution on real UPIQ HDR (n={n}, deployment PU-linear regime) ===")
dist("shipped", sh)
dist(a.label, cd)
print(f"  Δ(cand−shipped): mean {np.mean(cd-sh):+.2f}, median {np.median(cd-sh):+.2f}, "
      f"|Δ| p95 {np.percentile(np.abs(cd-sh),95):.2f}")

print("\n=== rank-invariance check: SROCC(shipped, cand) — must be ~1.000 ===")
print(f"  overall SROCC(shipped,cand): {spearmanr(sh,cd).correlation:.5f}")
for s in ("narwaria", "korshunov"):
    m = strat == s
    print(f"  {s:10s} SROCC(shipped,cand): {spearmanr(sh[m],cd[m]).correlation:.5f}")

if jod is not None:
    print("\n=== UPIQ SROCC vs JOD (must be IDENTICAL shipped vs cand — dial is rank-inv) ===")
    for s in ("narwaria", "korshunov"):
        m = strat == s
        print(f"  {s:10s}: shipped {spearmanr(sh[m],jod[m]).correlation:+.4f}  "
              f"{a.label} {spearmanr(cd[m],jod[m]).correlation:+.4f}")
    print(f"  pooled    : shipped {spearmanr(sh,jod).correlation:+.4f}  "
          f"{a.label} {spearmanr(cd,jod).correlation:+.4f}")
