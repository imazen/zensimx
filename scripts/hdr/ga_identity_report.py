#!/usr/bin/env python3
"""G-A / R1 sub-domain identity report: BHdr on 203-nit PQ re-encodes of SDR
content vs B on the same content through the native SDR path.

Gate (PLAN_HDR G-HDR-SDR-CONSISTENCY, campaign task #13 G-A):
p95 |Δdial| ≤ 2 points AND rank agreement (SROCC of the two dial vectors)
≥ 0.99. The SDR case is the sub-domain limit of the HDR path — no seam.

Joins the two feature parquets by q (= UPIQ subjective-csv row index),
forwards each bake via the bake_verdict pred-dump route, reports aggregate +
per-dataset (live = clean leg) + the worst offenders.

  usage: ga_identity_report.py [--out report.md]
"""
import argparse
import csv
import os

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

import importlib.util
spec = importlib.util.spec_from_file_location(
    "xdi", os.path.join(os.path.dirname(__file__), "upiq_crossdomain_instrument.py"))

REPO = os.path.expanduser("~/work/zen/zensimx")
PROBE = "/mnt/v/output/zensim-multicodec-probe"

ap = argparse.ArgumentParser()
ap.add_argument("--b-bake", default=f"{REPO}/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin")
ap.add_argument("--bhdr-bake", default=f"{REPO}/zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin")
ap.add_argument("--sdr-features", default=f"{PROBE}/upiq_sdr_features_372_u8shell.parquet")
ap.add_argument("--pl-features", default=f"{PROBE}/upiq_sdr_features_372_pulinear203.parquet")
ap.add_argument("--subjective", default="/mnt/v/datasets/upiq/upiq_subjective_scores.csv")
ap.add_argument("--out", default=None)
a = ap.parse_args()

# reuse score_bake from the cross-domain instrument without running its main
import sys
sys.argv = ["xdi"]
xdi = importlib.util.module_from_spec(spec)
_real_main = None
src = open(os.path.join(os.path.dirname(__file__), "upiq_crossdomain_instrument.py")).read()
# execute only up to (and including) score_bake's definition
head = src.split("# ---- SDR half")[0]
ns = {}
exec(compile(head, "xdi_head", "exec"), ns)
score_bake = ns["score_bake"]

rows = list(csv.DictReader(open(a.subjective)))

t_u8 = pq.read_table(a.sdr_features)
t_pl = pq.read_table(a.pl_features)
q_u8 = np.array([int(float(x)) for x in t_u8["q"].to_pylist()])
q_pl = np.array([int(float(x)) for x in t_pl["q"].to_pylist()])
common, i_u8, i_pl = np.intersect1d(q_u8, q_pl, return_indices=True)
assert len(common) == len(q_u8) == len(q_pl), (len(common), len(q_u8), len(q_pl))

dial_b = score_bake(a.b_bake, t_u8)[i_u8]
dial_h = score_bake(a.bhdr_bake, t_pl)[i_pl]
dset = np.array([rows[i]["dataset"] for i in common])
files = [rows[i]["test_file"] for i in common]

d = dial_h - dial_b
L = ["# G-A / R1 sub-domain identity report", ""]
L.append(f"- B: `{os.path.basename(a.b_bake)}` (native SDR path)")
L.append(f"- BHdr: `{os.path.basename(a.bhdr_bake)}` (203-nit PQ re-encode → PU-linear path)")
L.append(f"- n = {len(d)} SDR pairs (UPIQ images; JOD unused — no holdout burn)")
L.append("")
def block(name, m):
    L.append(f"## {name} (n={int(m.sum())})")
    L.append(f"- Δ = BHdr − B: mean {d[m].mean():+.2f}, median {np.median(d[m]):+.2f}")
    L.append(f"- |Δ|: p50 {np.percentile(np.abs(d[m]),50):.2f}, p95 {np.percentile(np.abs(d[m]),95):.2f}, max {np.abs(d[m]).max():.2f}")
    L.append(f"- rank agreement SROCC(B, BHdr): {spearmanr(dial_b[m], dial_h[m]).statistic:.4f}")
    p95 = np.percentile(np.abs(d[m]), 95)
    sr = spearmanr(dial_b[m], dial_h[m]).statistic
    L.append(f"- **GATE (p95 ≤ 2, SROCC ≥ 0.99): {'PASS' if p95 <= 2 and sr >= 0.99 else 'FAIL'}**")
    L.append("")
block("Aggregate", np.ones(len(d), bool))
for ds in ("live", "tid2013"):
    block(ds, dset == ds)
worst = np.argsort(-np.abs(d))[:10]
L.append("## Worst 10 |Δ|")
for i in worst:
    L.append(f"- {files[i]}: B {dial_b[i]:.1f} vs BHdr {dial_h[i]:.1f} (Δ {d[i]:+.1f})")
report = "\n".join(L)
print(report)
if a.out:
    open(a.out, "w").write(report + "\n")
    print(f"\nwrote {a.out}")
