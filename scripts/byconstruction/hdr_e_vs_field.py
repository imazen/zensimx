#!/usr/bin/env python3
"""E vs stored per-pair reference metrics on UPIQ HDR (380) and HDRTEACH VAL (3,900).

Reads predictions written by hdr_panel_upiq_all.py and the stored baseline panels; no fit, no selection.
Usage: hdr_e_vs_field.py <dir with pred_upiq_all_{E,ECodec}.tsv>
"""
import csv, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr

U = "/mnt/v/output/zenmetrics/upiq-pu/"
pred_dir = sys.argv[1]
keys = pq.read_table("/var/tmp/byc-hdr-20261029/banks/upiq-r2-bank/keys.parquet").to_pandas()
jod = [(r[0], float(r[1])) for r in csv.reader(open(U + "upiq_cid_jod.csv"))]
order = {c: i for i, (c, _) in enumerate(jod)}
pos = np.array([order[c] for c in keys.condition_id])  # bank row -> baseline row
y = np.array([jod[p][1] for p in pos])
M = {n: pd.read_csv(f"{pred_dir}/pred_upiq_all_{n}.tsv", sep="\t").iloc[:, 1].values for n in ("E", "ECodec")}
for n, f in [("CVVDP 10k nits", "panel_cvvdp_gpu_10k"), ("CVVDP default", "panel_cvvdp_gpu"), ("butteraugli p3", "panel_butter_pnorm3"),
             ("butteraugli max", "panel_butter_max"), ("HDR-VDP-2", "panel_base_HDRVDP2_2"), ("PU-PieAPP", "panel_base_PU_PieApp")]:
    t = pd.read_csv(U + f + ".tsv", sep="\t")
    assert np.allclose(t.target.values[pos], y, atol=1e-6), n  # positional join verified against JOD
    M[n] = t.predicted.values[pos]
ds = keys.dataset.values
ref = (keys.dataset + ":" + keys.content.astype(str)).values


def wref(x):
    return np.nanmean([spearmanr(x[ref == g], y[ref == g])[0] for g in np.unique(ref)])


print("| metric | pooled | Narwaria | Korshunov | within-reference |\n|---|---:|---:|---:|---:|")
for n, x in M.items():
    s = lambda m: spearmanr(x[m], y[m])[0]
    print(f"| {n} | {s(np.ones(len(y), bool)):.3f} | {s(ds == 'narwaria'):.3f} | {s(ds == 'korshunov'):.3f} | {wref(x):.3f} |")

refs = np.unique(ref)
rng = np.random.default_rng(0)
print("\nPaired bootstrap over the 30 references (2,000 draws), E minus metric:\n\n| vs | population | mean | 95% CI |\n|---|---|---:|---|")
for n in ("CVVDP 10k nits", "butteraugli p3", "HDR-VDP-2"):
    for kind in ("pooled", "narwaria", "korshunov"):
        d = []
        for _ in range(2000):
            idx = np.concatenate([np.where(ref == r)[0] for r in rng.choice(refs, len(refs))])
            m = np.ones(len(idx), bool) if kind == "pooled" else ds[idx] == kind
            if m.sum() < 5:
                continue
            d.append(spearmanr(M["E"][idx][m], y[idx][m])[0] - spearmanr(M[n][idx][m], y[idx][m])[0])
        print(f"| {n} | {kind} | {np.mean(d):+.3f} | [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |")

s = pq.read_table("/mnt/v/output/zensim/hdrteach-2026-10-04/hdrteach_val.parquet", columns=["ref_sha256", "hdrvdp3_q_jod", "cvvdp_jod"]).to_pandas()
yt, g = s.hdrvdp3_q_jod.values, s.ref_sha256.values
print("\nHDRTEACH VAL (3,900 rows, 300 references; target HDR-VDP-3 Q_JOD):\n\n| metric | pooled | within-reference | inversions / pairs |\n|---|---:|---:|---:|")
for n, x in (("E", pd.read_csv("/var/tmp/byc-hdr-20261029/panel-all/pred_hdrteach_val_E.tsv", sep="\t").pred.values), ("CVVDP", s.cvvdp_jod.values)):
    inv = pairs = 0
    for r in np.unique(g):
        i = np.where(g == r)[0]
        for a in range(len(i)):
            for b in range(a + 1, len(i)):
                if yt[i[a]] != yt[i[b]]:
                    pairs += 1
                    inv += (x[i[a]] - x[i[b]]) * (yt[i[a]] - yt[i[b]]) < 0
    w = np.nanmean([spearmanr(x[g == r], yt[g == r])[0] for r in np.unique(g)])
    print(f"| {n} | {spearmanr(x, yt)[0]:.3f} | {w:.4f} | {inv} / {pairs} |")
