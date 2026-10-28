#!/usr/bin/env python3
"""Cross-domain (SDR↔HDR) dial-alignment instrument on the FULL UPIQ set.

PLAN_HDR_SDR_ALIGNMENT step 1 (R2 baseline): UPIQ puts 3,779 SDR pairs
(tid2013 + live) and 380 HDR pairs (narwaria + korshunov) on ONE human
JOD-aligned scale. Alignment means one monotone dial↔JOD mapping fits both
domains: after fitting a pooled isotonic dial = g(JOD), the per-domain mean
residual is the SEAM in dial points.

Feeds (regime-honest):
  SDR: Profile-B bake over the plain-SDR u8 extraction
       (upiq_sdr_features_372_u8shell.parquet; join key q = subjective-csv
       row index).
  HDR: BHdr bake over the PU-linear extraction
       (upiq_features_372_pulinear.parquet; row order = upiq_cid_jod.csv).

  usage: upiq_crossdomain_instrument.py [--b-bake B.bin] [--bhdr-bake BHDR.bin]
                                        [--out report.md]

Caveat printed in the report: the tid2013 leg overlaps training corpora
(integrity-guard grade); `live` is the clean SDR leg.
"""
import argparse
import csv
import os
import subprocess
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPO = os.path.expanduser("~/work/zen/zensimx")
PROBE = "/mnt/v/output/zensim-multicodec-probe"

ap = argparse.ArgumentParser()
ap.add_argument("--b-bake", default=f"{REPO}/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin")
ap.add_argument("--bhdr-bake", default=f"{REPO}/zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin")
ap.add_argument("--sdr-features", default=f"{PROBE}/upiq_sdr_features_372_u8shell.parquet")
ap.add_argument("--hdr-features", default=f"{PROBE}/upiq_features_372_pulinear.parquet")
ap.add_argument("--subjective", default="/mnt/v/datasets/upiq/upiq_subjective_scores.csv")
ap.add_argument("--hdr-jod", default="/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv")
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.subjective)))


def score_bake(bake_path: str, feats: pa.Table) -> np.ndarray:
    """Forward a bake over a features table via bake_verdict's pred-dump
    (same route as upiq_panel.py — one 'ladder' per row, invertible by q)."""
    n = feats.num_rows
    fcols = sorted((c for c in feats.schema.names if c.startswith("feat_")),
                   key=lambda c: int(c.split("_")[1]))
    assert len(fcols) == 372, len(fcols)
    data = {
        "ref_basename": pa.array([f"x{i}" for i in range(n)]),
        "human_score": pa.array([0.5] * n),
        "image_path": pa.array([f"x{i}" for i in range(n)]),
        "image_id": pa.array(["x"] * n),
        "codec": pa.array(["x"] * n),
        "q": pa.array([float(i) for i in range(n)]),
        "knob_tuple_json": pa.array(["{}"] * n),
    }
    for i, c in enumerate(fcols):
        data[f"f{i}"] = feats[c]
    with tempfile.TemporaryDirectory() as td:
        gp = os.path.join(td, "grid.parquet")
        pq.write_table(pa.table(data), gp, compression="zstd")
        pred = os.path.join(td, "pred.tsv")
        env = dict(os.environ, ZENSIM_DIAL_GRID=gp, ZENSIM_DIAL_PRED_OUT=pred)
        subprocess.run([f"{REPO}/target/release/bake_verdict", "--bake", bake_path,
                        "--corpora", "aic3", "--output", os.devnull],
                       env=env, capture_output=True, check=True)
        lines = open(pred).read().splitlines()
        cols = lines[0].split("\t")
        qi, pi = cols.index("q"), cols.index("pred")
        out = np.full(n, np.nan)
        for l in lines[1:]:
            r = l.split("\t")
            out[int(float(r[qi]))] = float(r[pi])
    assert not np.isnan(out).any()
    return out


# ---- SDR half: features joined to csv rows by q = csv row index ----
sdr_t = pq.read_table(a.sdr_features)
sdr_idx = np.array([int(float(x)) for x in sdr_t["q"].to_pylist()])
sdr_jod = np.array([float(rows[i]["JOD"]) for i in sdr_idx])
sdr_dset = np.array([rows[i]["dataset"] for i in sdr_idx])
assert all(rows[i]["is_hdr"] in ("0", "false", "False") for i in sdr_idx)
sdr_dial = score_bake(a.b_bake, sdr_t)

# ---- HDR half: pulinear parquet row order == hdr-jod csv order ----
hdr_t = pq.read_table(a.hdr_features)
hlines = [l for l in open(a.hdr_jod).read().splitlines() if l.strip()]
hdr_jod = np.array([float(l.split(",")[1]) for l in hlines])
hdr_ids = [l.split(",")[0] for l in hlines]
assert len(hdr_jod) == hdr_t.num_rows
# sanity: ids/JODs must agree with the subjective csv (same source data)
sub_hdr = {r["condition_id"]: float(r["JOD"]) for r in rows if r["is_hdr"] not in ("0", "false", "False")}
for cid, j in list(zip(hdr_ids, hdr_jod))[:5] + list(zip(hdr_ids, hdr_jod))[-5:]:
    assert cid in sub_hdr and abs(sub_hdr[cid] - j) < 1e-6, (cid, j, sub_hdr.get(cid))
hdr_dset = np.array(["narwaria" if i.startswith("n-") else "korshunov" for i in hdr_ids])
hdr_dial = score_bake(a.bhdr_bake, hdr_t)


def iso_fit(x, y):
    """Increasing isotonic y = g(x) (pool-adjacent-violators), returns g(x_eval)."""
    order = np.argsort(x)
    ys = y[order].astype(float)
    w = np.ones_like(ys)
    # PAVA
    vals, wts, idx = [], [], []
    for v, ww in zip(ys, w):
        vals.append(v); wts.append(ww); idx.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            w2 = wts[-2] + wts[-1]; i2 = idx[-2] + idx[-1]
            vals = vals[:-2] + [v2]; wts = wts[:-2] + [w2]; idx = idx[:-2] + [i2]
    fitted = np.concatenate([np.full(i, v) for v, i in zip(vals, idx)])
    out = np.empty_like(fitted)
    out[order] = fitted
    return out


# pooled isotonic dial = g(JOD)
jod_all = np.concatenate([sdr_jod, hdr_jod])
dial_all = np.concatenate([sdr_dial, hdr_dial])
dom_all = np.array(["sdr"] * len(sdr_jod) + ["hdr"] * len(hdr_jod))
g = iso_fit(jod_all, dial_all)
resid = dial_all - g

L = []
L.append("# UPIQ cross-domain dial-alignment baseline")
L.append("")
L.append(f"- B bake: `{os.path.basename(a.b_bake)}` | BHdr bake: `{os.path.basename(a.bhdr_bake)}`")
L.append(f"- SDR: n={len(sdr_jod)} (u8-shell extraction) | HDR: n={len(hdr_jod)} (PU-linear extraction)")
L.append("")
L.append("## Rank (|SROCC| dial vs JOD)")
for name, d, j in [("SDR pooled", sdr_dial, sdr_jod), ("HDR pooled", hdr_dial, hdr_jod)]:
    L.append(f"- {name}: {abs(spearmanr(d, j).statistic):.4f}")
for ds in ("live", "tid2013"):
    m = sdr_dset == ds
    L.append(f"- SDR/{ds}: {abs(spearmanr(sdr_dial[m], sdr_jod[m]).statistic):.4f} (n={m.sum()})"
             + ("  [CLEAN leg]" if ds == "live" else "  [training-overlap: guard-grade]"))
for ds in ("narwaria", "korshunov"):
    m = hdr_dset == ds
    L.append(f"- HDR/{ds}: {abs(spearmanr(hdr_dial[m], hdr_jod[m]).statistic):.4f} (n={m.sum()})")
L.append("")
L.append("## Seam (residual vs pooled isotonic dial=g(JOD), dial points)")
for dom in ("sdr", "hdr"):
    m = dom_all == dom
    L.append(f"- {dom.upper()}: mean {resid[m].mean():+.2f}, median {np.median(resid[m]):+.2f}, "
             f"p95|.| {np.percentile(np.abs(resid[m]), 95):.2f}")
L.append(f"- **SEAM (mean_HDR − mean_SDR): {resid[dom_all=='hdr'].mean() - resid[dom_all=='sdr'].mean():+.2f} dial points**")
L.append("")
L.append("## Equal-JOD band means (dial per domain; alignment = rows match)")
L.append("| JOD band | SDR mean dial (n) | HDR mean dial (n) | Δ (HDR−SDR) |")
L.append("|---|---|---|---|")
edges = [(-100, -8), (-8, -6), (-6, -4), (-4, -3), (-3, -2), (-2, -1), (-1, -0.5), (-0.5, 0.25)]
for lo, hi in edges:
    ms = (dom_all == "sdr") & (jod_all >= lo) & (jod_all < hi)
    mh = (dom_all == "hdr") & (jod_all >= lo) & (jod_all < hi)
    if ms.sum() < 5 or mh.sum() < 5:
        continue
    ds_, dh_ = dial_all[ms].mean(), dial_all[mh].mean()
    L.append(f"| [{lo},{hi}) | {ds_:.1f} ({ms.sum()}) | {dh_:.1f} ({mh.sum()}) | {dh_-ds_:+.1f} |")

report = "\n".join(L)
print(report)
if a.out:
    open(a.out, "w").write(report + "\n")
    print(f"\nwrote {a.out}")
