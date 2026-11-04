"""E36 candidates against the field on HDR (record: R10 amendment 6, 2026-11-04). A generalisation of
`scripts/byconstruction/hdr_e_vs_field.py` to several candidate names and to metrics scored now on the stored
PU-encoded UPIQ renderings.

Reads the predictions `hdr_panel_upiq_all.py` wrote (`pred_upiq_all_<name>.tsv`, bank row order), the stored per-pair
UPIQ panels (`/mnt/v/output/zenmetrics/upiq-pu/panel_*.tsv`, JOD order; the join is asserted against the stored JOD),
an optional `jnd-ladder-score` output on the PU PNG pairs (`id` = condition id; SSIMULACRA2, butteraugli, FSIMc, HaarPSI),
and `hdr_panel.py`'s `pred_hdrteach_val_<name>.tsv` and `panel.json` for the other populations. No fit, no selection.

    hdr_candidates_vs_field.py --upiq-dir DIR --models F3_s0,mix_s0,... [--pu-scores SCORES.tsv] [--panel2-dir DIR2]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

U = "/mnt/v/output/zenmetrics/upiq-pu/"
STORED = [("CVVDP 10k nits", "panel_cvvdp_gpu_10k"), ("CVVDP default", "panel_cvvdp_gpu"), ("butteraugli p3 (stored, 10k)", "panel_butter_pnorm3_10k"),
          ("butteraugli p3 (stored)", "panel_butter_pnorm3"), ("butteraugli max (stored)", "panel_butter_max"),
          ("HDR-VDP-2", "panel_base_HDRVDP2_2"), ("PU-PieAPP", "panel_base_PU_PieApp"), ("PU-SSIM", "panel_base_PU_SSIM"),
          ("PU-FSIM", "panel_base_PU_FSIM")]
PU_NEW = [("SSIMULACRA2 (PU)", "ssim2", 1), ("butteraugli p3 (PU)", "ba_p3", -1), ("butteraugli max (PU)", "ba_max", -1),
          ("FSIMc (PU)", "fsimc", 1), ("FSIM (PU)", "fsim_y", 1), ("HaarPSI (PU)", "haarpsi", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upiq-dir", required=True)
    ap.add_argument("--models", required=True, help="comma-separated names; the first pair F3_sN,mix_sN per seed is compared")
    ap.add_argument("--pu-scores")
    ap.add_argument("--panel2-dir")
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    names = a.models.split(",")
    keys = pq.read_table("/var/tmp/byc-hdr-20261029/banks/upiq-r2-bank/keys.parquet").to_pandas()
    jod = [(r[0], float(r[1])) for r in csv.reader(open(U + "upiq_cid_jod.csv"))]
    order = {c: i for i, (c, _) in enumerate(jod)}
    pos = np.array([order[c] for c in keys.condition_id])
    y = np.array([jod[p][1] for p in pos])
    M = {}
    for n in names:
        M[n] = pd.read_csv(f"{a.upiq_dir}/pred_upiq_all_{n}.tsv", sep="\t").iloc[:, 1].values
    for n, f in STORED:
        t = pd.read_csv(U + f + ".tsv", sep="\t")
        assert np.allclose(t.target.values[pos], y, atol=1e-6), n
        M[n] = t.predicted.values[pos]
    if a.pu_scores:
        rows = {r["id"]: r for r in csv.DictReader(open(a.pu_scores), delimiter="\t")}
        for n, col, sign in PU_NEW:
            M[n] = np.array([sign * float(rows[c][col]) for c in keys.condition_id])
    ds = keys.dataset.values
    ref = (keys.dataset + ":" + keys.content.astype(str)).values

    def wref(x):
        return np.nanmean([spearmanr(x[ref == g], y[ref == g])[0] for g in np.unique(ref)])

    print("## UPIQ HDR (380 pairs, 30 references; target JOD)\n\n| score | pooled | Narwaria | Korshunov | within-reference |\n|---|---:|---:|---:|---:|")
    for n, x in M.items():
        s = lambda m: spearmanr(x[m], y[m])[0]  # noqa: E731
        print(f"| {n} | {s(np.ones(len(y), bool)):.3f} | {s(ds == 'narwaria'):.3f} | {s(ds == 'korshunov'):.3f} | {wref(x):.3f} |")
    refs = np.unique(ref)
    rng = np.random.default_rng(0)
    draws = [np.concatenate([np.where(ref == r)[0] for r in rng.choice(refs, len(refs))]) for _ in range(a.draws)]

    def paired(x1, x2, kind):
        d = []
        for idx in draws:
            m = np.ones(len(idx), bool) if kind == "pooled" else ds[idx] == kind
            if m.sum() < 5:
                continue
            d.append(spearmanr(x1[idx][m], y[idx][m])[0] - spearmanr(x2[idx][m], y[idx][m])[0])
        return np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5)

    seeds = sorted({n.rsplit("_s", 1)[1] for n in names if "_s" in n})
    print(f"\nPaired bootstrap over the 30 references ({a.draws} draws), mix minus F3 at the same seed:\n\n| seed | population | mean | 95% CI |\n|---|---|---:|---|")
    for s in seeds:
        if f"mix_s{s}" in M and f"F3_s{s}" in M:
            for kind in ("pooled", "narwaria", "korshunov"):
                m, lo, hi = paired(M[f"mix_s{s}"], M[f"F3_s{s}"], kind)
                print(f"| {s} | {kind} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    cand = [n for n in (f"F3_s{seeds[0]}", f"mix_s{seeds[0]}") if n in M] if seeds else []
    field = [n for n in ("CVVDP 10k nits", "butteraugli p3 (stored)", "SSIMULACRA2 (PU)", "butteraugli p3 (PU)", "HDR-VDP-2", "FSIMc (PU)", "HaarPSI (PU)") if n in M]
    print(f"\nPaired bootstrap, candidate (seed {seeds[0] if seeds else '?'}) minus metric:\n\n| candidate | vs | pooled | Narwaria | Korshunov |\n|---|---|---|---|---|")
    for c in cand:
        for n in field:
            cells = []
            for kind in ("pooled", "narwaria", "korshunov"):
                m, lo, hi = paired(M[c], M[n], kind)
                cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"| {c} | {n} | " + " | ".join(cells) + " |")
    if a.panel2_dir:
        p2 = Path(a.panel2_dir)
        s = pq.read_table("/mnt/v/output/zensim/hdrteach-2026-10-04/hdrteach_val.parquet", columns=["ref_sha256", "hdrvdp3_q_jod", "cvvdp_jod"]).to_pandas()
        yt, g = s.hdrvdp3_q_jod.values, s.ref_sha256.values
        print("\n## HDRTEACH VAL (3,900 rows, 300 references; target HDR-VDP-3 Q_JOD)\n\n| score | pooled | within-reference | inversions / pairs |\n|---|---:|---:|---:|")
        series = [(n, pd.read_csv(p2 / f"pred_hdrteach_val_{n}.tsv", sep="\t").iloc[:, 1].values) for n in names if (p2 / f"pred_hdrteach_val_{n}.tsv").exists()]
        series.append(("CVVDP", s.cvvdp_jod.values))
        groups = {r: np.where(g == r)[0] for r in np.unique(g)}
        for n, x in series:
            inv = pairs = 0
            for i in groups.values():
                for aa in range(len(i)):
                    for bb in range(aa + 1, len(i)):
                        if yt[i[aa]] != yt[i[bb]]:
                            pairs += 1
                            inv += (x[i[aa]] - x[i[bb]]) * (yt[i[aa]] - yt[i[bb]]) < 0
            w = np.nanmean([spearmanr(x[i], yt[i])[0] for i in groups.values()])
            print(f"| {n} | {spearmanr(x, yt)[0]:.3f} | {w:.4f} | {inv} / {pairs} |")
        pj = p2 / "panel.json"
        if pj.exists():
            res = json.loads(pj.read_text())
            print("\n## Other populations (hdr_panel.py: pooled SROCC / within-group Spearman mean; delta within group vs the first model)\n")
            for pop, v in res.items():
                print(f"- {pop}: " + "; ".join(f"{n} {m.get('pooled_srocc')!s:.6} / {m.get('within_group_spearman_mean')!s:.6}"
                                                 + (f" (Δ {m['delta_within_group_vs_base']})" if 'delta_within_group_vs_base' in m else "") for n, m in v.items()))


if __name__ == "__main__":
    main()
