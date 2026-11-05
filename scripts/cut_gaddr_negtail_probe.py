#!/usr/bin/env python3
"""Cut a G-ADDR negative-tail probe by the registered rule (THE owner for this cut).

RULE (`benchmarks/dial_addressability_gate_2026-09-04.md` sec.4): over the rows whose
reference metric is NEGATIVE, form 20 equal-count quantile bins on that metric and take
the LOWEST 100 ROW INDICES in each bin -> 2,000 rows. Deterministic, no RNG.

Why this file exists: the 372-wide `negtail_probe_372_2026-09-04.parquet` was cut ad hoc
and had no committed builder, so the rule lived only in prose -- and a probe cut at a
different feature width (which every 944-regime bake needs, since `bake_verdict` scores a
probe only when its column count equals the bake's caller width) had to re-derive it.

CONTROL, run 2026-09-04 before this was used for anything: re-cutting the ORIGINAL source
(`canonical-2026-07-15/train/kadis_negrich.parquet`, 266,111 rows / 237,675 negative) with
this code reproduces the stored 372 probe's `ssim2_gpu` column EXACTLY -- 2,000 of 2,000
rows, max abs diff 0.0, span -770.619744 .. -0.331242. So a probe this file cuts at
another width is the same instrument construction, not a lookalike.

    cut_gaddr_negtail_probe.py <src.parquet> <negative-truth column> <out.parquet> [truth-name]

The output carries `entry` (negtail_0..N), the truth column, and every fN/feat_N column of
the source at its native width. Written zstd -- NEVER snappy, which the Rust parquet
reader cannot decompress.
"""
import sys, numpy as np, pyarrow as pa, pyarrow.parquet as pq

def cut(src, metric_col, n_bins=20, per_bin=100):
    t = pq.read_table(src, columns=[metric_col])
    s = np.asarray(t.column(metric_col).to_pylist(), dtype=np.float64)
    neg_idx = np.nonzero(s < 0.0)[0]              # original row indices, ascending
    vals = s[neg_idx]
    order = np.argsort(vals, kind="stable")       # by metric, ascending
    parts = np.array_split(order, n_bins)         # equal-count quantile bins
    picked = []
    for p in parts:
        rows = np.sort(neg_idx[p])                # lowest ROW INDICES within the bin
        picked.append(rows[:per_bin])
    return np.sort(np.concatenate(picked)), len(neg_idx), len(s)

if __name__ == "__main__":
    src, metric, out = sys.argv[1], sys.argv[2], sys.argv[3]
    truth_name = sys.argv[4] if len(sys.argv) > 4 else "ssim2_gpu"
    rows, n_neg, n_all = cut(src, metric)
    print(f"source rows {n_all}  negative {n_neg}  picked {len(rows)}")
    sch = pq.ParquetFile(src).schema_arrow
    feats = []
    i = 0
    pref = "feat_" if "feat_0" in sch.names else "f"
    while f"{pref}{i}" in sch.names:
        feats.append(f"{pref}{i}"); i += 1
    t = pq.read_table(src, columns=[metric] + feats)
    t = t.take(pa.array(rows))
    cols = {"entry": pa.array([f"negtail_{k}" for k in range(len(rows))]),
            truth_name: t.column(metric).cast(pa.float64())}
    for f in feats:
        cols[f if pref == "f" else f] = t.column(f)
    tbl = pa.table(cols)
    if out != "-":
        pq.write_table(tbl, out, compression="zstd")
        print("wrote", out, tbl.num_rows, "x", tbl.num_columns)
    else:
        print("dry run; truth min", min(cols[truth_name].to_pylist()),
              "max", max(cols[truth_name].to_pylist()))
