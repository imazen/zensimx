#!/usr/bin/env python3
"""False-positive rate of a corruption head on HONEST codec output, per codec.

The corruption gate asks "does a broken decode rank below an honest q20?". It does
NOT ask the question that decides whether a head can sit in a closed loop: **how
often does the head fire on output a codec legitimately produced?** A head that
zeroes a quarter of honest AVIF is worse than no head, and the gate is blind to it
because the gate's only honest rows are two anchors from one reference.

MEASURED 2026-09-05 on the 2026-07-24 head, whose record reports 0.34%
broad-honest FP: on the ladder instrument (9,593 honest current-era imazen codec
cells, floor-dense) it fires on **14.33%** at T=0.9 — jxl 26.41%, avif-rav1e
24.44%, and on jxl up to q60, not merely at the floor. Its broad-honest negatives
were five legacy 720 corpora with no floor-dense codec output. Same-pixel two-era
control (1,344 honest anchors, stored vs postC) reads 0.00% at both, so that gap
is COVERAGE, not extraction era.

Scoring goes through `predict_features_with_bake` — the owner for "forward a bake
over feature rows", which applies the output calibration spline exactly as
`bake_verdict`'s gate does. Nothing here re-implements a metric; the only Python
arithmetic is a group-by mean.

Usage:
  corruption_head_honest_fp.py --bake head.bin --grid dial_grid_372col_ladder.parquet \
      [--codec-col codec] [--q-col q] [--split-col image_id] [--split-ids a,b,c] \
      [--thresholds 0.9,0.95]
"""
import argparse, os, struct, subprocess, sys, tempfile
import numpy as np, pyarrow.parquet as pq

FWD = os.path.expanduser(
    "~/work/zen/zensimx/target/release/predict_features_with_bake")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--fwd", default=FWD)
    ap.add_argument("--codec-col", default="codec")
    ap.add_argument("--q-col", default="q")
    ap.add_argument("--split-col", default=None,
                    help="restrict to rows whose value is in --split-ids (e.g. the "
                         "held-out ladder images, when the head TRAINED on the rest)")
    ap.add_argument("--split-ids", default=None)
    ap.add_argument("--thresholds", default="0.9,0.95",
                    help="detection thresholds T; a head baked to emit 100*(1-P) "
                         "fires when score < 100*(1-T)")
    ap.add_argument("--out-tsv", default=None)
    a = ap.parse_args()

    names = pq.ParquetFile(a.grid).schema_arrow.names
    nfeat = 1 + max(int(c[1:]) for c in names
                    if c.startswith("f") and c[1:].isdigit())
    cols = [f"f{i}" for i in range(nfeat)]
    extra = [c for c in (a.codec_col, a.q_col, a.split_col) if c and c in names]
    t = pq.read_table(a.grid, columns=cols + extra)
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False)
                         for c in cols]).astype(np.float32)
    codec = (np.array(t.column(a.codec_col).to_pylist()) if a.codec_col in extra
             else np.array(["all"] * len(X)))
    q = (np.array(t.column(a.q_col).to_pylist()) if a.q_col in extra
         else np.zeros(len(X)))
    keep = np.ones(len(X), bool)
    if a.split_col and a.split_ids:
        want = set(a.split_ids.split(","))
        keep = np.array([str(v) in want
                         for v in t.column(a.split_col).to_pylist()])
        print(f"restricted to {keep.sum()} of {len(X)} rows "
              f"({a.split_col} in {len(want)} ids)")
    X, codec, q = X[keep], codec[keep], q[keep]

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False,
                                     dir=os.path.expanduser("~/tmp")) as f:
        f.write(struct.pack("<II", nfeat, len(X)) + X.tobytes(order="C"))
        blob = f.name
    r = subprocess.run([a.fwd, "--bake", a.bake, "--bake-post", "raw",
                        "--features-file", blob], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{a.fwd} failed rc={r.returncode}: {r.stderr.strip()[:400]}")
    s = np.array([float(x) for x in r.stdout.split()])
    assert len(s) == len(X), (len(s), len(X))

    Ts = [float(x) for x in a.thresholds.split(",")]
    hdr = ["codec", "n"] + [f"fp_T{T}" for T in Ts] + ["max_q_flagged_T" + str(Ts[0])]
    rows = []
    for cc in list(sorted(set(codec))) + ["ALL"]:
        m = np.ones(len(s), bool) if cc == "ALL" else (codec == cc)
        fl = s[m] < 100.0 * (1.0 - Ts[0])
        rows.append([cc, int(m.sum())]
                    + [f"{100*(s[m] < 100.0*(1.0-T)).mean():.2f}" for T in Ts]
                    + [f"{q[m][fl].max():.1f}" if fl.any() else "-"])
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    for r in [hdr] + rows:
        print("  ".join(str(v).rjust(w[i]) for i, v in enumerate(r)))
    if a.out_tsv:
        with open(a.out_tsv, "w") as f:
            f.write("\t".join(hdr) + "\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")
        print(f"wrote {a.out_tsv}")


if __name__ == "__main__":
    main()
