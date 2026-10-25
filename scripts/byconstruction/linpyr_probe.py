"""Label-free probe: do coarse scales averaged in linear light change damage by distortion type?

zensim converts each image to positive XYB at full resolution and builds scales 1-3 by 2x2 averages of XYB values.
HaarXYB instead averages linear RGB and converts each level. For every TRAIN pair (KADID train split, TID2013,
KonFiG train split; keys only, no labels) this builds both pyramids and reports, per dataset, distortion type, scale
and channel, the median log ratio of two damage measures (mean squared XYB difference, and the mean squared difference
of 7x7 box-detail, i.e. pixel minus local mean), linear-light over XYB-averaged. A ratio that moves with the type would
change how a model places types on one scale; a ratio that is the same for every type would not.
Exploration record: benchmarks/byconstruction_2026-10-22.md.

  python3 linpyr_probe.py --out DIR [--sets kadid_train,tid2013,konfig_train] [--workers 12]
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BANK = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/bank")
M = np.array([[0.30, 1 - 0.078 - 0.30, 0.078],
              [0.23, 1 - 0.078 - 0.23, 0.078],
              [0.24342269, 0.20476745, 1 - 0.24342269 - 0.20476745]])
B0 = 0.0037930734
AB = -np.cbrt(B0)


def linear(path):
    v = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def xyb(lin):
    m = np.maximum(lin @ M.T + B0, 0.0)
    t = np.cbrt(m)
    c0, c1 = t[..., 0] + AB, t[..., 1] + AB
    x, y = 0.5 * (c0 - c1), 0.5 * (c0 + c1)
    return np.stack([14.0 * x + 0.42, y + 0.01, (t[..., 2] - y) + 0.55], axis=-1)


def down(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def box7(a):
    p = np.pad(a, ((3, 3), (3, 3), (0, 0)), mode="reflect")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0), (0, 0)))
    s = c[7:, 7:] - c[:-7, 7:] - c[7:, :-7] + c[:-7, :-7]
    return s / 49.0


def damage(r, d):
    mse = ((r - d) ** 2).mean(axis=(0, 1))
    det = ((r - box7(r) - (d - box7(d))) ** 2).mean(axis=(0, 1))
    return mse, det


def one(args):
    ref_path, dist_path = args
    lr, ld = linear(ref_path), linear(dist_path)
    if lr.shape != ld.shape:
        return None
    xa_r, xa_d = xyb(lr), xyb(ld)
    out = []
    lin_r, lin_d = lr, ld
    for _s in (1, 2, 3):
        xa_r, xa_d = down(xa_r), down(xa_d)
        lin_r, lin_d = down(lin_r), down(lin_d)
        xb_r, xb_d = xyb(lin_r), xyb(lin_d)
        a_mse, a_det = damage(xa_r, xa_d)
        b_mse, b_det = damage(xb_r, xb_d)
        out.append([a_mse.tolist(), b_mse.tolist(), a_det.tolist(), b_det.tolist()])
    return out


def kind(s, dist):
    name = Path(dist).name
    if s.startswith("kadid"):
        return f"kadid:{int(re.match(r'I\d+_(\d+)_', name, re.I).group(1)):02d}"
    if s == "tid2013":
        return f"tid:{int(re.match(r'i\d+_(\d+)_', name, re.I).group(1)):02d}"
    return f"konfig:{Path(dist).parent.name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="kadid_train,tid2013,konfig_train")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in a.sets.split(","):
        keys = pq.read_table(BANK / s / "keys.parquet", columns=["ref_path", "dist_path", "pixels_identical"]).to_pydict()
        pairs = [(r, d) for r, d, ident in zip(keys["ref_path"], keys["dist_path"], keys["pixels_identical"]) if not ident]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, pairs, chunksize=8))
        for (r, d), v in zip(pairs, res):
            if v is not None:
                rows.append({"set": s, "kind": kind(s, d), "ref": r, "dist": d, "scales": v})
        print(f"{s}: {len(pairs)} pairs", flush=True)
    (out / "linpyr_rows.json").write_text(json.dumps(rows))
    eps = 1e-12
    summary = {}
    for measure, (ia, ib) in {"mse": (0, 1), "detail": (2, 3)}.items():
        by = {}
        for row in rows:
            for s in range(3):
                a_, b_ = np.array(row["scales"][s][ia]), np.array(row["scales"][s][ib])
                lr = np.log((b_ + eps) / (a_ + eps))
                by.setdefault(row["kind"], {}).setdefault(s + 1, []).append(lr)
        summary[measure] = {k: {s: np.median(np.array(v), axis=0).round(4).tolist() for s, v in d.items()}
                            for k, d in sorted(by.items())}
    (out / "linpyr_summary.json").write_text(json.dumps(summary, indent=1))
    for measure in ("mse", "detail"):
        print(f"== {measure}: median log(linear-light / XYB-averaged), scale 1..3 x channel X,Y,B")
        for k, d in summary[measure].items():
            cells = "  ".join(",".join(f"{x:+.2f}" for x in d[s]) for s in (1, 2, 3))
            print(f"  {k:28s} {cells}")


if __name__ == "__main__":
    main()
