"""Label-free probe: does damage split by orientation (Haar horizontal, vertical, diagonal bands) differ by distortion type?

Every zensim edge and texture feature is isotropic. HaarPSI and HaarXYB measure horizontal and vertical (HaarXYB also
diagonal) Haar bands at every level. For every TRAIN pair (KADID train split, TID2013, KonFiG train split; keys only,
no labels) this builds zensim's pyramid (positive XYB at full resolution, 2x2 averages), takes the three Haar detail
bands of each scale's 2x2 blocks (scales 0-3) and reports, per dataset, distortion type and scale, on Y: the share of
band-difference energy in the diagonal band, and the horizontal/vertical asymmetry |E_H - E_V| / (E_H + E_V) of the
difference energy. Types whose damage is orientation-selective would separate on these; isotropic damage gives a
diagonal share near the reference's own and an asymmetry near zero. The probe says whether orientation is information
the features lack, not that it helps a model. Exploration record: benchmarks/byconstruction_2026-10-22.md.

  python3 haar_orient_probe.py --out DIR [--sets kadid_train,tid2013,konfig_train] [--workers 6]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from linpyr_probe import BANK, down, kind, linear, xyb


def bands(a):
    """Haar detail energies of 2x2 blocks: horizontal edges (rows differ), vertical edges (columns differ), diagonal."""
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    p, q, r, s = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
    return 0.25 * (p + q - r - s), 0.25 * (p - q + r - s), 0.25 * (p - q - r + s)


def one(args):
    ref_path, dist_path = args
    yr, yd = xyb(linear(ref_path))[..., 1], xyb(linear(dist_path))[..., 1]
    if yr.shape != yd.shape:
        return None
    out = []
    for _s in range(4):
        br, bd = bands(yr), bands(yd)
        e = [float(((x - y) ** 2).mean()) for x, y in zip(br, bd)]
        e_ref = [float((x ** 2).mean()) for x in br]
        out.append([e, e_ref])
        yr, yd = down(yr[..., None])[..., 0], down(yd[..., None])[..., 0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="kadid_train,tid2013,konfig_train")
    ap.add_argument("--workers", type=int, default=6)
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
    (out / "orient_rows.json").write_text(json.dumps(rows))
    eps = 1e-12
    summary = {}
    for row in rows:
        for s, (e, e_ref) in enumerate(row["scales"]):
            tot = sum(e) + eps
            d = summary.setdefault(row["kind"], {}).setdefault(s, {"diag": [], "asym": [], "diag_ref": []})
            d["diag"].append(e[2] / tot)
            d["asym"].append(abs(e[0] - e[1]) / (e[0] + e[1] + eps))
            d["diag_ref"].append(e_ref[2] / (sum(e_ref) + eps))
    med = {k: {s: {m: round(float(np.median(v)), 4) for m, v in d.items()} for s, d in sc.items()}
           for k, sc in sorted(summary.items())}
    (out / "orient_summary.json").write_text(json.dumps(med, indent=1))
    print("== median per type, scales 0..3: diagonal share of difference energy (reference's own) | H/V asymmetry")
    for k, sc in med.items():
        cells = "  ".join(f"{sc[s]['diag']:.2f}({sc[s]['diag_ref']:.2f})|{sc[s]['asym']:.2f}" for s in range(4))
        print(f"  {k:28s} {cells}")


if __name__ == "__main__":
    main()
