"""Prototype: masking-normalised band-error features (Python, exploration only).

Exploration record: benchmarks/byconstruction_2026-10-22.md, section "AIC2026 TRAIN". A candidate feature family for
near-threshold (JND) ranking, measured before any Rust implementation:

- XYB (the JPEG XL opsin transform, cube-root; X, Y, B planes) of each sRGB8 image;
- a Laplacian pyramid per plane: G_0 = plane, G_{s+1} = 2x2 box mean of G_s, band L_s = G_s - up2(G_{s+1}) for
  s = 0..S-1 (up2 = pixel replication), so every band is zero-mean in flat regions;
- the error in band s: e = |L_s(dst) - L_s(src)|; the reference's local band energy m = box_r(|L_s(src)|) (radius 2 at
  that band's resolution); a masking threshold t = 1 + (m / k_c)^alpha;
- pooled per band and plane: the p-norm over pixels of e / t for p in P, and the max.

Every value is a non-negative damage measurement that is 0 for identical images and non-decreasing in the error at every
pixel, so a non-negative combination of them is monotone in pixel damage by construction.

With `--tiles T` the image is cut into a T x T grid of tiles (at every band's resolution) and each tile's
p-norm (p = 4) of the masked error is computed; the image features are the max over tiles, the mean of the top 10% of
tiles and the mean over tiles, per band, plane and masking exponent.

  python3 jnd_band_proto.py --pairs PAIRS.tsv --out OUT.parquet [--jobs 16] [--tiles 12] [--family v3]
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

S = 7
ALPHAS = (0.0, 0.5, 1.0)  # 0 = unmasked
P = (2, 4, 8)
K = {"X": 0.004, "Y": 0.02, "B": 0.02}  # masking knee per plane (XYB units), fixed before any result


def xyb(path: str) -> dict:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    bias = 0.0037930732552754493
    m0 = 0.30 * r + 0.622 * g + 0.078 * b + bias
    m1 = 0.23 * r + 0.692 * g + 0.078 * b + bias
    m2 = 0.24342268924547819 * r + 0.20476744424496821 * g + 0.55180986650955360 * b + bias
    cb = np.cbrt(bias)
    l0, l1, l2 = np.cbrt(m0) - cb, np.cbrt(m1) - cb, np.cbrt(m2) - cb
    return {"X": (l0 - l1) / 2, "Y": (l0 + l1) / 2, "B": l2}


def down(p):
    h, w = p.shape[0] // 2 * 2, p.shape[1] // 2 * 2
    p = p[:h, :w]
    return 0.25 * (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2])


def bands(p):
    out = []
    for _ in range(S):
        n = down(p)
        up = np.repeat(np.repeat(n, 2, 0), 2, 1)
        out.append(p[: up.shape[0], : up.shape[1]] - up)
        p = n
    return out


def box(m, r=2):
    c = np.pad(m, r, mode="edge").cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * r + 1
    return np.maximum((c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n), 0.0)  # cumsum rounding


_REF = {}
TILES = 0


def tile_pool(r, tag, out):
    h, w = r.shape
    T = TILES
    if h < T or w < T:
        return
    ys, xs = np.linspace(0, h, T + 1).astype(int), np.linspace(0, w, T + 1).astype(int)
    v = np.array([np.mean(r[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] ** 4) ** 0.25 for i in range(T) for j in range(T)])
    v.sort()
    out[f"{tag}tmax"] = float(v[-1])
    out[f"{tag}ttop"] = float(v[-max(1, len(v) // 10):].mean())
    out[f"{tag}tmean"] = float(v.mean())


FAMILY = "v1"


def features_v3(rb, db):
    """Band-energy gain (added detail: ringing, blocking) and loss (blur) next to the symmetric error, per band/plane,
    unmasked and masked (alpha 0.5), pooled as p2, p4 and the top-10% / max of a 12 x 12 tile grid (tile p4)."""
    out = {}
    for c in "XYB":
        for s in range(6):
            er, ed = box(np.abs(rb[c][s]), 1), box(np.abs(db[c][s]), 1)
            maps = {"d": np.abs(db[c][s] - rb[c][s]), "g": np.maximum(ed - er, 0.0), "l": np.maximum(er - ed, 0.0)}
            m = box(np.abs(rb[c][s]))
            for kind, e in maps.items():
                for alpha in (0.0, 0.5):
                    r = e if alpha == 0 else e / (1.0 + (m / K[c]) ** alpha)
                    tag = f"{c}{s}{kind}a{alpha:g}"
                    out[f"{tag}p2"] = float(np.sqrt(np.mean(r * r)))
                    out[f"{tag}p4"] = float(np.mean(r ** 4) ** 0.25)
                    h, w = r.shape
                    if h >= 12 and w >= 12:
                        ys, xs = np.linspace(0, h, 13).astype(int), np.linspace(0, w, 13).astype(int)
                        v = np.sort([np.mean(r[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] ** 4) ** 0.25
                                     for i in range(12) for j in range(12)])
                        out[f"{tag}ttop"] = float(v[-14:].mean())
                        out[f"{tag}tmax"] = float(v[-1])
    return out


def spectral_residual(y):
    """Spectral-residual saliency (Hou & Zhang 2007) of a plane: computed at 64 px on the long side, nearest-upsampled."""
    h, w = y.shape
    f = max(1, max(h, w) // 64)
    small = y[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f).mean((1, 3))
    F = np.fft.fft2(small)
    logamp = np.log(np.abs(F) + 1e-9)
    avg = box(logamp, 1)
    sal = np.abs(np.fft.ifft2(np.exp(logamp - avg + 1j * np.angle(F)))) ** 2
    sal = box(box(sal, 2), 2)
    big = np.repeat(np.repeat(sal, f, 0), f, 1)
    out = np.full((h, w), big.mean())
    out[: big.shape[0], : big.shape[1]] = big
    return out / out.mean()


def features_v4(ref_planes, rb, db):
    """Attention-weighted pooling of the luma band error at scales 0-3: centre bias, spectral-residual saliency of the
    reference, and display brightness (the reference's own Y), each as a weighted p2 and p4 mean; unweighted p4 too."""
    out = {}
    y0 = ref_planes["Y"]
    sal0 = spectral_residual(y0)
    for s in range(4):
        e = np.abs(db["Y"][s] - rb["Y"][s])
        h, w = e.shape
        f = 2 ** s
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx, sg = (h - 1) / 2, (w - 1) / 2, 0.25 * min(h, w)
        centre = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg * sg)))
        sal = sal0[: h * f : f, : w * f : f][:h, :w]
        lum = np.maximum(down_n(y0, s)[:h, :w], 0.0)
        out[f"Y{s}p4"] = float(np.mean(e ** 4) ** 0.25)
        for name, wt in (("ctr", centre), ("sal", sal), ("lum", lum)):
            wt = wt / wt.mean()
            out[f"Y{s}{name}p2"] = float(np.sqrt(np.mean(wt * e * e)))
            out[f"Y{s}{name}p4"] = float(np.mean(wt * e ** 4) ** 0.25)
    return out


def down_n(p, n):
    for _ in range(n):
        p = down(p)
    return p


def features(args):
    ref_path, dist_path = args
    if ref_path not in _REF:
        _REF.clear()
        _REF[ref_path] = {c: bands(v) for c, v in xyb(ref_path).items()}
    rb = _REF[ref_path]
    db = {c: bands(v) for c, v in xyb(dist_path).items()}
    if FAMILY == "v3":
        return features_v3(rb, db)
    if FAMILY == "v4":
        return features_v4(xyb(ref_path), rb, db)
    out = {}
    for c in "XYB":
        for s in range(S):
            e = np.abs(db[c][s] - rb[c][s])
            m = box(np.abs(rb[c][s]))
            for alpha in ALPHAS:
                r = e if alpha == 0 else e / (1.0 + (m / K[c]) ** alpha)
                tag = f"{c}{s}a{alpha:g}"
                if TILES:
                    tile_pool(r, tag, out)
                    continue
                for p in P:
                    out[f"{tag}p{p}"] = float(np.mean(r ** p) ** (1.0 / p))
                out[f"{tag}max"] = float(r.max())
    return out


def _set_tiles(t, fam="v1"):
    global TILES, FAMILY
    TILES, FAMILY = t, fam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--tiles", type=int, default=0)
    ap.add_argument("--family", default="v1", choices=("v1", "v3", "v4"))
    a = ap.parse_args()
    global TILES
    TILES = a.tiles
    p = pd.read_csv(a.pairs, sep="\t")
    # rows sorted by reference so each worker reuses its reference's pyramid; restore file order after
    order = np.argsort(p.ref_path.to_numpy(), kind="stable")
    args = [(p.ref_path[i], p.dist_path[i]) for i in order]
    with mp.Pool(a.jobs, initializer=_set_tiles, initargs=(a.tiles, a.family)) as pool:
        res = pool.map(features, args, chunksize=32)
    df = pd.DataFrame(res)
    df.index = order
    df = df.sort_index()
    df.insert(0, "key", p.key.astype(str).to_numpy())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} rows, {df.shape[1] - 1} features -> {a.out}")


if __name__ == "__main__":
    main()
