"""Label-free damage-mixture test: adding unambiguous damage on top of a distortion must not raise the score.

Exploration record: benchmarks/byconstruction_2026-10-22.md. No human label is read.

  generate --out DIR [--refs 48] [--d1 standard|global]
                                      write ref / D1 / D2∘D1 PNGs and manifest.json
  score    --out DIR --bake NAME=PATH [--bake ...] [--jobs 6] [--post raw]
  report   --out DIR

References: TRAIN-eligible E34 inventory origins (256 px, `p4-inventory-r2/references`), excluding E34's 128
augmentation and 8 development origins, chosen by sha256(origin) order within each category prefix.
D1 (base distortion): Gaussian blur σ 1.2, luma noise σ 4, chroma noise σ 8, posterize to 12 levels. `--d1 global`
uses global edits instead, where a global-change descriptor acts: luma offset ±8, luma contrast ×0.85 / ×1.15 about the
image's mean luma, and chroma (saturation) ×0.8 / ×1.25.
D2 (added on top): luma noise σ 4 / 8 / 16 and chroma noise σ 8 / 16 / 32, fixed seeds. Noise is added in a
BT.601 YCbCr decomposition and the result rounded and clipped to 8 bits. Every D2 adds error energy against the
reference, so a score that rises from D1 to D2∘D1 is an inversion.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

INV = Path("/mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2")
SCORER = "score_pair_with_bake"


def ycc(rgb):
    r, g, b = (rgb[..., i] for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, 0.564 * (b - y), 0.713 * (r - y)


def rgb(y, cb, cr):
    r = y + cr / 0.713
    b = y + cb / 0.564
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.stack([r, g, b], axis=-1)


def to8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def blur(a, sigma):
    k = np.arange(-int(3 * sigma) - 1, int(3 * sigma) + 2)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    out = a.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * 3
        pad[axis] = (len(k) // 2, len(k) // 2)
        p = np.pad(out, pad, mode="reflect")
        out = sum(wi * np.take(p, np.arange(i, i + a.shape[axis]), axis=axis) for i, wi in enumerate(w))
    return out


def luma_noise(a, sigma, seed):
    y, cb, cr = ycc(a.astype(np.float64))
    y = y + np.random.default_rng(seed).normal(0, sigma, y.shape)
    return rgb(y, cb, cr)


def chroma_noise(a, sigma, seed):
    y, cb, cr = ycc(a.astype(np.float64))
    g = np.random.default_rng(seed)
    return rgb(y, cb + g.normal(0, sigma, cb.shape), cr + g.normal(0, sigma, cr.shape))


def posterize(a, levels):
    step = 255.0 / (levels - 1)
    return np.rint(a.astype(np.float64) / step) * step


D1 = {"blur1.2": lambda a, s: blur(a, 1.2), "luma4": lambda a, s: luma_noise(a, 4, s),
      "chroma8": lambda a, s: chroma_noise(a, 8, s), "poster12": lambda a, s: posterize(a, 12)}
def luma_offset(a, d):
    y, cb, cr = ycc(a.astype(np.float64))
    return rgb(y + d, cb, cr)


def luma_contrast(a, k):
    y, cb, cr = ycc(a.astype(np.float64))
    return rgb(y.mean() + k * (y - y.mean()), cb, cr)


def saturation(a, k):
    y, cb, cr = ycc(a.astype(np.float64))
    return rgb(y, k * cb, k * cr)


D1_GLOBAL = {"bright8": lambda a, s: luma_offset(a, 8), "dark8": lambda a, s: luma_offset(a, -8),
             "contrast0.85": lambda a, s: luma_contrast(a, 0.85), "contrast1.15": lambda a, s: luma_contrast(a, 1.15),
             "sat0.8": lambda a, s: saturation(a, 0.8), "sat1.25": lambda a, s: saturation(a, 1.25)}
D2 = {f"luma{s}": (lambda a, seed, s=s: luma_noise(a, s, seed)) for s in (4, 8, 16)}
D2.update({f"chroma{s}": (lambda a, seed, s=s: chroma_noise(a, s, seed)) for s in (8, 16, 32)})


def pick_refs(n):
    sel = json.loads((INV / "selection.json").read_text())
    used = {str(o) for part in ("augmentation", "development") for o in sel[part]["origins"]}
    elig = json.loads((INV / "eligibility.json").read_text())["eligible"]
    by_cat = {}
    for e in elig:
        o = str(e["origin"])
        if e.get("excluded_family") or o in used or not (INV / "references" / f"{o}.png").is_file():
            continue
        by_cat.setdefault(o[:2], []).append(o)
    for v in by_cat.values():
        v.sort(key=lambda o: hashlib.sha256(o.encode()).hexdigest())
    out, i = [], 0
    while len(out) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop(0))
        i += 1
    return out


def cmd_generate(a):
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=False)
    rows = []
    for ri, origin in enumerate(pick_refs(a.refs)):
        ref = np.asarray(Image.open(INV / "references" / f"{origin}.png").convert("RGB"))
        rp = out / "img" / f"{origin}_ref.png"
        Image.fromarray(ref).save(rp)
        for di, (d1, f1) in enumerate((D1_GLOBAL if a.d1 == "global" else D1).items()):
            base = to8(f1(ref, 1000 * ri + di))
            bp = out / "img" / f"{origin}_{d1}.png"
            Image.fromarray(base).save(bp)
            for ei, (d2, f2) in enumerate(D2.items()):
                more = to8(f2(base, 100000 + 1000 * ri + 10 * di + ei))
                mp = out / "img" / f"{origin}_{d1}+{d2}.png"
                Image.fromarray(more).save(mp)
                rows.append({"origin": origin, "d1": d1, "d2": d2, "ref": rp.name, "base": bp.name, "more": mp.name,
                             "mse_base": float(np.mean((base.astype(float) - ref) ** 2)),
                             "mse_more": float(np.mean((more.astype(float) - ref) ** 2))})
    (out / "manifest.json").write_text(json.dumps({"refs": a.refs, "d1": a.d1, "rows": rows}, indent=1) + "\n")
    worse = sum(r["mse_more"] > r["mse_base"] for r in rows)
    print(f"{len(rows)} compositions; RGB MSE rises from D1 to D2∘D1 in {worse} of them")
    return 0


def cmd_score(a):
    out = Path(a.out)
    man = json.loads((out / "manifest.json").read_text())
    images = sorted({(r["ref"], r["base"]) for r in man["rows"]} | {(r["ref"], r["more"]) for r in man["rows"]})
    bin_dir = Path(a.bin_dir)
    for spec in a.bake:
        name, path = spec.split("=", 1)
        dest = out / f"scores_{name}.json"
        if dest.exists():
            print(f"exists: {dest}")
            continue

        def one(pair, path=path):
            ref, dist = pair
            r = subprocess.run([str(bin_dir / SCORER), "--bake", path, "--bake-post", a.post, "--ref",
                                str(out / "img" / ref), "--dist", str(out / "img" / dist)],
                               capture_output=True, text=True, check=True)
            return dist, float(r.stdout.strip().splitlines()[-1])

        with cf.ThreadPoolExecutor(a.jobs) as pool:
            scores = dict(pool.map(one, images))
        dest.write_text(json.dumps({"bake": path, "post": a.post,
                                    "ring_form": os.environ.get("ZENSIM_RING_FORM", "raw"),
                                    "pyramid": os.environ.get("ZENSIM_PYRAMID", "xyb"),
                                    "bake_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                    "scores": scores}, indent=1) + "\n")
        print(f"{name}: {len(scores)} images scored")
    return 0


def cmd_report(a):
    out = Path(a.out)
    man = json.loads((out / "manifest.json").read_text())
    res = {}
    for f in sorted(out.glob("scores_*.json")):
        s = json.loads(f.read_text())["scores"]
        rise = np.array([s[r["more"]] - s[r["base"]] for r in man["rows"]])
        by = {}
        for r, d in zip(man["rows"], rise):
            by.setdefault(f'{r["d1"]}+{r["d2"]}', []).append(d)
        res[f.stem[7:]] = {"n": int(rise.size), "inversions": int((rise > 0).sum()),
                           "inversions_over_1pt": int((rise > 1).sum()), "max_rise": float(rise.max()),
                           "worst_cells": sorted(((k, float(np.max(v)), int(np.sum(np.array(v) > 0)))
                                                  for k, v in by.items()), key=lambda t: -t[1])[:5]}
    print(json.dumps(res, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--refs", type=int, default=48)
    g.add_argument("--d1", choices=("standard", "global"), default="standard")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--bake", action="append", required=True, help="NAME=PATH")
    s.add_argument("--bin-dir", required=True, help="directory holding score_pair_with_bake")
    s.add_argument("--post", default="raw")
    s.add_argument("--jobs", type=int, default=6)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"generate": cmd_generate, "score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
