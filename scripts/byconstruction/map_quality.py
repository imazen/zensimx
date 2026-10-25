"""Map-quality study for the candidate attribution map (by-construction exploration, "Maps" in
benchmarks/byconstruction_2026-10-22.md). TRAIN images only; no labels are read.

  pairs   --out DIR              pick the pair list: KADID E21 TRAIN (3 references x 25 types, level 3) and the JXL
                                 map corpus (CID22 training references, gb82 screenshots, imazen-26 TRAIN illustrations)
                                 at distance 4.0 (JXL without the loop, `none` arm)
  leak    --out DIR [--n 30]     for the first N pairs, write two repaired distorted images each (a 64-px square of the
                                 distorted image replaced by the reference: the square with the most butteraugli mass,
                                 and a fixed off-centre one) and their pair list
  metrics --out DIR --tags A,B   per tag: block agreement with the judges (SROCC of 16-px block sums against butteraugli,
                                 SSIMULACRA2's score mass and DVIFM 0.3's map; judges against each other), locality
                                 (fraction of the map change outside a repaired square) and the mass a map puts where
                                 the two images are equal
  gallery --out DIR --tags A,B [--keys k1,k2]   side-by-side PNG panels, one shared colour scale per row

`map_dump` (zensim-bench) writes the maps: DIR/maps/<key>.<tag>.f32 for the candidate under each research paint mode,
`legacy` (B's basic-plane diffmap), `butter`, `ssim2`, `dvifm`.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

KADID_PAIRS = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/dvifm03/dvifm.pairs.tsv")
JXL = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/jxl-maprd-large/jxl")


def by_hash(xs):
    return sorted(xs, key=lambda s: hashlib.sha256(s.encode()).hexdigest())


def cmd_pairs(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    kp = pd.read_csv(KADID_PAIRS, sep="\t")
    kp = kp[kp.ref_path.str.contains("/kadid10k/")]
    refs = by_hash(sorted(set(kp.ref_path)))[: a.kadid_refs]
    for ref in refs:
        for t in range(1, 26):
            stem = Path(ref).stem
            dist = Path(ref).with_name(f"{stem}_{t:02d}_03.png")
            if dist.exists() and (kp.dist_path == str(dist)).any():
                rows.append((f"k{stem}_{t:02d}", ref, str(dist), "kadid"))
    man = pd.read_csv(JXL / "manifest_none.tsv", sep="\t")
    for _, r in man[man.distance.round(2) == 4.0].iterrows():
        rows.append((f"j{r.image}_d4", r.ref_path, r.dist_path, r.corpus))
    with open(out / "pairs.tsv", "w") as f:
        f.write("key\tref_path\tdist_path\tgroup\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"{len(rows)} pairs -> {out / 'pairs.tsv'}")
    return 0


def load(out: Path, key: str, tag: str, shape):
    v = np.fromfile(out / "maps" / f"{key}.{tag}.f32", dtype="<f4").astype(np.float64)
    return v.reshape(shape)


def rows_of(out: Path):
    import pandas as pd
    return pd.read_csv(out / "pairs.tsv", sep="\t")


def dims(out: Path):
    d = {}
    for line in (out / "maps" / "judges.jsonl").read_text().splitlines():
        r = json.loads(line)
        d[r["key"]] = (r["height"], r["width"])
    return d


def blocks(m, b=16):
    h, w = m.shape
    hb, wb = h // b, w // b
    return m[: hb * b, : wb * b].reshape(hb, b, wb, b).sum(axis=(1, 3)).ravel()


def srocc(x, y):
    from scipy.stats import spearmanr
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def cmd_leak(a):
    from PIL import Image
    out = Path(a.out)
    rows = rows_of(out).head(a.n)
    dd = dims(out)
    lk = out / "leak"
    lk.mkdir(exist_ok=True)
    recs = []
    for _, r in rows.iterrows():
        ref = np.asarray(Image.open(r.ref_path).convert("RGB"))
        dist = np.asarray(Image.open(r.dist_path).convert("RGB"))
        h, w = dd[r.key]
        butter = load(out, r.key, "butter", (h, w))
        s = 64
        if h < 2 * s or w < 2 * s:
            continue
        # The square with the most butteraugli mass, on a 32-px lattice, and a fixed off-centre square.
        best, at = -1.0, (0, 0)
        for y in range(0, h - s + 1, 32):
            for x in range(0, w - s + 1, 32):
                m = butter[y : y + s, x : x + s].sum()
                if m > best:
                    best, at = m, (y, x)
        fixed = ((h // 3) // 32 * 32, (w // 3) // 32 * 32)
        for name, (y, x) in (("hot", at), ("fixed", fixed)):
            rep = dist.copy()
            rep[y : y + s, x : x + s] = ref[y : y + s, x : x + s]
            p = lk / f"{r.key}.{name}.png"
            Image.fromarray(rep).save(p)
            recs.append({"key": f"{r.key}~{name}", "base": r.key, "ref_path": r.ref_path, "dist_path": str(p),
                         "y": int(y), "x": int(x), "size": s})
    with open(lk / "pairs.tsv", "w") as f:
        f.write("key\tref_path\tdist_path\n")
        for q in recs:
            f.write(f"{q['key']}\t{q['ref_path']}\t{q['dist_path']}\n")
    (lk / "squares.json").write_text(json.dumps(recs, indent=1) + "\n")
    print(f"{len(recs)} repaired images -> {lk}")
    return 0


def cmd_metrics(a):
    from PIL import Image
    out = Path(a.out)
    rows = rows_of(out)
    dd = dims(out)
    tags = a.tags.split(",")
    judges = ["butter", "ssim2", "dvifm"]
    res = {"tags": {}, "judges": {}}
    # Judges against each other.
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1 :]:
            v = [srocc(blocks(load(out, k, j1, dd[k])), blocks(load(out, k, j2, dd[k]))) for k in rows.key]
            res["judges"][f"{j1}~{j2}"] = float(np.nanmedian(v))
    squares = json.loads((out / "leak" / "squares.json").read_text()) if (out / "leak" / "squares.json").exists() else []
    for tag in tags:
        t = {}
        for j in judges:
            v = [srocc(blocks(np.abs(load(out, k, tag, dd[k]))), blocks(load(out, k, j, dd[k]))) for k in rows.key]
            t[f"srocc_{j}"] = float(np.nanmedian(v))
            t[f"srocc_{j}_by_group"] = {g: float(np.nanmedian([vv for vv, gg in zip(v, rows.group) if gg == g]))
                                        for g in sorted(set(rows.group))}
        # Mass where the images are equal (all three channels identical at that pixel).
        eq = []
        for _, r in rows.iterrows():
            m = np.abs(load(out, r.key, tag, dd[r.key]))
            ref = np.asarray(Image.open(r.ref_path).convert("RGB")).astype(np.int16)
            dist = np.asarray(Image.open(r.dist_path).convert("RGB")).astype(np.int16)
            same = (np.abs(ref - dist).max(axis=2) == 0)
            tot = m.sum()
            if tot > 0 and same.any() and not same.all():
                eq.append(float(m[same].sum() / tot))
        t["mass_where_equal_median"] = float(np.median(eq)) if eq else None
        # Locality: |change| outside the repaired square / total |change|.
        leaks = []
        for q in squares:
            k = q["base"]
            base = load(out, k, tag, dd[k])
            try:
                rep = load(out, q["key"], tag, dd[k])
            except FileNotFoundError:
                continue
            delta = np.abs(rep - base)
            y, x, s = q["y"], q["x"], q["size"]
            inside = delta[y : y + s, x : x + s].sum()
            tot = delta.sum()
            if tot > 0:
                leaks.append(float(1.0 - inside / tot))
        t["leak_median"] = float(np.median(leaks)) if leaks else None
        t["leak_p90"] = float(np.quantile(leaks, 0.9)) if leaks else None
        res["tags"][tag] = t
        print(tag, json.dumps({k: v for k, v in t.items() if not isinstance(v, dict)}))
    (out / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    print("judges", json.dumps(res["judges"]))
    return 0


def inferno(t):
    stops = np.array([[0, 0, 4], [27, 12, 65], [74, 12, 107], [120, 28, 109], [165, 44, 96], [207, 68, 70],
                      [237, 105, 37], [251, 155, 6], [247, 209, 61], [252, 255, 164]], dtype=np.float64)
    t = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def cmd_gallery(a):
    from PIL import Image, ImageDraw
    out = Path(a.out)
    rows = rows_of(out)
    dd = dims(out)
    tags = a.tags.split(",")
    keys = a.keys.split(",") if a.keys else list(rows.key[:: max(1, len(rows) // 12)][:12])
    gal = out / "gallery"
    gal.mkdir(exist_ok=True)
    for k in keys:
        r = rows[rows.key == k].iloc[0]
        h, w = dd[k]
        panels = [("reference", np.asarray(Image.open(r.ref_path).convert("RGB"))),
                  ("distorted", np.asarray(Image.open(r.dist_path).convert("RGB")))]
        for tag in tags:
            m = np.abs(load(out, k, tag, (h, w)))
            top = np.quantile(m, 0.995) or m.max() or 1.0
            panels.append((tag, inferno(np.sqrt(m / top))))
        pad = 18
        canvas = Image.new("RGB", (w * len(panels), h + pad), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for i, (name, img) in enumerate(panels):
            canvas.paste(Image.fromarray(img), (i * w, pad))
            d.text((i * w + 4, 2), name, fill=(0, 0, 0))
        canvas.save(gal / f"{k}.png")
    print(f"{len(keys)} panels -> {gal}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--kadid-refs", type=int, default=3)
    p = sub.add_parser("leak")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=30)
    p = sub.add_parser("metrics")
    p.add_argument("--out", required=True)
    p.add_argument("--tags", required=True)
    p = sub.add_parser("gallery")
    p.add_argument("--out", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--keys", default="")
    a = ap.parse_args()
    return {"pairs": cmd_pairs, "leak": cmd_leak, "metrics": cmd_metrics, "gallery": cmd_gallery}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
