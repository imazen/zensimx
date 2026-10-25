"""TRAIN-only corpus for the JXL map-utility runs (guided coarse redistribution, coarse-upsample kernels).

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Map todo"). No labels are read. Sources, chosen in
SHA-256(name) order so the choice is fixed:
- photo: 12 CID22-512 training references (the 201-reference TRAIN side; never the 49 validation references);
- screen: 6 gb82-sc screenshots (messaging screenshots left out), centre-cropped to at most 512 x 512;
- illustration: 6 imazen-26 TRAIN-eligible E34-inventory origins from the AI clip art / AI illustration / plot /
  patent-scan families, centre-cropped from the full-size source.
Crops are pixel-exact (no resampling). Writes OUT/<name>.png and OUT/corpus.tsv (path, name, class).

  python3 jxl_map_corpus.py --out DIR [--photos 12] [--illustrations 6] [--screen-crops 1]

`--screen-crops 3` takes the centre, top-left and bottom-right 512 crops of each screenshot (distinct content).
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

CID22_TRAIN = Path("/home/lilith/work/codec-corpus/CID22/CID22-512/training")
# Two of the selected CID22 training references are graphics, not photographs (checked by eye, 2026-10-24).
CID22_GRAPHICS = {"newplot": "illustration", "ularapi_Semarang_City_Logo": "illustration"}
GB82 = Path("/home/lilith/work/codec-corpus/gb82-sc")
GB82_USE = ("codec_wiki", "gui", "terminal", "windows", "imac_dark", "windows95")
INV = Path("/mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/eligibility.json")
ILLUSTRATION_FAMILIES = ("9000-lilith-ai-clipart", "9094-lilith-ai-illustrations", "7000-lilith-plots",
                         "6000-lilith-scans-public-patents")


def by_hash(names):
    return sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())


def centre_crop(im: Image.Image, side: int = 512, where: str = "centre") -> Image.Image:
    w, h = im.size
    s = min(side, w, h)
    x0, y0 = {"centre": ((w - s) // 2, (h - s) // 2), "topleft": (0, 0), "bottomright": (w - s, h - s)}[where]
    return im.crop((x0, y0, x0 + s, y0 + s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--photos", type=int, default=12, help="CID22 training references (graphics included, relabelled)")
    ap.add_argument("--illustrations", type=int, default=6)
    ap.add_argument("--screen-crops", type=int, default=1, choices=(1, 2, 3))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in by_hash([p.stem for p in CID22_TRAIN.glob("*.png")])[:a.photos]:
        im = Image.open(CID22_TRAIN / f"{name}.png").convert("RGB")
        dest = out / f"cid22t_{name}.png"
        im.save(dest)
        rows.append((dest, dest.stem, CID22_GRAPHICS.get(name, "photo")))
    for name in GB82_USE:
        for where in ("centre", "topleft", "bottomright")[:a.screen_crops]:
            dest = out / (f"gb82_{name}.png" if where == "centre" else f"gb82_{name}_{where}.png")
            centre_crop(Image.open(GB82 / f"{name}.png").convert("RGB"), where=where).save(dest)
            rows.append((dest, dest.stem, "screen"))
    eligible = json.loads(INV.read_text())["eligible"]
    pool = [e for e in eligible if not e.get("excluded_family") and Path(e["source"]).parent.name in ILLUSTRATION_FAMILIES]
    for e in sorted(pool, key=lambda e: hashlib.sha256(str(e["origin"]).encode()).hexdigest())[:a.illustrations]:
        dest = out / f"i26_{e['origin']}.png"
        centre_crop(Image.open(e["source"]).convert("RGB")).save(dest)
        rows.append((dest, dest.stem, "illustration"))
    (out / "corpus.tsv").write_text("".join(f"{p}\t{n}\t{c}\n" for p, n, c in rows))
    print(f"{len(rows)} images -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
