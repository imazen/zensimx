#!/usr/bin/env python3
"""Speed-pass case matrix: image sizes x content types for byc_score_timing.

Writes center crops of codec-corpus images (lossless PNG) plus `cases.tsv` (label, class, width, height, harness
arguments) into OUT_DIR. Synthetic cases use the harness's own textured pair (`--width/--height`); image cases pass
`--image PATH`, which every harness build since the speed pass's start accepts, so before/after arms need no new flag.
Crops are taken from the image center at the stated size, or the whole image when it is smaller. The images are
timing inputs only: no labels, no training. Record: benchmarks/byconstruction_2026-10-22.md ("Speed pass").

  perf_cases.py OUT_DIR
"""

import sys
from pathlib import Path

from PIL import Image

CC = Path.home() / "work/codec-corpus"
G82 = Path.home() / ".cache/codec-corpus/v1/gb82"
I26 = CC / "imazen-26"

# (label, class, source or None for synthetic, (width, height) crop or None for the whole image)
CASES = [
    ("syn64", "synthetic", None, (64, 64)),
    ("syn256", "synthetic", None, (256, 256)),
    ("syn1024", "synthetic", None, (1024, 1024)),
    ("syn2048", "synthetic", None, (2048, 2048)),
    ("syn4096", "synthetic", None, (4096, 4096)),
    ("photo-gb82", "photo", G82 / "city-lossless.png", None),
    ("photo-clic", "photo", "clic", None),
    ("photo-clic256", "photo", "clic", (256, 256)),
    ("photo-clic64", "photo", "clic", (64, 64)),
    ("screen-web", "screen", "web", None),
    ("screen-web256", "screen", "web", (256, 256)),
    ("screen-mobile", "screen", "mobile", None),
    ("screen-gui", "screen", CC / "gb82-sc/gui.png", None),
    ("plot", "lineart", "plot", None),
    ("plot64", "lineart", "plot", (64, 64)),
    ("clipart", "illustration", "clipart", None),
    ("text-scan", "document", "text", None),
    ("brochure4096", "document", "brochure", (4096, 4096)),
]


def first(pattern: str) -> Path:
    for p in sorted(I26.glob(pattern)):
        if p.stat().st_size > 2000:  # skip LFS pointers
            return p
    raise SystemExit(f"no image for {pattern} under {I26}")


def resolve(src):
    named = {
        "clic": lambda: sorted((CC / "clic2025/training").glob("*.png"))[0],
        "web": lambda: first("8100-lilith-web-screenshots/**/*.png"),
        "mobile": lambda: first("8000-lilith-mobile-screenshots/**/*.png"),
        "plot": lambda: first("7000-lilith-plots/**/*.png"),
        "clipart": lambda: first("9000-lilith-ai-clipart/**/*.png"),
        "text": lambda: first("6800-ia-scans-manuscript-text/**/*.png"),
        "brochure": lambda: first("5000-national-park-service-brochures/**/*.png"),
    }
    return named[src]() if isinstance(src, str) else src


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = ["label\tclass\twidth\theight\tsource\targs"]
    for label, cls, src, crop in CASES:
        if src is None:
            w, h = crop
            rows.append(f"{label}\t{cls}\t{w}\t{h}\tsynthetic\t--width {w} --height {h}")
            continue
        path = resolve(src)
        im = Image.open(path).convert("RGB")
        if crop is not None:
            cw, ch = min(crop[0], im.width), min(crop[1], im.height)
            x0, y0 = (im.width - cw) // 2, (im.height - ch) // 2
            im = im.crop((x0, y0, x0 + cw, y0 + ch))
        dst = out / f"{label}.png"
        im.save(dst)
        rows.append(f"{label}\t{cls}\t{im.width}\t{im.height}\t{path}\t--image {dst}")
    (out / "cases.tsv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
