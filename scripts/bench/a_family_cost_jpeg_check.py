#!/usr/bin/env python3
"""Independent JPEG decode oracle; never supplies benchmark or training pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import PIL
from PIL import Image, features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    pairs = json.loads((a.study / "pairs/pairs.json").read_text())
    checks = []
    for pair in pairs:
        path = a.study / f'pairs/{pair["id"]}.jpg'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == pair["jpeg_sha256"]
        with Image.open(path) as image:
            assert image.format == "JPEG"
            image.load()
            assert image.size == (pair["width"], pair["height"])
            checks.append(dict(case=pair["id"], jpeg_sha256=sha,
                               size=list(image.size), mode=image.mode))
    assert len(checks) == 80
    assert not a.out.exists(), "preserve an existing oracle receipt"
    a.out.write_text(json.dumps(dict(pillow=PIL.__version__, jpeg_api=features.version("jpg"),
                                    libjpeg_turbo=features.version("libjpeg_turbo"),
                                    checks=checks), indent=2) + "\n")
    print(f'Independent full JPEG decode: {len(checks)}/80 pass; Pillow {PIL.__version__}; '
          f'JPEG API {features.version("jpg")}, libjpeg-turbo {features.version("libjpeg_turbo")}. '
          'No oracle pixels enter timing.')


if __name__ == "__main__":
    main()
