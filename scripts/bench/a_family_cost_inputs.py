#!/usr/bin/env python3
"""Stage pinned TRAIN corpus bytes without modifying corpus checkouts.

Canonical png-v3 LFS pointers are resolved through the corpus's public access
endpoint and must match the checkout's SHA-256. Existing staged bytes are kept.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import struct
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train = sorted((a.corpus / "clic2025/training").glob("*.png"))[:20]
    assert len(train) == 20
    small = sorted((a.corpus / "CID22/CID22-512/training").glob("*.png"))[:10]
    small += [a.corpus / "clic2025-1024" / f.name for f in train[:10]]
    large = []
    for folder in ["1000-lilith-photos-general", "1200-lilith-interiors",
                   "1400-lilith-nature", "1600-lilith-food"]:
        # Canonical last-digit TRAIN split; five images per photograph folder.
        files = [f for f in sorted((a.canonical / folder).glob("*.sdr.png"))
                 if f.name[:4].isdigit() and int(f.name[3]) % 2 == 0
                 and max(map(int, re.search(r"_(\d+)x(\d+)\.sdr\.png$", f.name).groups())) >= 4000][:5]
        assert len(files) == 5, folder
        large += files
    sets = [("64", 64, None, small), ("256", 256, None, small),
            ("1024", None, None, [a.corpus / "clic2025-1024" / f.name for f in train]),
            ("large", None, 4096, large)]
    sources, pins = [], []
    for size, side, edge, files in sets:
        for i, source in enumerate(files):
            key = hashlib.sha256(str(source).encode()).hexdigest()[:16] + ".png"
            dest = a.out / key
            expected = None
            with source.open("rb") as fp:
                prefix = fp.read(256)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                expected = prefix.decode().split("oid sha256:")[1].splitlines()[0]
            if not dest.exists():
                if expected:
                    relative = source.relative_to(a.canonical).as_posix()
                    url = "https://codec-corpus.r2.imazen.org/imazen-26-png-v3/" + relative
                    subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                                    "--max-time", "180", "--output", str(dest), url], check=True)
                else:
                    shutil.copyfile(source, dest)
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            assert expected is None or sha == expected, (source, sha, expected)
            with dest.open("rb") as fp:
                header = fp.read(24)
            assert header[:8] == b"\x89PNG\r\n\x1a\n", source
            w, h = struct.unpack(">II", header[16:24])
            if edge:
                assert max(w, h) >= 4000, (source, w, h)
            case_id = f"{size}_{i:02}"
            sources.append(dict(id=case_id, path=a.remote_root + "/" + key,
                                size=size, side=side, long_edge=edge))
            pins.append(dict(id=case_id, source=str(source), sha256=sha,
                             source_width=w, source_height=h))
            print(case_id, sha, source, flush=True)
            (a.out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
            (a.out / "source-pins.json").write_text(json.dumps(pins, indent=2) + "\n")


if __name__ == "__main__":
    main()
