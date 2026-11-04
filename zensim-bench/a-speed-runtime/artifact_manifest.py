#!/usr/bin/env python3
"""Freeze an external artifact inventory after all writers have stopped."""
import hashlib
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
manifest = root / "SHA256SUMS"
assert not manifest.exists(), "preserve the existing frozen inventory"
rows, total = [], 0
for directory, dirs, files in os.walk(root):
    dirs[:] = sorted(d for d in dirs if d not in {"target", ".git", ".jj"})
    for name in sorted(files):
        if name == ".workongoing":
            continue
        path = Path(directory) / name
        assert not path.is_symlink(), (path, "archive symlinks require an explicit policy")
        relative = str(path.relative_to(root))
        assert "\n" not in relative and "\\" not in relative
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(f"{digest}  {relative}\n")
        total += path.stat().st_size
manifest.write_text("".join(sorted(rows)))
print(f"files={len(rows)} bytes={total} manifest_sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}")
