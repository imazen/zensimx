#!/usr/bin/env python3
"""Independently decode retained JPEGs without replacing the timed RGB files."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    output = root / "FIXTURE_ORACLE.json"
    assert not output.exists(), output
    pairs = json.loads((root / "fixtures/pairs.json").read_text())
    paths = [root / "fixtures" / (pair["id"] + ".q40.jpg") for pair in pairs]
    assert len(paths) == len(set(paths)) == 260
    assert all(path.is_file() for path in paths)
    env = dict(os.environ, MAGICK_TMPDIR=str(root / "scratch"), MAGICK_THREAD_LIMIT="1")
    version = subprocess.check_output(["magick", "-version"], text=True, env=env)
    rows = []
    for path in paths:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        command = ["magick", "-regard-warnings", str(path), "-depth", "8", "RGB:-"]
        # Consume the full independently decoded raster. Only the retained
        # imazen RGB files are used by the timing instrument.
        with open(os.devnull, "wb") as sink:
            subprocess.run(command, env=env, stdout=sink, check=True)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        rows.append({"file": str(path.relative_to(root)), "sha256": before})
        print(f"decoded {len(rows)}/{len(paths)} {path.name}", flush=True)
    output.write_text(json.dumps({"decoder": version, "command": "magick -regard-warnings <jpeg> -depth 8 RGB:-",
                                  "accepted": len(rows), "required": len(paths), "files": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
