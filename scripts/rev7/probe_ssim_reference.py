#!/usr/bin/env python3
"""Compile-probe test-only SSIM references, restoring the exact source bytes in finally.

Run under run-heavy with no concurrent cargo job. OUT must be new; it retains
the original source and complete compiler output. No source change is landed.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = root / "zensim/src/ssim_form.rs"
    original = path.read_bytes()
    source = original.decode()
    changed = source
    for marker in ["#[derive(Default)]\npub(crate) struct StableSsimScratch", "#[autoversion]\npub(crate) fn stable_ssim_plane"]:
        assert changed.count(marker) == 1, f"source owner changed: {marker}"
        changed = changed.replace(marker, "#[cfg(test)]\n" + marker)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "ssim_form.rs.bak").write_bytes(original)
    try:
        path.write_text(changed)
        command = ["cargo", "check", "--locked", "-p", "zensim", "--lib", "--all-features"]
        with (args.out / "check.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        print(f"test-only StableSsimScratch/stable_ssim_plane: cargo check rc={result.returncode}")
        if result.returncode:
            print((args.out / "check.log").read_text())
        return result.returncode
    finally:
        path.write_bytes(original)
        assert path.read_bytes() == original
        print("restored ssim_form.rs sha256=" + hashlib.sha256(original).hexdigest())


if __name__ == "__main__":
    raise SystemExit(main())
