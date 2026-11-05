#!/usr/bin/env python3
"""Restore the public checker, reproduce its complete TSV and safely reclaim."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    source = root / "public-after"
    kernel = digest(source / "zensim/src/feature_v2.rs")
    assert kernel == json.loads((root / "after/BINARY.json").read_text())["kernel_sha256"]
    assert not os.environ.get("RUSTFLAGS") and not os.environ.get("CARGO_ENCODED_RUSTFLAGS")
    env = dict(os.environ, TMPDIR=str(root / "scratch"), CARGO_TARGET_DIR=str(root / "target-tests"),
               CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0",
               CARGO_INCREMENTAL="0", RUST_TEST_THREADS="1")
    output = root / "public-after-parity.rebuilt.tsv"
    with output.open("x") as tsv:
        subprocess.run(["cargo", "run", "--release", "-p", "zensim", "--features",
                        "training,classification", "--example", "split_parity_check"],
                       cwd=source, env=env, stdout=tsv, check=True)
    original = root / "public-after-parity.tsv"
    assert digest(output) == digest(original), "rebuilt public checker TSV changed"
    artifact = root / "target-tests/release/examples/split_parity_check"
    binary = root / "bin/split-parity-public-gated"
    assert not binary.exists()
    shutil.copy2(artifact, binary)
    assert digest(artifact) == digest(binary)
    receipt = root / "PUBLIC_CHECKER_REBUILT.json"
    assert not receipt.exists()
    receipt.write_text(json.dumps({"kernel_sha256": kernel, "binary": str(binary),
                                   "binary_sha256": digest(binary),
                                   "original_tsv_sha256": digest(original),
                                   "rebuilt_tsv_sha256": digest(output),
                                   "rustc": subprocess.check_output(["rustc", "-vV"], text=True)}, indent=2) + "\n")
    # The frozen binary is outside Cargo output before any cache is removed.
    subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("reclaim_cargo.py")),
                    str(root), "rebuilt-public-checker"], check=True)


if __name__ == "__main__":
    main()
