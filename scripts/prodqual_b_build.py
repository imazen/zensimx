#!/usr/bin/env python3
"""Build canonical gate owners with a reviewable instrument-only CLI overlay.

The overlay only adds explicit --corpora none, preserving every numerical owner.
Store exact built source externally and restore the original owned bytes even on
build failure. No large existing source file is committed by this lane.
"""
import hashlib
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
out = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')
source = root / 'zensim-validate/src/bin/bake_verdict.rs'
original = source.read_bytes()
anchor = b"fn parse_corpora_arg(arg: &str) -> Result<Vec<&'static Corpus>, String> {\n"
assert original.count(anchor) == 1
updated = original.replace(anchor, anchor + b'    // Explicit instrument-only qualification: no human corpus is opened.\n    if arg == "none" {\n        return Ok(Vec::new());\n    }\n', 1)
(out / 'built-source').mkdir(exist_ok=True)
(out / 'built-source/bake_verdict.rs').write_bytes(updated)
env = dict(os.environ, TMPDIR=str(Path.home() / 'tmp/prodqual-b'),
           CARGO_TARGET_DIR=str(Path.home() / 'tmp/prodqual-b/target'),
           CARGO_INCREMENTAL='0', CARGO_PROFILE_DEV_DEBUG='0')
try:
    source.write_bytes(updated)
    subprocess.run(['cargo', 'build', '-p', 'zensim-validate', '--bin', 'bake_verdict',
                    '--bin', 'panel', '-p', 'zensim', '--example', 'diffmap_block_coherence',
                    '--features', 'zensim/feature-regime-v2,zensim/custom-profiles,zensim/candidate-profiles'],
                   cwd=root, env=env, check=True)
finally:
    assert source.read_bytes() == updated, 'concurrent source edit; refusing to overwrite'
    source.write_bytes(original)
assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(original).digest()
