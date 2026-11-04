#!/usr/bin/env bash
# Caller holds the shared heavy lock and run-heavy scope.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TMPDIR=$HOME/tmp/zensim-linear CARGO_TARGET_DIR=/var/tmp/byc-linear-20261104/target
cargo fmt -p zensim -p zensim-validate --check
python3 -B -m unittest discover -s scripts/tests -p test_subjective_training_leg.py -v
python3 -B scripts/lint_scripts.py
just clippy
