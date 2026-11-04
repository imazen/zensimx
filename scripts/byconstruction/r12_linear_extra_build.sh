#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CARGO_TARGET_DIR=/var/tmp/byc-linear-20261104/target TMPDIR=$HOME/tmp/zensim-linear
cargo test --release -p zensim-validate --lib r12_two_piece_learns_single
cargo test --release -p zensim-validate --bin bake_dial_refit r12_hidden_pruning
cargo test --release -p zensim-validate --bin zensim_mlp_train subjective_training_leg::tests
cargo build --release -p zensim-validate --bin bake_dial_refit --bin zensim_mlp_train --bin predict_features_with_bake --example prune_forward_bench
O=/var/tmp/byc-linear-20261104/extra/bin
cp "$CARGO_TARGET_DIR/release/bake_dial_refit" "$CARGO_TARGET_DIR/release/zensim_mlp_train" "$CARGO_TARGET_DIR/release/predict_features_with_bake" "$O/"
for b in byc_predict byc_fsid panel score_pair_with_bake; do ln -s /var/tmp/byc-linear-20261104/bin/$b "$O/$b"; done
