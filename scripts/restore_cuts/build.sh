#!/usr/bin/env bash
# Clean-snapshot builds of the extractor (CODEX_NOTE crates-on-main rule).
#   build.sh cand   this workspace (must be committed; a dirty tree builds but writes no build_meta.json)
#   build.sh base   main@origin as fetched (the bit-identity baseline for the f0..f1501 gate)
# Sibling repos are `git archive`s of their fetched mains under $ROOT/src (snapshot_revs.tsv); no
# sibling working copy is read or written. Outputs: $ROOT/bin/extract_{base,cand}.
set -euo pipefail
which=${1:?usage: build.sh cand|base}
ROOT=/var/tmp/restore-cuts
WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)  # repo root, derived at run time
export CARGO_HOME=$ROOT/cargo-home CARGO_TARGET_DIR=$ROOT/target-rel
mkdir -p "$ROOT/bin"
overlay() {
    cat >> "$1/zensim-bench/Cargo.toml" <<'OVERLAY'

# restore-cuts local build overlay: clean archives of fetched main commits.
[patch."https://github.com/imazen/zenanalyze"]
zenpredict = { path = "../../zenanalyze/zenpredict" }
zenpredict-bake = { path = "../../zenanalyze/zenpredict-bake" }

[patch."https://github.com/imazen/rav1d-safe"]
rav1d-safe = { path = "../../rav1d-safe" }
rav1d-disjoint-mut = { path = "../../rav1d-safe/crates/rav1d-disjoint-mut" }

[patch."https://github.com/imazen/zenbench"]
zenbench = { path = "../../zenbench" }

[patch."https://github.com/imazen/zenresize"]
zenresize = { path = "../../zenresize" }

[patch."https://github.com/imazen/codec-corpus"]
corruption-corpus = { path = "../../codec-corpus/crate/corruption-corpus" }
OVERLAY
}
build_in() {
    ( cd "$1/zensim-bench"
      cargo build --release -p zensim-bench --example extract_features_372col --features training,zen-decode,verify-all
      sha256sum "$CARGO_TARGET_DIR/release/examples/extract_features_372col" )
}
if [ "$which" = base ]; then
    rev=$(git -C /home/lilith/work/zen/zensimx rev-parse refs/remotes/origin/main)
    B=$ROOT/base
    rm -rf "$B"; mkdir -p "$B/zensim"
    git -C /home/lilith/work/zen/zensimx archive "$rev" | tar -x -C "$B/zensim"
    for d in "$ROOT"/src/*/; do n=$(basename "$d"); [ "$n" = zensim ] || ln -s "$ROOT/src/$n" "$B/$n"; done
    overlay "$B/zensim"
    echo "BASE_COMMIT=$rev"
    build_in "$B/zensim"
    cp "$CARGO_TARGET_DIR/release/examples/extract_features_372col" "$ROOT/bin/extract_base"
    sha256sum "$ROOT/bin/extract_base"
    exit 0
fi
# An extraction binary must come from COMMITTED source.
# (capture first: `jj status | grep -q` under pipefail dies of SIGPIPE when grep exits early)
status=$(cd "$WS" && jj status)
if grep -q 'The working copy has no changes' <<<"$status"; then
    CLEAN=1
else
    CLEAN=0
    rm -f "$ROOT/build_meta.json"
    echo "WARNING: uncommitted changes; DEV BUILD, no build_meta.json will be written"
fi
mkdir -p "$ROOT/src/zensim"
rsync -a --delete --exclude '/target' --exclude '.jj' --exclude '.git' --exclude '.workongoing' \
    "$WS/" "$ROOT/src/zensim/"
overlay "$ROOT/src/zensim"
build_in "$ROOT/src/zensim"
cp "$CARGO_TARGET_DIR/release/examples/extract_features_372col" "$ROOT/bin/extract_cand"
sha256sum "$ROOT/bin/extract_cand"
[ "$CLEAN" = 1 ] || exit 0
cd "$WS"
zc=$(jj log -r @- --no-graph -T 'commit_id' --ignore-working-copy)
base=$(jj log -r 'main@origin' --no-graph -T 'commit_id' --ignore-working-copy)
python3 "$WS/scripts/restore_cuts/write_build_meta.py" "$zc" "$base"
