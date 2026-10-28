#!/usr/bin/env bash
# E5A render-regression lane — end-to-end driver.
#
#   e5a_pipeline.sh gen      fixture twins + benign for all 12 origins
#   e5a_pipeline.sh pairs    pairs.tsv from per-origin manifests
#   e5a_pipeline.sh score    all metric arms (e5a_render_score, peer, dssim, tuner)
#   e5a_pipeline.sh analyze  thresholds/testlin/detection/srocc/loc/bootstrap
#   e5a_pipeline.sh all      gen -> pairs -> score -> analyze
#
# Heavy steps are run by the caller under ~/tmp/devin/heavy (the lock).
set -euo pipefail

LANE=${LANE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TARGET=${CARGO_TARGET_DIR:-/var/tmp/e5a-render/target}
WORK=/var/tmp/e5a-render
SRC=/mnt/v/output/imazen-26-variants/cleanpicker-ladder11@2026-08-23
FIXTURES=$WORK/fixtures
CAL=/var/tmp/zensim-validation-2026-09-15/recovery/calibrated
ZMB=/var/tmp/cvvdp-safesyn/target-zenmetrics/release/zenmetrics
PANEL=/home/lilith/work/zen/zensimx/target/release/panel
TUNER=/home/lilith/work/zen/zensimx/target/release/score_pairs_tuner

M3=$TARGET/release/examples/m3_fixture_gen
SCORER=$TARGET/release/examples/e5a_render_score
PEER=$TARGET/release/examples/peer_metric_pairs

declare -A R256 R512
R256[2010]=o_2010.png.scale205x256.png;  R512[2010]=o_2010.png.scale410x512.png
R256[1054]=o_1054.png.scale192x256.png;  R512[1054]=o_1054.png.scale384x512.png
R256[1214]=o_1214.png.scale216x256.png;  R512[1214]=o_1214.png.scale432x512.png
R256[6068]=o_6068.png.scale196x256.png;  R512[6068]=o_6068.png.scale393x512.png
R256[6610]=o_6610.png.scale256x202.png;  R512[6610]=o_6610.png.scale512x403.png
R256[6064]=o_6064.png.scale196x256.png;  R512[6064]=o_6064.png.scale393x512.png
R256[7066]=o_7066.png.scale256x256.png;  R512[7066]=o_7066.png.scale512x512.png
R256[9380]=o_9380.png.scale171x256.png;  R512[9380]=o_9380.png.scale341x512.png
R256[9066]=o_9066.png.scale171x256.png;  R512[9066]=o_9066.png.scale341x512.png
R256[8206]=o_8206.png.scale256x144.png;  R512[8206]=o_8206.png.scale512x288.png
R256[8384]=o_8384.png.scale192x256.png;  R512[8384]=o_8384.png.scale384x512.png
R256[8462]=o_8462.png.scale256x160.png;  R512[8462]=o_8462.png.scale512x320.png
ORIGINS="2010 1054 1214 6068 6610 6064 7066 9380 9066 8206 8384 8462"

cmd=${1:-all}

gen() {
    mkdir -p "$FIXTURES"
    for o in $ORIGINS; do
        "$M3" corruption render \
            --in256 "$SRC/${R256[$o]}" --in512 "$SRC/${R512[$o]}" \
            --out "$FIXTURES/o_$o" --ref-id "o_$o" --seed 1 --set both
    done
}

pairs() {
    python3 "$LANE/scripts/e5a_pairs.py" "$FIXTURES" "$WORK/pairs.tsv"
}

score() {
    mkdir -p "$WORK/maps" "$WORK/maps_peer"
    "$SCORER" --pairs "$WORK/pairs.tsv" --output "$WORK/e5a_scores.tsv" \
        --maps "$WORK/maps" --threads "${E5A_THREADS:-8}"
    "$PEER" --pairs "$WORK/pairs.tsv" --output "$WORK/peer.tsv" \
        --diffmaps "$WORK/maps_peer" --threads "${E5A_THREADS:-8}"
    "$ZMB" batch --metric dssim --pairs "$WORK/pairs.tsv" \
        --output "$WORK/dssim.tsv"
    "$TUNER" --pairs "$WORK/pairs.tsv" --output "$WORK/tuner.parquet" \
        --profile d \
        --ensemble "r915_fast=$CAL/R915_y60_h32_s17101.bin,$CAL/R915_y60_h32_s17103.bin,$CAL/R915_y60_h32_s17107.bin,$CAL/R915_y60_h32_s17111.bin,$CAL/R915_y60_h32_s17113.bin" \
        --ensemble "r915_rich=$CAL/R915_basic228_h128_s17101.bin,$CAL/R915_basic228_h128_s17103.bin,$CAL/R915_basic228_h128_s17107.bin,$CAL/R915_basic228_h128_s17111.bin,$CAL/R915_basic228_h128_s17113.bin" \
        --threads "${E5A_THREADS:-8}"
}

analyze() {
    python3 "$LANE/scripts/e5a_analyze.py" \
        --scores "$WORK/e5a_scores.tsv" --peer "$WORK/peer.tsv" \
        --dssim "$WORK/dssim.tsv" --tuner "$WORK/tuner.parquet" \
        --maps "$WORK/maps" --peer-maps "$WORK/maps_peer" \
        --panel "$PANEL" --out "$WORK/results.json"
}

case "$cmd" in
    gen) gen ;;
    pairs) pairs ;;
    score) score ;;
    analyze) analyze ;;
    all) gen; pairs; score; analyze ;;
    *) echo "usage: $0 {gen|pairs|score|analyze|all}" >&2; exit 2 ;;
esac
