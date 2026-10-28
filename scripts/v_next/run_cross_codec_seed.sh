#!/usr/bin/env bash
# ONE parameterized cross-codec seed-training recipe (issue #41, Tier-1 item 3).
#
# Before 2026-08-29 this recipe existed eleven times, as
# run_cross_codec_v{2,3,4,4b,5,6,7,8,9}_seed.sh plus the two v9 follow-ups
# run_cross_codec_v9_{conservative,mono_recovery}.sh. Those bodies shared ~55-70%
# of their lines (measured in benchmarks/sweep_training_script_dedup_2026-05-26.md);
# every one of them repeated the whole trainer invocation so that changing a
# shared flag meant editing nine files, and the ones you miss keep running
# stale args and produce mislabeled bakes.
#
# Now: this driver holds the shared body, and each experiment is a config file
# in cross_codec_variants/<variant>.conf that sets ONLY the knobs it changes
# (plus its historical rationale header). The eleven original entry points remain
# as thin shims that exec this driver, so every command line quoted in
# benchmarks/*.md still works unchanged. The retired V9 full pipeline is preserved in git. That
# thin-wrapper shape is the same one _picker_lib.py already proved in-repo.
#
# EQUIVALENCE IS GATED, NOT ASSUMED. scripts/v_next/tests/test_cross_codec_seed_argv.sh
# renders every variant's trainer argv in dry-run mode and diffs it against
# cross_codec_variants/golden/<variant>.args — goldens captured by executing the
# eleven PRE-consolidation scripts against a stub trainer (the nine seed drivers
# at e9a705c0, the two v9 follow-ups at 5f17a99e).
#
#   Usage:  run_cross_codec_seed.sh <variant> <seed> [variant-specific args...]
#           run_cross_codec_v6_seed.sh <seed> <anchor_w> <anchor_p>   # via shim
#
#   Env:
#     KBATCH       minibatch size          (per-variant default; v9 = 32, rest 1)
#     LR_OVERRIDE  learning rate           (per-variant default; v9 = 5.66e-3, rest 1e-3)
#     CC_ROOT      data root               (default /mnt/v)
#     CC_TRAINER   trainer binary          (default the release build in this checkout's tree)
#     CC_DRY_RUN=1 print the trainer argv, one token per line, and run nothing
#
#   Two deliberate (documented) behavior supersets vs the pre-consolidation
#   scripts, neither reachable without opting in:
#     * KBATCH / LR_OVERRIDE are now honored for every variant. v2/v3/v4/v4b
#       hardcoded --minibatch-size 1 and --lr 1e-3 and the v9 family hardcoded
#       32 and 5.66e-3; unset env reproduces all of those exactly (the golden
#       gate proves it), so only an explicit override differs.
#     * v9 and its two follow-ups now print the same trailing "DONE ..." line
#       the other eight already printed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${1:-}"
if [ -z "${VARIANT}" ]; then
    echo "usage: $0 <variant> <seed> [variant args...]" >&2
    echo "variants: $(cd "${HERE}/cross_codec_variants" && ls ./*.conf | sed 's#^\./##;s#\.conf$##' | paste -sd' ' -)" >&2
    exit 2
fi
shift

CONF="${HERE}/cross_codec_variants/${VARIANT}.conf"
if [ ! -f "${CONF}" ]; then
    echo "unknown cross-codec variant '${VARIANT}' (no ${CONF})" >&2
    exit 2
fi

CC_ROOT="${CC_ROOT:-/mnt/v}"
CC_TRAINER="${CC_TRAINER:-/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train}"
CC_DRY_RUN="${CC_DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Shared recipe: the common body of all nine pre-consolidation drivers.
# A variant .conf overrides only what it changes.
# ---------------------------------------------------------------------------
PARQ_DIR="${CC_ROOT}/zen/zensim-training/canonical-2026-05-18/train"
EQUIV="${CC_ROOT}/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

EXP_LABEL=""                       # conf must set
OUT_SUBDIR=""                      # conf must set (under ${CC_ROOT}/zen/zensim-eval/)
ANCHOR_REL=""                      # conf must set (under ${CC_ROOT}/zen/zensim-training/)

HIDDEN=128
EPOCHS=300
PAIRS_PER_EPOCH=50000
LR_DEFAULT=1e-3
L2=1e-5
LEAKY_ALPHA=0.01
VAL_POLICY=min
EARLY_STOP_PATIENCE=0
MAX_FEATURES=372
KBATCH_DEFAULT=1
TARGET_COLUMN=mix_cv40_iw60
TARGET_SCALE=1.0
OUT_DTYPE=f32
TANH_OUTPUT_HEAD_SCALE=""          # flag emitted only when non-empty
RANKNET_WEIGHT=0.0
MSE_WEIGHT=1.0
MONOTONICITY_REG=1.0
MONOTONICITY_MARGIN=0.0
ANCHOR_LOSS_WEIGHT=1.0
ANCHOR_TARGET_SCORE=63.0
ANCHOR_STEP_P=0.10
CC_EQ_WEIGHT=1.0
CC_EQ_STEP_P=0.10
CC_RANK_PRESERVE_WEIGHT=""         # flag emitted only when non-empty
DYN_RANGE_FLOOR_WEIGHT=""          # the four dynamic-range flags emit as one group
DYN_RANGE_SIGMA_THRESHOLD=15.0
DYN_RANGE_STEP_P=0.05
DYN_RANGE_PROBE_N=40

CC_ARG_SUMMARY=""
CC_ENTRY="${CC_ENTRY:-$0}"

# The conf defines cc_parse_args (maps positional args onto knobs, sets SEED and
# CC_ARG_SUMMARY) and cc_bake_stem (prints the bake filename stem).
# shellcheck source=/dev/null
. "${CONF}"

cc_parse_args "$@"

ANCHOR="${CC_ROOT}/zen/zensim-training/${ANCHOR_REL}"
OUT_DIR="${CC_ROOT}/zen/zensim-eval/${OUT_SUBDIR}"
KBATCH="${KBATCH:-${KBATCH_DEFAULT}}"
LR="${LR_OVERRIDE:-${LR_DEFAULT}}"

STEM="$(cc_bake_stem)"
BAKE="${OUT_DIR}/${STEM}.bin"
LOG="${OUT_DIR}/${STEM}.log"
STDOUT="${OUT_DIR}/${STEM}.stdout"

ARGS=(
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0"
    --hidden "${HIDDEN}" --epochs "${EPOCHS}" --pairs-per-epoch "${PAIRS_PER_EPOCH}"
    --lr "${LR}" --l2 "${L2}"
    --leaky-alpha "${LEAKY_ALPHA}" --val-policy "${VAL_POLICY}"
    --early-stop-patience "${EARLY_STOP_PATIENCE}"
    --max-features "${MAX_FEATURES}" --minibatch-size "${KBATCH}"
    --target-column "${TARGET_COLUMN}" --target-scale "${TARGET_SCALE}" --out-dtype "${OUT_DTYPE}"
    --per-sample-alpha-head
)
if [ -n "${TANH_OUTPUT_HEAD_SCALE}" ]; then
    ARGS+=(--tanh-output-head-scale "${TANH_OUTPUT_HEAD_SCALE}")
fi
ARGS+=(
    --ranknet-weight "${RANKNET_WEIGHT}"
    --mse-weight "${MSE_WEIGHT}"
    --monotonicity-reg "${MONOTONICITY_REG}"
    --monotonicity-margin "${MONOTONICITY_MARGIN}"
    --anchor-parquet "${ANCHOR}"
    --anchor-loss-weight "${ANCHOR_LOSS_WEIGHT}"
    --anchor-target-score "${ANCHOR_TARGET_SCORE}"
    --anchor-step-p "${ANCHOR_STEP_P}"
    --cross-codec-eq-parquet "${EQUIV}"
    --cross-codec-eq-weight "${CC_EQ_WEIGHT}"
    --cross-codec-eq-step-p "${CC_EQ_STEP_P}"
)
if [ -n "${CC_RANK_PRESERVE_WEIGHT}" ]; then
    ARGS+=(--cross-codec-rank-preserve-weight "${CC_RANK_PRESERVE_WEIGHT}")
fi
if [ -n "${DYN_RANGE_FLOOR_WEIGHT}" ]; then
    ARGS+=(
        --dynamic-range-floor-weight "${DYN_RANGE_FLOOR_WEIGHT}"
        --dynamic-range-sigma-threshold "${DYN_RANGE_SIGMA_THRESHOLD}"
        --dynamic-range-step-p "${DYN_RANGE_STEP_P}"
        --dynamic-range-probe-n "${DYN_RANGE_PROBE_N}"
    )
fi
ARGS+=(--seed "${SEED}" --out "${BAKE}" --log-path "${LOG}")

if [ "${CC_DRY_RUN}" = "1" ]; then
    printf '%s\n' "${ARGS[@]}"
    exit 0
fi

mkdir -p "${OUT_DIR}"

echo "${EXP_LABEL}: ${CC_ARG_SUMMARY}"
echo "  trainer:  ${CC_TRAINER}"
echo "  anchor:   ${ANCHOR}"
echo "  out_bake: ${BAKE}"

"${CC_TRAINER}" "${ARGS[@]}" 2>&1 | tee "${STDOUT}"

echo "DONE ${CC_ARG_SUMMARY} bake=${BAKE} log=${LOG}"
