#!/usr/bin/env bash
# Equivalence gate for the cross-codec driver consolidation (issue #41 Tier-1 #3).
#
# The nine run_cross_codec_v<N>_seed.sh scripts are now shims over one recipe
# (run_cross_codec_seed.sh + cross_codec_variants/<N>.conf). This test proves the
# consolidation did not change what any of them runs: each shim is rendered in
# CC_DRY_RUN=1 mode and diffed against cross_codec_variants/golden/<N>.args —
# the argv captured from the PRE-consolidation scripts at commit e9a705c0.
#
# Needs no data, no trainer, and no network. Runs anywhere bash does.
#
#   bash scripts/v_next/tests/test_cross_codec_seed_argv.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VNEXT="$(cd "${HERE}/.." && pwd)"
GOLDEN="${VNEXT}/cross_codec_variants/golden"

# The sample args the goldens were captured with. Keep in lockstep with
# cross_codec_variants/README.md.
run_variant() {
    case "$1" in
        v2 | v3 | v4) echo "7 0.5" ;;
        v4b) echo "7 0.05" ;;
        v6) echo "7 1.0 0.30" ;;
        v5 | v7 | v8 | v9 | v9cons | v9mono) echo "7" ;;
        *)
            echo "no sample args for variant $1" >&2
            exit 2
            ;;
    esac
}

fails=0
checked=0

for variant in v2 v3 v4 v4b v5 v6 v7 v8 v9 v9cons v9mono; do
    case "${variant}" in
        v9cons) shim="${VNEXT}/run_cross_codec_v9_conservative.sh" ;;
        v9mono) shim="${VNEXT}/run_cross_codec_v9_mono_recovery.sh" ;;
        *) shim="${VNEXT}/run_cross_codec_${variant}_seed.sh" ;;
    esac
    gold="${GOLDEN}/${variant}.args"

    if [ ! -x "${shim}" ]; then
        echo "FAIL ${variant}: ${shim} missing or not executable"
        fails=$((fails + 1))
        continue
    fi
    if [ ! -f "${gold}" ]; then
        echo "FAIL ${variant}: golden ${gold} missing"
        fails=$((fails + 1))
        continue
    fi

    # shellcheck disable=SC2046  # word splitting of the sample args is intended
    actual="$(CC_DRY_RUN=1 CC_ROOT=/mnt/v \
        CC_TRAINER=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train \
        "${shim}" $(run_variant "${variant}") 2>&1)" || {
        echo "FAIL ${variant}: dry run exited nonzero"
        printf '%s\n' "${actual}"
        fails=$((fails + 1))
        continue
    }

    if diff -u "${gold}" <(printf '%s\n' "${actual}") > /dev/null; then
        checked=$((checked + 1))
    else
        echo "FAIL ${variant}: trainer argv differs from the pre-consolidation golden"
        diff -u "${gold}" <(printf '%s\n' "${actual}") || true
        fails=$((fails + 1))
    fi
done

# The driver must reject an unknown variant rather than silently training with
# defaults — a typo'd variant that ran would burn GPU hours on the wrong recipe.
if "${VNEXT}/run_cross_codec_seed.sh" nosuchvariant 1 > /dev/null 2>&1; then
    echo "FAIL: unknown variant was accepted"
    fails=$((fails + 1))
else
    checked=$((checked + 1))
fi

# A missing required positional must fail too (v6 needs three).
if CC_DRY_RUN=1 "${VNEXT}/run_cross_codec_v6_seed.sh" 7 > /dev/null 2>&1; then
    echo "FAIL: v6 accepted a missing anchor_w/anchor_p"
    fails=$((fails + 1))
else
    checked=$((checked + 1))
fi

if [ "${fails}" -ne 0 ]; then
    echo "cross-codec argv gate: ${fails} FAILED, ${checked} passed"
    exit 1
fi
echo "cross-codec argv gate: ${checked} checks passed (11 variants byte-identical to pre-consolidation argv)"
