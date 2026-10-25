fmt-check:
    cargo fmt -p zensim -p zensim-validate --check

# Run through the caller's shared heavy lock and run-heavy wrapper.
# Explicit engineering inputs and an empty corpus root keep human payloads out.
# A1-A8r remain unmeasured; this replay cannot qualify a model for shipping.
[positional-arguments]
bake-verdict-engineering program bake corpus_root grid negtail identity missing_root out rev="5":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$8"
    ZENSIM_FORMULA_REV="$9" RAYON_NUM_THREADS=1 ZENSIM_DIAL_PRED_OUT="$8/predictions.tsv" \
        "$1" --bake "$2" --regime 720 --features-root "$3" --corpora none \
        --dial-grid "$4" --negtail-probe "$5" --identity-probe "$6" \
        --corruption-grid "$7" --perpair-metrics "$7" --require-feature-set-match \
        --gaddr-json "$8/gaddr.json" --fulleval "$8/fulleval.json" --output "$8/verdict.md"
