#!/usr/bin/env bash
# Run the complete registered base-fit matrix under one externally held heavy lock.
set -euo pipefail
cd "$(dirname "$0")/../.."
X=$PWD
R=/var/tmp/byc-fchroma-20261029/r3/w100/root
O=/var/tmp/byc-linear-20261104
export ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export REV4_V2_BIN_DIR=$O/bin REV4_V2_ROOT=$R BYC_ROOT=$R
export ZEN_PANEL_BIN=$REV4_V2_BIN_DIR/panel
export PYTHONPATH=$X/scripts/rev4_featpot:$X/scripts
export BYC_CELLS=/var/tmp/byc-fchroma-20261029/r3/w100/cells
mkdir -p "$O/logs" "$O/scratch"
for arm in L1 L2; do
  for seed in 0 1 2; do
    name=${arm}-s${seed}
    export TMPDIR=$O/scratch/$name BYC_SCRATCH=$O/scratch/$name
    mkdir -p "$TMPDIR"
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$name" > "$O/STAGE"
    extra=()
    if [[ $arm == L2 ]]; then extra=(--r12-mask "$X/scripts/byconstruction/masks/r12_top8.tsv"); fi
    python3 -B scripts/rev4_featpot/r12_linear_fit.py --r12-arm "$arm" --root "$R" \
      --spec 'sel:45acffa603e8@h32:H1:cv16:cf98:fx2:mono' \
      --columns "$(cat "$HOME/tmp/hc1-profile/cols_fx2.txt")" --head N --seed-index "$seed" \
      --strict-admission --data-role-decision "$R/human_role_decision.json" \
      --dest "$O/fits/$name" --train-only --pack-production --e33-output-stage \
      --human-sources tid2013 \
      --exclude-datasets pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022 \
      --mos-share 36.035 "${extra[@]}" > "$O/logs/$name.log" 2>&1
    printf '%s complete %s\n' "$(date -u +%FT%TZ)" "$name" >> "$O/PROGRESS"
  done
done
printf '%s base-fits-complete\n' "$(date -u +%FT%TZ)" > "$O/STAGE"
