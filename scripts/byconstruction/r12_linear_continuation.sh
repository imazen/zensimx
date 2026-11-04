#!/usr/bin/env bash
# Conditional L3, admitted by the frozen TRAIN development feasibility decision.
set -euo pipefail
cd "$(dirname "$0")/../.."
X=$PWD
O=/var/tmp/byc-linear-20261104
R=/var/tmp/byc-fchroma-20261029/r3/w100/root
python3 -c 'import json; assert json.load(open("/var/tmp/byc-linear-20261104/dev/DECISION.json"))["L3_run"], "L3 development condition failed"'
export ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export REV4_V2_BIN_DIR=$O/bin REV4_V2_ROOT=$R BYC_ROOT=$R
export ZEN_PANEL_BIN=$O/bin/panel PYTHONPATH=$X/scripts/rev4_featpot:$X/scripts
export BYC_CELLS=/var/tmp/byc-fchroma-20261029/r3/w100/cells
for seed in 0 1 2; do
  name=L3-s$seed
  export TMPDIR=$O/scratch/$name BYC_SCRATCH=$O/scratch/$name
  mkdir -p "$TMPDIR"
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$name" > "$O/STAGE"
  python3 -B scripts/rev4_featpot/r12_linear_fit.py --r12-arm L3 --root "$R" \
    --spec 'sel:45acffa603e8@h32:H1:cv16:cf98:fx2:mono' \
    --columns "$(cat "$HOME/tmp/hc1-profile/cols_fx2.txt")" --head N --seed-index "$seed" \
    --strict-admission --data-role-decision "$R/human_role_decision.json" \
    --dest "$O/fits/$name" --train-only --init-from "$O/fits/L1-s$seed/refit/last.bin" \
    --fine-tune-epochs 60 --fine-tune-lr 0.00001 \
    --aic-public-train /mnt/v/output/zensim/aic-finetune-20261026/folds-rev7/aic_full_fit.parquet \
    --aic-weight 256 --diagnostic-population mixed-aic256-xref \
    --human-sources tid2013 \
    --exclude-datasets pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022 \
    --mos-share 36.035 > "$O/logs/$name.log" 2>&1
  python3 -B - "$O/fits/$name" "$R" > "$O/logs/$name-pack.log" 2>&1 <<'PY'
import json,sys
from pathlib import Path
from v2_production_pack import pack_production
p=Path(sys.argv[1]); dest=p/'pack_e059'; (dest/'refit').mkdir(parents=True,exist_ok=False)
r=pack_production(p/'ckpt/ckpt_epoch059.bin',dest,Path(sys.argv[2])/'wide/main/real/cid22_fit.parquet',
                  e33_output_stage=True,no_prune=True)
(dest/'PACK.json').write_text(json.dumps(r,indent=2)+'\n')
PY
  printf '%s complete %s\n' "$(date -u +%FT%TZ)" "$name" >> "$O/PROGRESS"
done
printf '%s continuation-complete\n' "$(date -u +%FT%TZ)" > "$O/STAGE"
