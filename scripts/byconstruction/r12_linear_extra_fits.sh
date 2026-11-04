#!/usr/bin/env bash
# R12 addition, registered d3ec4dab before construction. Caller holds heavy lock.
set -euo pipefail
cd "$(dirname "$0")/../.."
X=$PWD; O=/var/tmp/byc-linear-20261104/extra
R=/var/tmp/byc-fchroma-20261029/r3/w100/root
mkdir -p "$O/logs" "$O/bakes" "$O/scratch"
for s in 0 1 2; do
  A=$HOME/tmp/e36test/r10/runs/mixF3-s$s/pack_e059/refit/production-f16.bin
  "$O/bin/bake_dial_refit" prune-hidden --in "$A" --out "$O/bakes/L0_s$s.bin" > "$O/logs/prune-L0_s$s.json"
  "$O/bin/bake_dial_refit" certify-monotone --bake "$O/bakes/L0_s$s.bin" > "$O/logs/cert-L0_s$s.json"
done
"$O/bin/bake_dial_refit" prune-hidden --in "$HOME/tmp/e36test/r10/runs/mixF3-s0/pack_e059/refit/production-f16.bin" \
  --out "$O/bakes/L0r_s0.bin" --keep-units 126,127 > "$O/logs/prune-L0r_s0.json"
export ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export REV4_V2_BIN_DIR=$O/bin REV4_V2_ROOT=$R BYC_ROOT=$R
export ZEN_PANEL_BIN=$O/bin/panel PYTHONPATH=$X/scripts/rev4_featpot:$X/scripts
export BYC_CELLS=/var/tmp/byc-fchroma-20261029/r3/w100/cells
for seed in 0 1 2; do
  export TMPDIR=$O/scratch/L4-s$seed BYC_SCRATCH=$O/scratch/L4-s$seed
  mkdir -p "$TMPDIR"
  printf '%s L4-s%s\n' "$(date -u +%FT%TZ)" "$seed" > "$O/STAGE"
  python3 -B scripts/rev4_featpot/r12_linear_fit.py --r12-arm L4 --root "$R" \
    --spec 'sel:45acffa603e8@h32:H2:cv16:cf98:fx2:mono:thr:mp1' \
    --columns "$(cat "$HOME/tmp/hc1-profile/cols_fx2.txt")" --head N --seed-index "$seed" \
    --strict-admission --data-role-decision "$R/human_role_decision.json" \
    --dest "$O/fits/L4-s$seed" --train-only --pack-production --e33-output-stage \
    --human-sources tid2013 --exclude-datasets pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022 \
    --mos-share 36.035 > "$O/logs/L4-s$seed.log" 2>&1
  "$O/bin/bake_dial_refit" certify-monotone --bake "$O/fits/L4-s$seed/refit/production-f16.bin" > "$O/logs/cert-L4_s$seed.json"
done
python3 - "$O" <<'PY'
import json,sys,hashlib
from pathlib import Path
r=Path(sys.argv[1]);models={};rows=[]
for arm in ('L0','L4'):
 for s in range(3):
  p=r/f'bakes/L0_s{s}.bin' if arm=='L0' else r/f'fits/L4-s{s}/refit/production-f16.bin'
  models[f'{arm}_s{s}']=str(p)
  rows.append(dict(arm=arm,seed=s,packed_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),packed_bytes=p.stat().st_size,
                   certificate=json.load(open(r/f'logs/cert-{arm}_s{s}.json'))))
assert json.load(open(r/'logs/prune-L0_s0.json'))['old_hidden']-json.load(open(r/'logs/prune-L0_s0.json'))['new_hidden']==95
(r/'MODELS.json').write_text(json.dumps(models,indent=2)+'\n')
(r/'FROZEN_FITS.json').write_text(json.dumps(dict(models=rows),indent=2)+'\n')
PY
printf '%s extra-fits-complete\n' "$(date -u +%FT%TZ)" > "$O/STAGE"
