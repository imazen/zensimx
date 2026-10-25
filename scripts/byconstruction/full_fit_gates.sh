#!/bin/bash
# Full-data fit of one by-construction recipe on every E21 TRAIN leg (seed 0, strict admission, production f16 pack),
# its monotone certificate, then the label-free product gates through E33/E34's owners (gates.py: NEARID, the standard
# and ladder dial verdicts, G-STEER, identity at every tier). Exploration record: benchmarks/byconstruction_2026-10-22.md
# ("Full-data fits"; the gate review). Each step refuses an existing destination.
#
#   full_fit_gates.sh NAME SPEC_TOKENS [GATE_NAME]
#     NAME          fit directory under ~/tmp/byc/full (e.g. mono-thr-mp64_s0)
#     SPEC_TOKENS   recipe tokens after B's base spec (e.g. mono:thr:mp64)
#     GATE_NAME     lowercase name for the gate outputs (default: NAME without punctuation)
# BIN_DIR (trainer-side tools, just byc-train-bins), ROOT (training root) and COLS (column list) override the defaults.
set -euo pipefail
NAME=$1
TOKENS=$2
GATE=${3:-$(echo "$NAME" | tr -cd 'a-z0-9')}
cd "$(dirname "$0")/../.."
ROOT=${ROOT:-/mnt/v/output/zensim/v40r4-2026-10-08/v2e29}
BIN_DIR=${BIN_DIR:?set BIN_DIR (just byc-train-bins)}
COLS=${COLS:-$HOME/tmp/byc/cols_c.txt}
DEST=$HOME/tmp/byc/full/$NAME
GATES=/mnt/v/output/zensim/byconstruction-2026-10-22/gates-$GATE
mkdir -p "$HOME/tmp/byc/scratch/full-$NAME"
# One trainer thread, as the full-data M and T fits ran (run-heavy --jobs 1).
REV4_V2_BIN_DIR=$BIN_DIR ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=$HOME/tmp/byc/scratch/full-$NAME \
  python3 scripts/rev4_featpot/v2_confirm_fit.py --root "$ROOT" \
  --spec "sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1:$TOKENS" --columns "$(cat "$COLS")" --head N --seed-index 0 \
  --strict-admission --data-role-decision "$ROOT/human_role_decision.json" --dest "$DEST" --train-only \
  --pack-production --e33-output-stage
MODEL=$DEST/refit/production-f16.bin
"$BIN_DIR/bake_dial_refit" certify-monotone --bake "$MODEL"
TMPDIR=$HOME/tmp/byc/scratch python3 scripts/byconstruction/gates.py --name "$GATE" --model "$MODEL" --out "$GATES" \
  --steps nearid,verdict,steer,identity
echo "FULL-FIT-GATES-DONE $NAME $(sha256sum "$MODEL" | cut -c1-16)"
