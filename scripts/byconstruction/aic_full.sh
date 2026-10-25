#!/bin/bash
# Full-data research fit of one arm (every E21 TRAIN leg plus all of AIC2026 public_train, research replay), then the
# frozen candidate's reads: AIC2026 public_test (ONE read per frozen candidate; the split informed this study's design
# through the 42-source phase) and public_train in sample (the memorization gap against the arm's out-of-fold score).
# Exploration record: benchmarks/byconstruction_2026-10-22.md, section "AIC2026 TRAIN".
# Usage: aic_full.sh GLOBAFF_ROOT BIN_DIR LABEL TOKENS AIC_WEIGHT [BASE]   (BASE=s1 for fx1gs1)
set -euo pipefail
GROOT=$1; BIN=$2; LABEL=$3; TOKENS=$4; W=$5; BASE=${6:-}
H=$(cd "$(dirname "$0")" && pwd)
AIC=/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/folds_train
OUT=/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/full/$LABEL
G=sel:0e6f70c166e6@h32:H128:cv16:cf98:fx1g; GC=$HOME/tmp/byc/cols_fx1g.txt; DECL=$H/../rev4_featpot/byc_fx1g_declaration.json
if [ "$BASE" = s1 ]; then
  G=sel:d6e48dbce419@h32:H128:cv16:cf98:fx1gs1; GC=$HOME/tmp/byc/cols_fx1gs1.txt; DECL=$H/../rev4_featpot/byc_fx1gs1_declaration.json
fi
SPEC=$G${TOKENS:+:$TOKENS}
[ -e "$OUT" ] && { echo "$OUT exists; move it aside" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
export REV4_V2_BIN_DIR=$BIN RAYON_NUM_THREADS=1 ZENSIM_MAX_TIER=v3
TMPDIR=$HOME/tmp/byc/scratch/full-$LABEL; mkdir -p "$TMPDIR"; export TMPDIR
python3 "$H/../rev4_featpot/v2_confirm_fit.py" --root "$GROOT" --spec "$SPEC" --columns "$(cat "$GC")" --head N \
  --seed-index 0 --strict-admission --data-role-decision "$GROOT/human_role_decision.json" --dest "$OUT" --train-only \
  --research-extra-group "aic:$AIC/aic_train_all.parquet:$W:0:rank" \
  --research-replay "byconstruction AIC2026 TRAIN research fit (owner goal 2026-10-23); full public_train; never ship"
for set in public_test train_all; do
  t=$AIC/aic_$set.parquet
  "$BIN/byc_predict" --bake "$OUT/refit/last.bin" --decl "$DECL" --corpus "$t" --out "$OUT/pred_$set.tsv"
done
python3 - "$OUT" "$AIC" "$H" <<'PY'
import json, sys, pandas as pd
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from aic_train import panel
out, aic = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for name, pairs in (("public_test", "pairs_test.tsv"), ("train_all", "pairs_train.tsv")):
    p = pd.read_csv(aic / pairs, sep="\t")
    pred = pd.read_csv(out / f"pred_{name}.tsv", sep="\t").pred.to_numpy()
    res[name] = panel(pred, p)
    print(name, "global", round(res[name]["global"], 4), "within-source", round(res[name]["within_ref_mean"], 4))
(out / "aic_reads.json").write_text(json.dumps(res, indent=1) + "\n")
PY
