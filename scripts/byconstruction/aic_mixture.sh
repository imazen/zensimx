#!/bin/bash
# Label-free damage-mixture test (table route) for every cell of the given research arms (fx1g / fx1gs1 bakes read
# the byc_globaff descriptor, so they score from the hpm mixture table through byc_predict; fx1o bakes read the
# byc_orient damage, from a table built with `mixture_table.py build --orient`). Exploration record:
# benchmarks/byconstruction_2026-10-22.md, section "AIC2026 TRAIN". Usage: aic_mixture.sh BIN_DIR ARM [ARM ...]
# MIX and TABLE select another mixture set and its table (default: the standard set and its hpm table).
set -euo pipefail
BIN=$1; shift
H=$(cd "$(dirname "$0")" && pwd)
MIX=${MIX:-/mnt/v/output/zensim/byconstruction-2026-10-22/mixture}
TABLE=${TABLE:-/mnt/v/output/zensim/byconstruction-2026-10-22/mixture_table_hpm}
for arm in "$@"; do
  for d in "$HOME"/tmp/byc/cells/"$arm"/without_*_s*; do
    cell=$(basename "$d"); cell=${cell#without_}
    spec=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['spec'])" "$d/byc_record.json")
    decl=$H/../rev4_featpot/byc_fx1g_declaration.json
    case ":$spec:" in *:fx1gs1:*) decl=$H/../rev4_featpot/byc_fx1gs1_declaration.json ;;
      *:fx1gg:*) decl=$H/../rev4_featpot/byc_fx1gg_declaration.json ;;
      *:fx1o:*) decl=$H/../rev4_featpot/byc_fx1o_declaration.json ;; esac
    python3 "$H/mixture_table.py" score --mixture "$MIX" --table "$TABLE" --bin-dir "$BIN" \
      --bake "${arm}_${cell}=$d/refit/last.bin" --decl "$decl"
  done
done
