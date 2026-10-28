#!/usr/bin/env bash
# Scale-0 chroma (fx2) for the revision-7 models, stage 1: Mlic's recipe with the fx2 declaration.
# Registration: benchmarks/byconstruction_2026-10-22.md, "Scale-0 chroma for the revision-7 models (fx2)".
#
#   fx2_study.sh prepare STUDY   pinned binaries, columns, root link, source pin
#   fx2_study.sh control STUDY   Mlic's KADID seed-0 cell retrained (fx1); must reproduce the stored cell
#   fx2_study.sh cells STUDY     Mlicfx2's 12 E21 cells (seeds 0-2)
#   fx2_study.sh full STUDY      Mlicfx2's three full-data seeds, packed for the gates
#
# Cells and full fits may run before or beside the control; no comparison uses them unless CONTROL.json passes.
# The caller holds ~/tmp/heavy.lock and run-heavy (dev: zensim training only). No label outside TRAIN is read.
set -euo pipefail
cmd=$1 study=$2
repo=$(cd "$(dirname "$0")/../.." && pwd)
mixes=/var/tmp/byc-mixes-20261025
root=$mixes/Mlic/root
fx1_spec=sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1

env_for() {
  export BYC_ROOT="$root" REV4_V2_BIN_DIR="$study/bin" BYC_CELLS="$study/cells" BYC_SCRATCH="$study/scratch"
  export ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ZEN_PANEL_BIN="$study/bin/panel"
}

fx2_spec() {
  local id
  id=$(python3 -B -c "import sys; sys.path.insert(0, '$repo/scripts/rev4_featpot')
from v2_common import selection_id
print(selection_id([int(x) for x in open('$study/frozen_configuration/cols_fx2.txt').read().split(',')]))")
  echo "sel:$id@h32:H128:cv16:cf98:fx2"
}

case "$cmd" in
  prepare)
    test ! -e "$study/PREPARE_COMPLETE"
    mkdir -p "$study/bin" "$study/frozen_configuration" "$study/cells" "$study/scratch"
    (cd "$mixes/bin" && sha256sum -c "$mixes/BIN_SHA256SUMS" --quiet)
    cp -p "$mixes"/bin/* "$study/bin/"
    sed "s#$mixes/bin#$study/bin#" "$mixes/BIN_SHA256SUMS" > "$study/BIN_SHA256SUMS"
    (cd "$study/bin" && sha256sum -c "$study/BIN_SHA256SUMS" --quiet)
    python3 -B - "$repo" "$study" "$mixes" <<'PY'
import json, sys
repo, study, mixes = sys.argv[1:]
sys.path.insert(0, f"{repo}/scripts/rev4_featpot")
from v2_common import derived_declaration, selection_id
path, pin, decl = derived_declaration("fx2")
fx1 = [int(x) for x in open(f"{mixes}/frozen_configuration/cols_c.txt").read().strip().split(",")]
cols = sorted(decl["direct"])
if not set(fx1) < set(cols) or len(cols) != 492:
    raise SystemExit("fx2 columns must strictly extend fx1's 410")
open(f"{study}/frozen_configuration/cols_fx2.txt", "w").write(",".join(map(str, cols)) + "\n")
json.dump({"fx2_declaration": str(path), "fx2_declaration_sha256": pin, "fx2_selection": selection_id(cols),
           "fx1_selection": selection_id(fx1), "added": sorted(set(cols) - set(fx1))},
          open(f"{study}/frozen_configuration/SELECTION.json", "w"), indent=1)
PY
    cp -p "$mixes/frozen_configuration/cols_c.txt" "$study/frozen_configuration/cols_c.txt"
    sha256sum "$root/wide/frozen.json" > "$study/ROOT_SHA256"
    grep -q a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952 "$study/ROOT_SHA256"
    git -C "$repo" rev-parse HEAD > "$study/SOURCE_COMMIT.txt"
    git -C "$repo" diff HEAD -- scripts > "$study/SOURCE_CHANGES.patch"
    touch "$study/PREPARE_COMPLETE" ;;
  control)
    test -f "$study/PREPARE_COMPLETE"
    env_for
    python3 -B "$repo/scripts/rev4_featpot/byc_cells.py" run --arm mono:thr:mp32 --base-spec "$fx1_spec" \
      --cols "$study/frozen_configuration/cols_c.txt" --label Mlic --folds kadid --seeds 0 --jobs 1
    python3 -B - "$study" "$mixes" <<'PY'
import json, sys, hashlib
study, mixes = sys.argv[1:]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
new, old = f"{study}/cells/Mlic/without_kadid_s0", f"{mixes}/cells/Mlic/without_kadid_s0"
out = {k: {"new": sha(f"{new}/{f}"), "stored": sha(f"{old}/{f}")}
       for k, f in (("prediction", "pred_kadid.tsv"), ("last_bin", "refit/last.bin"))}
out["pass"] = all(v["new"] == v["stored"] for v in out.values())
json.dump(out, open(f"{study}/CONTROL.json", "w"), indent=1)
print(json.dumps(out))
if not out["pass"]:
    raise SystemExit("control cell does not reproduce the stored Mlic cell: no comparison is made")
PY
    ;;
  cells)
    test -f "$study/PREPARE_COMPLETE"
    env_for
    python3 -B "$repo/scripts/rev4_featpot/byc_cells.py" run --arm mono:thr:mp32 --base-spec "$(fx2_spec)" \
      --cols "$study/frozen_configuration/cols_fx2.txt" --label Mlicfx2 \
      --folds kadid,tid2013,konfig,cid22_a25 --seeds 0,1,2 --jobs 12 ;;
  full)
    test -f "$study/PREPARE_COMPLETE"
    env_for
    spec="$(fx2_spec):mono:thr:mp32"
    mkdir -p "$study/full/Mlicfx2"
    pids=()
    for seed in 0 1 2; do
      dest="$study/full/Mlicfx2/full_s$seed"
      test ! -e "$dest"
      mkdir -p "$study/scratch/full_s$seed"
      TMPDIR="$study/scratch/full_s$seed" python3 -B "$repo/scripts/rev4_featpot/v2_confirm_fit.py" \
        --root "$root" --spec "$spec" --columns "$(cat "$study/frozen_configuration/cols_fx2.txt")" \
        --head N --seed-index "$seed" --strict-admission --data-role-decision "$root/human_role_decision.json" \
        --dest "$dest" --train-only --pack-production --e33-output-stage > "$study/full/full_s$seed.log" 2>&1 &
      pids+=($!)
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    exit "$failed" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
