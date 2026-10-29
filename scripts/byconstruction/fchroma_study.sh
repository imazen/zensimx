#!/usr/bin/env bash
# Profile F (path P1 chroma): Mlic's recipe on the chroma-leg root, under fx1 and under fx2.
# Registration: benchmarks/byconstruction_2026-10-22.md, "Profile F (chroma-aware revision-7.1)".
# Same commands and pins as fx2_study.sh; the root is the stage-C derived root instead of the Mlic root.
#
#   ARM=fx1|fx2 fchroma_study.sh prepare STUDY   pinned binaries, columns; STUDY/root is built beforehand by
#                                ring_root.py chroma (fx1) and its selection-rebind --declaration fx2 (fx2)
#   fchroma_study.sh control STUDY   Mlic's KADID seed-0 cell retrained on the ORIGINAL Mlic root must reproduce the stored cell
#   fchroma_study.sh cells STUDY     E21 cells (12: four folds x seeds 0-2) on STUDY/root
#   fchroma_study.sh full STUDY      three full-data seeds, packed for the gates
#
# The fx2 arm keeps fx2_study.sh's labels (Mlicfx2) so the continuation owners (aic_matched_continuation.py fx2-plan)
# accept the study unchanged; the fx1 arm is labelled Mlicfc1. The caller holds ~/tmp/heavy.lock and run-heavy.
set -euo pipefail
cmd=$1 study=$2
ARM=${ARM:-fx2}
repo=$(cd "$(dirname "$0")/../.." && pwd)
mixes=/var/tmp/byc-mixes-20261025
orig_root=$mixes/Mlic/root  # the control reproduces the stored Mlic cell on this root
root=$study/root            # the derived chroma root of this arm
fx1_spec=sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1
if [ "$ARM" = fx2 ]; then label=Mlicfx2; else label=Mlicfc1; fi

env_for() {
  export BYC_ROOT="$1" REV4_V2_BIN_DIR="$study/bin" BYC_CELLS="$study/cells" BYC_SCRATCH="$study/scratch"
  export ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ZEN_PANEL_BIN="$study/bin/panel"
}

arm_spec() {
  if [ "$ARM" = fx1 ]; then echo "$fx1_spec"; else fx2_spec; fi
}
arm_cols() {
  if [ "$ARM" = fx1 ]; then echo "$study/frozen_configuration/cols_c.txt"; else echo "$study/frozen_configuration/cols_fx2.txt"; fi
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
    sha256sum "$orig_root/wide/frozen.json" > "$study/ROOT_SHA256"
    grep -q a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952 "$study/ROOT_SHA256"
    sha256sum "$root/wide/frozen.json" > "$study/CHROMA_ROOT_SHA256"
    jj --repository "$repo" log -r @- --no-graph -T commit_id > "$study/SOURCE_COMMIT.txt"  # jj workspaces have no .git; @- is the committed source
    jj --repository "$repo" diff --git -- scripts > "$study/SOURCE_CHANGES.patch"
    touch "$study/PREPARE_COMPLETE" ;;
  control)
    test -f "$study/PREPARE_COMPLETE"
    env_for "$orig_root"
    python3 -B "$repo/scripts/rev4_featpot/byc_cells.py" run --arm mono:thr:mp32 --base-spec "$fx1_spec" \
      --cols "$study/frozen_configuration/cols_c.txt" --label Mlic --folds kadid --seeds 0 --jobs 1
    # The bake embeds run provenance (zentrain.repro: paths, run time), so the model bytes are compared
    # with that one entry removed by the pinned bake owner (the E33 control's rule).
    for side in new stored; do
      if [ $side = new ]; then cell="$study/cells/Mlic/without_kadid_s0"; else cell="$mixes/cells/Mlic/without_kadid_s0"; fi
      "$study/bin/bake_dial_refit" strip --in "$cell/refit/last.bin" --out "$study/control_$side.norepro.bin" \
        --key zentrain.repro > /dev/null
    done
    python3 -B - "$study" "$mixes" <<'PY'
import json, sys, hashlib
study, mixes = sys.argv[1:]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
new, old = f"{study}/cells/Mlic/without_kadid_s0", f"{mixes}/cells/Mlic/without_kadid_s0"
out = {"prediction": {"new": sha(f"{new}/pred_kadid.tsv"), "stored": sha(f"{old}/pred_kadid.tsv")},
       "last_bin_without_repro": {"new": sha(f"{study}/control_new.norepro.bin"),
                                  "stored": sha(f"{study}/control_stored.norepro.bin")},
       "last_bin_raw": {"new": sha(f"{new}/refit/last.bin"), "stored": sha(f"{old}/refit/last.bin")}}
out["pass"] = all(out[k]["new"] == out[k]["stored"] for k in ("prediction", "last_bin_without_repro"))
json.dump(out, open(f"{study}/CONTROL.json", "w"), indent=1)
print(json.dumps(out))
if not out["pass"]:
    raise SystemExit("control cell does not reproduce the stored Mlic cell: no comparison is made")
PY
    ;;
  cells)
    test -f "$study/PREPARE_COMPLETE" -a -f "$study/root/wide/frozen.json"
    env_for "$study/root"
    python3 -B "$repo/scripts/rev4_featpot/byc_cells.py" run --arm mono:thr:mp32 --base-spec "$(arm_spec)" \
      --cols "$(arm_cols)" --label "$label" \
      --folds kadid,tid2013,konfig,cid22_a25 --seeds 0,1,2 --jobs 12 ;;
  full)
    test -f "$study/PREPARE_COMPLETE" -a -f "$study/root/wide/frozen.json"
    env_for "$study/root"
    spec="$(arm_spec):mono:thr:mp32"
    mkdir -p "$study/full/$label"
    pids=()
    for seed in 0 1 2; do
      dest="$study/full/$label/full_s$seed"
      test ! -e "$dest"
      mkdir -p "$study/scratch/full_s$seed"
      TMPDIR="$study/scratch/full_s$seed" python3 -B "$repo/scripts/rev4_featpot/v2_confirm_fit.py" \
        --root "$study/root" --spec "$spec" --columns "$(cat "$(arm_cols)")" \
        --head N --seed-index "$seed" --strict-admission --data-role-decision "$study/root/human_role_decision.json" \
        --dest "$dest" --train-only --pack-production --e33-output-stage > "$study/full/full_s$seed.log" 2>&1 &
      pids+=($!)
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    exit "$failed" ;;
  compare)
    python3 -c "import json,sys; sys.exit(0 if json.load(open('$study/CONTROL.json'))['pass'] else 1)"
    out="$study/scorecard"
    test ! -e "$out"
    mkdir -p "$out/label_reads" "$study/compare/cells"
    ln -sfn "$mixes/cells/Mlic" "$study/compare/cells/Mlic"
    ln -sfn "$study/cells/$label" "$study/compare/cells/$label"
    export BYC_ROOT="$orig_root" BYC_CELLS="$study/compare/cells" REV4_V2_BIN_DIR="$study/bin" AIC_BIN="$study/bin"
    export ZEN_PANEL_BIN="$study/bin/panel" ZENSIM_MAX_TIER=v3 BYC_LABEL_LEDGER_DIR="$out/label_reads"
    export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    tool="$repo/scripts/rev4_featpot/byc_cells.py"
    python3 -B "$tool" score --arms Mlic,$label --paired-with Mlic > "$out/e21.json"
    for arm in Mlic $label; do python3 -B "$tool" bytype --arm "$arm" > "$out/bytype_$arm.json"; done
    python3 -B "$repo/scripts/byconstruction/type_bias.py" Mlic $label --out "$out/type_bias.json" > "$out/type_bias.txt"
    python3 -B "$tool" knobinv --arms Mlic,$label --out "$out/knobinv.json" > "$out/knobinv.txt"
    python3 -B "$repo/scripts/byconstruction/aic_panel.py" --out "$out/panel.json" --arms base-c,Mlic,$label \
      --geo --no-aic --no-public-test > "$out/panel.txt" 2>&1
    for set in mixture mixture_global; do
      mix="$study/$set"
      mkdir -p "$mix"
      cp -n "$mixes/$set/manifest.json" "$mix/manifest.json"
      cmp "$mixes/$set/manifest.json" "$mix/manifest.json"
      ln -sfn "$(readlink -f "$mixes/$set/img")" "$mix/img"
      for fold in kadid tid2013 konfig cid22_a25; do
        for seed in 0 1 2; do cp -n "$mixes/$set/scores_Mlic_${fold}_s$seed.json" "$mix/"; done
      done
      MIX="$mix" BYC_MIX_FOLDS=kadid,tid2013,konfig,cid22_a25 ZENSIM_PYRAMID=xyb \
        bash "$repo/scripts/byconstruction/ring_mixture.sh" "$study/bin" raw 12 $label
      python3 -B "$repo/scripts/byconstruction/mixture_test.py" summary --out "$mix" --json-out "$out/$set.json" \
        Mlic $label > "$out/$set.txt" 2>&1
    done ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
