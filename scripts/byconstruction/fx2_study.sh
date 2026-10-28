#!/usr/bin/env bash
# Scale-0 chroma (fx2) for the revision-7 models, stage 1: Mlic's recipe with the fx2 declaration.
# Registration: benchmarks/byconstruction_2026-10-22.md, "Scale-0 chroma for the revision-7 models (fx2)".
#
#   fx2_study.sh prepare STUDY   pinned binaries, columns, source pin; then build STUDY/root with
#                                ring_root.py selection-rebind --declaration fx2 (parent a7fd4af0...)
#   fx2_study.sh control STUDY   Mlic's KADID seed-0 cell retrained (fx1); must reproduce the stored cell
#   fx2_study.sh cells STUDY     Mlicfx2's 12 E21 cells (seeds 0-2)
#   fx2_study.sh full STUDY      Mlicfx2's three full-data seeds, packed for the gates
#   fx2_study.sh compare STUDY   E21 scorecard against Mlic's stored cells (requires the passing control):
#                                paired E21 SROCC, per-type bias, knob-concurring inversions, geometry panel,
#                                standard and global-edit mixtures; TRAIN labels only
#   fx2_study.sh live STUDY      the 2026-10-18 trace's 37 exposed LIVE rows (10 severe colour-noise cases and their
#                                same-source controls; no other LIVE row is read), scored by Mlic, the curated
#                                general/codec models and the Mlicfx2 seeds; diagnostic, never fitted or selected on
#
# Cells and full fits may run before or beside the control; no comparison uses them unless CONTROL.json passes.
# The caller holds ~/tmp/heavy.lock and run-heavy (dev: zensim training only). No label outside TRAIN is read.
set -euo pipefail
cmd=$1 study=$2
repo=$(cd "$(dirname "$0")/../.." && pwd)
mixes=/var/tmp/byc-mixes-20261025
root=$mixes/Mlic/root  # fx1 (control); fx2 runs use "$study/root", its selection-rebind
fx1_spec=sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1

env_for() {
  export BYC_ROOT="$1" REV4_V2_BIN_DIR="$study/bin" BYC_CELLS="$study/cells" BYC_SCRATCH="$study/scratch"
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
    env_for "$root"
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
    python3 -B "$repo/scripts/rev4_featpot/byc_cells.py" run --arm mono:thr:mp32 --base-spec "$(fx2_spec)" \
      --cols "$study/frozen_configuration/cols_fx2.txt" --label Mlicfx2 \
      --folds kadid,tid2013,konfig,cid22_a25 --seeds 0,1,2 --jobs 12 ;;
  full)
    test -f "$study/PREPARE_COMPLETE" -a -f "$study/root/wide/frozen.json"
    env_for "$study/root"
    spec="$(fx2_spec):mono:thr:mp32"
    mkdir -p "$study/full/Mlicfx2"
    pids=()
    for seed in 0 1 2; do
      dest="$study/full/Mlicfx2/full_s$seed"
      test ! -e "$dest"
      mkdir -p "$study/scratch/full_s$seed"
      TMPDIR="$study/scratch/full_s$seed" python3 -B "$repo/scripts/rev4_featpot/v2_confirm_fit.py" \
        --root "$study/root" --spec "$spec" --columns "$(cat "$study/frozen_configuration/cols_fx2.txt")" \
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
    ln -sfn "$study/cells/Mlicfx2" "$study/compare/cells/Mlicfx2"
    export BYC_ROOT="$root" BYC_CELLS="$study/compare/cells" REV4_V2_BIN_DIR="$study/bin" AIC_BIN="$study/bin"
    export ZEN_PANEL_BIN="$study/bin/panel" ZENSIM_MAX_TIER=v3 BYC_LABEL_LEDGER_DIR="$out/label_reads"
    export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    tool="$repo/scripts/rev4_featpot/byc_cells.py"
    python3 -B "$tool" score --arms Mlic,Mlicfx2 --paired-with Mlic > "$out/e21.json"
    for arm in Mlic Mlicfx2; do python3 -B "$tool" bytype --arm "$arm" > "$out/bytype_$arm.json"; done
    python3 -B "$repo/scripts/byconstruction/type_bias.py" Mlic Mlicfx2 --out "$out/type_bias.json" > "$out/type_bias.txt"
    python3 -B "$tool" knobinv --arms Mlic,Mlicfx2 --out "$out/knobinv.json" > "$out/knobinv.txt"
    python3 -B "$repo/scripts/byconstruction/aic_panel.py" --out "$out/panel.json" --arms base-c,Mlic,Mlicfx2 \
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
        bash "$repo/scripts/byconstruction/ring_mixture.sh" "$study/bin" raw 12 Mlicfx2
      python3 -B "$repo/scripts/byconstruction/mixture_test.py" summary --out "$mix" --json-out "$out/$set.json" \
        Mlic Mlicfx2 > "$out/$set.txt" 2>&1
    done ;;
  live)
    out="$study/live"
    test ! -e "$out"
    mkdir -p "$out"
    python3 -B - "$study" "$out" <<'PY'
import concurrent.futures as cf, csv, hashlib, json, math, subprocess, sys
from pathlib import Path
study, out = map(Path, sys.argv[1:])
trace_path = Path("/mnt/v/output/zensim/gauntlet-live-color-trace-2026-10-18/TRACE.json")
trace = json.loads(trace_path.read_text())
clean = {'building2','carnivaldolls','cemetry','churchandcapitol','coinsinfountain','dancers','flowersonih35',
         'manfishing','monarch','studentsculpture'}
table = Path("/mnt/v/datasets/LIVE/live_r2_pairs_png.tsv")
rows = [r for r in csv.DictReader(open(table), delimiter="\t") if Path(r["ref_path"]).stem.lower() in clean]
indices = sorted(set(trace["cases"]) | set(trace["controls"]))
fx = "/mnt/v/output/zensim"
bakes = {"Mlic": "/var/tmp/byc-finalists-20261026/Mlic/full_s0/refit/production-f16.bin",
         "general": f"{fx}/mlic-aic-fulldata-xrefaic256e59-20261027-run01/full/XrefAIC256/full_s0/pack_e059/refit/production-f16.bin",
         "codec": f"{fx}/mlic-aic-fulldata-aiconlye29-20261027-run01/full/AICOnly/full_s0/pack_e029/refit/production-f16.bin",
         **{f"Mlicfx2_s{s}": str(study/f"full/Mlicfx2/full_s{s}/refit/production-f16.bin") for s in range(3)}}
def score(bake, r):
    p = subprocess.run([str(study/"bin/score_pair_with_bake"), "--bake", bake, "--bake-post", "raw",
                        "--ref", r["ref_path"], "--dist", r["dist_path"]], capture_output=True, text=True, check=True)
    v = float(p.stdout.strip().splitlines()[-1])
    if not math.isfinite(v):
        raise ValueError("nonfinite score")
    return v
result = dict(trace=str(trace_path), trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              table=str(table), table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
              rows_read=len(indices), bakes={k: dict(path=v, sha256=hashlib.sha256(Path(v).read_bytes()).hexdigest())
                                              for k, v in bakes.items()}, rows=[])
with cf.ThreadPoolExecutor(12) as pool:
    scores = {k: list(pool.map(lambda i: score(v, rows[i]), indices)) for k, v in bakes.items()}
for j, i in enumerate(indices):
    r = rows[i]
    result["rows"].append(dict(index=i, case=i in trace["cases"], ref=r["ref_path"], dist=r["dist_path"],
                               human=float(r["human_score"]), **{k: scores[k][j] for k in bakes}))
# Per source: does the severe case score above any less-damaged control of the same source?
report = {}
for i in trace["cases"]:
    case = next(x for x in result["rows"] if x["index"] == i)
    same = [x for x in result["rows"] if not x["case"] and x["ref"] == case["ref"]]
    report[Path(case["dist"]).parent.name + "/" + Path(case["dist"]).name + " " + Path(case["ref"]).stem] = {
        k: sum(1 for x in same if x["human"] > case["human"] and case[k] > x[k]) for k in bakes}
result["severe_above_less_damaged_controls"] = report
(out/"LIVE.json").write_text(json.dumps(result, indent=1) + "\n")
print("inversions (severe case scored above a less-damaged same-source control):")
for k in bakes:
    print(f"  {k:12s} {sum(v[k] for v in report.values())} over {len(report)} cases")
PY
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
