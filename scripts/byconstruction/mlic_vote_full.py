"""Three fixed-seed full fits after a passing, frozen MlicVote TRAIN screen.

Uses the screened study's frozen Python owners and binary. This creates model
artifacts for qualification; it neither reads EVAL nor promotes a serving model.
Caller holds heavy.lock and run-heavy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts/rev4_featpot"))
from byc_cells import neff_decision
from run_processes import compute_lock, process_snapshot
from v2_common import sha, confirm_seeds

SPEC="sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1:mono:thr:mp32"
FOLDS=("kadid","tid2013","konfig","cid22_a25","pipal","pairs","mos7")


def admitted_selection(selection):
    if (selection.get("schema")!="mlic-native-vote-TRAIN-selection-v1"
            or selection.get("rule_commit")!="b2ee920aa351" or selection.get("eval_labels_used") is not False):
        raise ValueError("registered native-vote TRAIN screen required")
    counts=[selection["mixtures"]["mixture_global"][f"MlicVote_{fold}_s{seed}"]["inversions"]
            for fold in FOLDS for seed in range(3)]
    decision=neff_decision(selection["arms"],counts,"MlicVote")
    if any(selection.get(k)!=v for k,v in decision.items()):
        raise ValueError("stored native-vote decision differs from registered rule")
    if decision["finalist"]!="MlicVote":
        raise ValueError("native-vote arm did not pass; retain Mlic and continue independent trials")


def full_command(source,root,dest,columns,seed):
    if seed not in (0,1,2):raise ValueError("registered full-fit seeds are 0,1,2")
    return [sys.executable,"-B",str(source/"scripts/rev4_featpot/v2_confirm_fit.py"),
        "--root",str(root),"--spec",SPEC,"--columns",columns,"--head","N","--seed-index",str(seed),
        "--strict-admission","--data-role-decision",str(root/"human_role_decision.json"),
        "--dest",str(dest),"--train-only","--pack-production","--e33-output-stage"]


def run(study,output):
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("full fitting requires the shared compute lock")
    selection_path=study/"TRAIN_SELECTION.json"
    selection=json.loads(selection_path.read_text());admitted_selection(selection)
    complete=json.loads((study/"TRAIN_SCREEN_COMPLETE.json").read_text())
    if complete["selection_sha256"]!=sha(selection_path):raise ValueError("screen completion pin changed")
    for path,pin in selection["input_sha256"].items():
        if sha(path)!=pin:raise ValueError(f"screen input changed: {path}")
    source=json.loads((study/"SOURCE.json").read_text())
    if source["candidate"]!="MlicVote":raise ValueError("native-vote source required")
    binary=study/"bin";root=study/"MlicVote/root";frozen=study/"source"
    if sha(binary/"zensim_mlp_train")!=source["trainer_sha256"]:raise ValueError("screened trainer changed")
    columns=(study/"frozen_configuration/cols_c.txt").read_text().strip()
    output.mkdir()
    process=process_snapshot(os.getpid())
    state=dict(stage="full-fits",candidate="MlicVote",run_directory=str(output),screen_study=str(study),
        selection_sha256=sha(selection_path),source_commit=source["commit"],binary_sha256=source["trainer_sha256"],
        pid=process["pid"],start_ticks=process["start_ticks"],required=3,completed=[],failures=[],
        next_action="qualify all frozen full fits; fixed assessment seed 0; no serving promotion",objective_complete=False)
    def record():
        state["checked_utc"]=datetime.now(timezone.utc).isoformat()
        p=output/"RUN_STATE.json";tmp=p.with_suffix(".json.new")
        tmp.write_text(json.dumps(state,indent=2)+"\n");tmp.replace(p)
    record()
    (output/"INPUTS.json").write_text(json.dumps(dict(source=source,selection_sha256=sha(selection_path),
        root_sha256=sha(root/"wide/frozen.json"),runner_sha256=sha(__file__),assessment_seed=0,
        inference_model_selection="fixed seed 0; seeds 1 and 2 assess stability; no best-seed choice"),indent=2)+"\n")
    def fit(seed):
        dest=output/f"full_s{seed}"
        environment=dict(os.environ,REV4_V2_BIN_DIR=str(binary),ZENSIM_MAX_TIER="v3",RAYON_NUM_THREADS="1",
            OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(Path.home()/"tmp"))
        command=full_command(frozen,root,dest,columns,seed)
        with (output/f"full_s{seed}.log").open("x") as log:
            subprocess.run(command,cwd=frozen,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((dest/"result.json").read_text())
            init,sample=confirm_seeds(seed)
            model=dest/"refit/production-f16.bin"
            if (result["selection"]["selected_epoch"]!=119 or result["epochs"]!=120
                    or result["pairs_per_epoch"]!=50000 or result["seed_index"]!=seed
                    or (result["init_seed"],result["sample_seed"])!=(init,sample)
                    or result["spec"]!=SPEC or result["head"]!="N" or result["training_only"] is not True
                    or result["execution_contract"]!="registered-fit"
                    or result["frozen_sha256"]!=sha(root/"wide/frozen.json")
                    or result["packed_model_sha256"]!=sha(model)
                    or result["output_stage"]!="e33-identity-knot-tail2"):
                raise ValueError("native-vote full-fit binding differs")
            subprocess.run([str(binary/"bake_dial_refit"),"certify-monotone","--bake",str(model)],
                env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        return dict(seed=seed,model=str(model),sha256=sha(model),result_sha256=sha(dest/"result.json"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fit,seed):seed for seed in range(3)}
        for future in as_completed(futures):
            try:state["completed"].append(future.result())
            except BaseException as error:state["failures"].append(dict(seed=futures[future],error=repr(error)))
            record()
    if state["failures"]:
        state["stage"]="failed";record();raise RuntimeError("full-fit failures preserved; inspect individual logs")
    state["completed"].sort(key=lambda r:r["seed"])
    state["stage"]="full-fits-complete-requires-qualification";record()
    (output/"FULL_FITS.json").write_text(json.dumps(dict(models=state["completed"],assessment_seed=0,
        selection_sha256=sha(selection_path),production_qualified=False,objective_complete=False),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("study",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.study.resolve(),a.output.resolve())
