"""Run the preregistered mass comparison through existing fit and assessment owners.

Caller holds heavy.lock and run-heavy. Never resume a partly failed stage by
overwriting it: preserve the failure and inspect its artifacts first.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts/rev4_featpot"))
from run_processes import compute_lock, process_snapshot
from v2_common import sha

OLD = Path("/var/tmp/byc-mixes-20261025")
FOLDS = ("kadid","tid2013","konfig","cid22_a25","pipal","pairs","mos7")
TRAINER_SHA = "acfb2a9f24fd7a294f027b552b19eef1b9b4d707cbdf5e0bdf22ef328db3fe28"


def write(path, value):
    temporary = path.with_suffix(path.suffix+".new")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("--frozen",action="store_true")
    parser.add_argument("--source-commit",help="exact committed driver snapshot, fixed before queuing")
    args = parser.parse_args()
    study = args.study.resolve()
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("caller must hold the shared compute lock")
    if not args.frozen:
        commit = args.source_commit
        if commit is None or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("exact committed source must be fixed before queuing")
        subprocess.run(["git","merge-base","--is-ancestor",commit,"origin/main"],cwd=REPO,check=True)
        source = study/"source"
        source.mkdir()
        archive = study/"source.tar"
        with archive.open("xb") as stream:
            subprocess.run(["git","archive",commit],cwd=REPO,stdout=stream,check=True)
        subprocess.run(["tar","-xf",str(archive),"-C",str(source)],check=True)
        write(study/"SOURCE.json",dict(commit=commit,archive_sha256=sha(archive),
              driver_sha256=sha(source/"scripts/byconstruction/mlic_neff_chain.py")))
        os.execv(sys.executable,[sys.executable,"-B",str(source/"scripts/byconstruction/mlic_neff_chain.py"),
                                 str(study),"--frozen"])
    source = json.loads((study/"SOURCE.json").read_text())
    if sha(Path(__file__)) != source["driver_sha256"] or REPO != study/"source":
        raise ValueError("frozen coordinator source differs")
    state = dict(stage="prepare",run_directory=str(study),source_commit=source["commit"],
                 binary_sha256=TRAINER_SHA,pid=os.getpid(),start_ticks=process_snapshot(os.getpid())["start_ticks"],
                 next_action="21 paired TRAIN cells, fixed screen, then qualified-candidate follow-through")
    def run(stage, command, environment=None, output=None):
        state.update(stage=stage,command=list(map(str,command)),updated_utc=datetime.now(timezone.utc).isoformat())
        write(study/"RUN_STATE.json",state)
        print(json.dumps(dict(stage=stage,command=state["command"])),flush=True)
        with (study/f"{stage}.log").open("x") as log:
            if output is None:
                subprocess.run(command,cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            else:
                with output.open("x") as stream:
                    subprocess.run(command,cwd=REPO,env=environment,stdout=stream,stderr=log,check=True)
    try:
        write(study/"RUN_STATE.json",state)
        control = study/"control-cells/MlicControl/without_kadid_s0"
        record = json.loads((control/"byc_record.json").read_text())
        expected = OLD/"cells/Mlic/without_kadid_s0/pred_kadid.tsv"
        actual = control/"pred_kadid.tsv"
        if (record["trainer_sha256"] != TRAINER_SHA or record["rows"] != 7869
                or record["pred_sha256"] != sha(actual) or actual.read_bytes() != expected.read_bytes()):
            raise ValueError("unchanged Mlic prediction control failed")
        write(study/"CONTROL_VERIFIED.json",dict(byte_identical=True,rows=7869,prediction_sha256=sha(actual)))
        (study/"Mlic").mkdir()
        (study/"Mlic/root").symlink_to(OLD/"Mlic/root",target_is_directory=True)
        (study/"cells").mkdir()
        (study/"cells/Mlic").symlink_to(OLD/"cells/Mlic",target_is_directory=True)
        (study/"bin").symlink_to(OLD/"bin",target_is_directory=True)
        shutil.copytree(OLD/"frozen_configuration",study/"frozen_configuration")
        if sha(OLD/"BIN_SHA256SUMS") != "ef2d70d6e62b4c4bbeb1a553ffabf1e17ceefa1d62d99ac50bcdc21859965202":
            raise ValueError("original binary inventory changed")
        if sha(study/"frozen_configuration/cols_c.txt") != "e662c7225c7f6aca27bf208161b0d1b99f6d2ba7be427dfa37c4ac4ec1c49c75":
            raise ValueError("original feature column inventory changed")
        binaries = {}
        for line in (OLD/"BIN_SHA256SUMS").read_text().splitlines():
            pin,name = line.split(maxsplit=1)
            path = OLD/"bin"/Path(name.lstrip("*")).name
            if sha(path) != pin:
                raise ValueError("frozen study binary changed")
            binaries[str(path)] = pin
        if binaries[str(OLD/"bin/zensim_mlp_train")] != TRAINER_SHA:
            raise ValueError("Mlic trainer differs")
        run("root",["just","dataset-neff-root",str(OLD/"Mlic/root"),str(study/"MlicNeff/root")])
        pins = dict(binaries)
        for root in (OLD/"Mlic/root",study/"MlicNeff/root"):
            for relative in ("wide/frozen.json","datasets/receipt.json"):
                pins[str(root/relative)] = sha(root/relative)
        for fold in FOLDS:
            for seed in range(3):
                cell = OLD/f"cells/Mlic/without_{fold}_s{seed}"
                for path in [cell/"byc_record.json",cell/"result.json",cell/"refit/last.bin",*cell.glob("pred*.tsv")]:
                    pins[str(path)] = sha(path)
        for name in ("mixture","mixture_global"):
            dest = study/name
            dest.mkdir()
            shutil.copyfile(OLD/name/"manifest.json",dest/"manifest.json")
            pins[str(dest/"manifest.json")] = sha(dest/"manifest.json")
            (dest/"img").symlink_to((OLD/name/"img").resolve(),target_is_directory=True)
            for fold in FOLDS:
                for seed in range(3):
                    path = OLD/name/f"scores_Mlic_{fold}_s{seed}.json"
                    shutil.copyfile(path,dest/path.name)
                    pins[str(dest/path.name)] = sha(dest/path.name)
        write(study/"BASELINE.json",dict(trainer_sha256=TRAINER_SHA,input_sha256=pins,
                                        registration_commit="89d3e738af4e"))
        env = dict(os.environ,BYC_ROOT=str(study/"MlicNeff/root"),BYC_COLS=str(study/"frozen_configuration/cols_c.txt"),
                   BYC_CELLS=str(study/"cells"),BYC_SCRATCH=str(study/"scratch"),REV4_V2_BIN_DIR=str(study/"bin"),
                   ZENSIM_MAX_TIER="v3",RAYON_NUM_THREADS="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                   ZEN_PANEL_BIN=str(study/"bin/panel"),ZENSIM_PYRAMID="xyb")
        run("cells",[sys.executable,"-B","scripts/rev4_featpot/byc_cells.py","run","--arm","mono:thr:mp32",
                     "--label","MlicNeff","--folds",",".join(FOLDS),"--seeds","0,1,2","--jobs","12"],env)
        (study/"scorecard/label_reads").mkdir(parents=True)
        env["BYC_LABEL_LEDGER_DIR"] = str(study/"scorecard/label_reads")
        run("e21",[sys.executable,"-B","scripts/rev4_featpot/byc_cells.py","score","--arms","Mlic,MlicNeff",
                   "--paired-with","Mlic"],env,output=study/"scorecard/e21.json")
        run("diagnostics",["just","dataset-mix-diagnostics",str(study),str(study/"scorecard"),"Mlic,MlicNeff"],env)
        for name in ("mixture","mixture_global"):
            run(name,["bash","scripts/byconstruction/ring_mixture.sh",str(study/"bin"),"raw","12","MlicNeff"],
                dict(env,MIX=str(study/name),BYC_MIX_FOLDS=",".join(FOLDS)))
        run("selection",[sys.executable,"-B","scripts/rev4_featpot/byc_cells.py","neff-select","--study",str(study)],env)
        selection = json.loads((study/"TRAIN_SELECTION.json").read_text())
        state.update(stage="train-screen-complete",finalist=selection["finalist"],
                     next_action="full-data fits and production qualification" if selection["finalist"] else
                                 "retain Mlic; execute separately registered vote-loss/ordering comparison")
        write(study/"RUN_STATE.json",state)
        write(study/"TRAIN_SCREEN_COMPLETE.json",dict(selection_sha256=sha(study/"TRAIN_SELECTION.json"),
                                                     objective_complete=False))
    except BaseException as error:
        state.update(failure=repr(error),failed_utc=datetime.now(timezone.utc).isoformat())
        write(study/"FAILED.json",state)
        write(study/"RUN_STATE.json",state)
        raise


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
            write(Path(sys.argv[1])/"LAUNCH_FAILED.json",dict(error=repr(error),
                  utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid()))
        raise
