"""One coordinator EVAL assessment of the fixed, certified AIC TRAIN winner.

Uses the existing dataset_eval owners and the already admitted EVAL copies.
Never reads AIC public_test. No selection, refitting or checkpoint search.
Caller owns heavy.lock/run-heavy.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def mean_predictions(paths, output):
    tables = []
    for path in paths:
        with path.open() as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames != ["row_idx", "pred"]:
                raise ValueError("prediction columns differ")
            rows = list(reader)
        if [int(row["row_idx"]) for row in rows] != list(range(len(rows))):
            raise ValueError("prediction row order differs")
        values = [float(row["pred"]) for row in rows]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("empty or nonfinite prediction")
        tables.append(values)
    if len(tables) != 3 or len({len(table) for table in tables}) != 1:
        raise ValueError("three complete matching seed predictions required")
    with output.open("x") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["row_idx", "pred"])
        writer.writerows((i, sum(values) / 3) for i, values in enumerate(zip(*tables)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recovery", type=Path, help="bound missing-key recovery; preserve the failed assessment")
    args = parser.parse_args()
    study = args.study.resolve()
    output = args.output.resolve() if args.output else study / "assessment"
    if output.parent != study:
        raise ValueError("assessment output must be a fresh child of the study")
    output.mkdir()  # Never overwrite or silently repeat an assessment.
    from aic_finetune_status import status
    def state(stage):
        write(study / "EVAL_RUN_STATE.json", dict(stage=stage, pid=os.getpid(),
            utc=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(Path(__file__)),
            assessment_directory=str(output)))
    stop = threading.Event()
    def progress():
        while not stop.wait(15):
            write(study / "PROGRESS.json", status(study))
    monitor = threading.Thread(target=progress, daemon=True)
    try:
        state("coordinator-EVAL-preflight")
        write(study / "PROGRESS.json", status(study))
        monitor.start()
        plan = json.loads((study / "EVAL_PLAN.json").read_text())
        recovery = json.loads(args.recovery.read_text()) if args.recovery else None
        if recovery:
            if sha(study / "EVAL_PLAN.json") != recovery["plan_sha256"]:
                raise ValueError("recovery changes the frozen assessment plan")
            prior = Path(recovery["prior"])
            if sha(prior / "FAILED.json") != recovery["failure_sha256"]:
                raise ValueError("recovery failure receipt differs")
            failure_log = prior / "coordinator-EVAL-report-main.log"
            if (sha(failure_log) != recovery["failure_log_sha256"] or
                    "FileNotFoundError" not in failure_log.read_text() or
                    "judge_pairs.tsv" not in failure_log.read_text() or (prior / "main/report.json").exists()):
                raise ValueError("only missing judge-key report recovery is admitted")
        for population in ("main", "bapps"):
            available = set(plan["populations"][population]["files"])
            if recovery:
                available.update(recovery["extra_files"][population])
                if set(recovery["extra_files"][population]) != {"judge_pairs.tsv"}:
                    raise ValueError("recovery may only add the frozen judge-key table")
            if not {"features.parquet", "eval_rows.json", "LABEL_READS.json", "judge_pairs.tsv"} <= available:
                raise ValueError("EVAL owner requires feature, label-ledger and judge-key inputs")
        if not plan["label_free_gate_pass"] or plan["uses_EVAL_for_selection"]:
            raise ValueError("fixed TRAIN selection and passing label-free gates required")
        for relative, digest in plan["evidence_sha256"].items():
            if sha(study / relative) != digest:
                raise ValueError("frozen selection or qualification evidence changed")
        full = json.loads((study / "FULL_COMPLETE.json").read_text())
        if len(full["seeds"]) != 3 or {row["seed"] for row in full["seeds"]} != {0, 1, 2}:
            raise ValueError("all three frozen seeds required")
        for row in full["seeds"]:
            if sha(Path(row["bake"])) != row["sha256"]:
                raise ValueError("frozen EVAL model changed")
        frozen = json.loads((study / "BIN_FROZEN.json").read_text())
        for name in ("bake_dial_refit", "panel"):
            if sha(study / "bin" / name) != frozen["binaries"][name]:
                raise ValueError("frozen scoring/statistics tool changed")
        os.environ.update(ZEN_PANEL_BIN=str(study / "bin/panel"), ZENSIM_MAX_TIER="v3",
            ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb", RAYON_NUM_THREADS="1",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        owner = Path(__file__).resolve().parents[1] / "byconstruction/dataset_eval.py"
        if sha(owner) != plan["owner_sha256"]:
            raise ValueError("EVAL statistic owner changed")
        def command(label, argv):
            state(label)
            with (output / (label + ".log")).open("x") as log:
                subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        for population in ("main", "bapps"):
            state("coordinator-EVAL-copy-" + population)
            source = Path(plan["populations"][population]["source"])
            dest = output / population
            dest.mkdir()
            receipt = dict(role="EVAL", purpose="single frozen TRAIN-winner assessment",
                plan_sha256=sha(study / "EVAL_PLAN.json"), models=full["seeds"], source=str(source),
                files=dict(plan["populations"][population]["files"]), state="authorized-before-copy-and-read",
                existing_baseline_predictions_reused=True, AIC_public_test="unread")
            if recovery:
                receipt["files"].update(recovery["extra_files"][population])
                receipt["recovery_sha256"] = sha(args.recovery)
            write(dest / "EVAL_ASSESSMENT.json", receipt)
            for name, digest in receipt["files"].items():
                if sha(source / name) != digest:
                    raise ValueError("frozen EVAL copy or baseline changed")
                shutil.copyfile(source / name, dest / name)
                if sha(dest / name) != digest:
                    raise ValueError("EVAL copied bytes differ")
            names, paths = [], []
            for row in sorted(full["seeds"], key=lambda value: value["seed"]):
                name = f"{full['arm']}_s{row['seed']}"
                names.append(name)
                path = dest / f"pred_{name}.tsv"
                paths.append(path)
                reused = recovery and recovery["predictions"].get(population, {}).get(name)
                if reused:
                    previous = Path(recovery["prior"]) / population / path.name
                    if sha(previous) != reused:
                        raise ValueError("preserved seed prediction differs")
                    shutil.copyfile(previous, path)
                    if sha(path) != reused:
                        raise ValueError("reused seed prediction differs")
                else:
                    command(f"coordinator-EVAL-{population}-s{row['seed']}",
                        [str(study / "bin/bake_dial_refit"), "predict", "--bake", row["bake"],
                         "--corpus", str(dest / "features.parquet"), "--score-units", "--out", str(path)])
            ensemble = full["arm"] + "_ens"
            mean_predictions(paths, dest / f"pred_{ensemble}.tsv")
            command("coordinator-EVAL-report-" + population,
                [sys.executable, "-B", str(owner), "report", "--out", str(dest)])
            if population == "main":
                command("coordinator-EVAL-paired-main", [sys.executable, "-B", str(owner), "paired",
                    "--out", str(dest), "--dest", str(dest / "paired"), "--a", ensemble,
                    "--vs", "Mlic_ens,M7_ens,Mdata_ens,B7,ssim2,butteraugli,cvvdp", "--resamples", "2000"])
            write(dest / "COMPLETE.json", dict(report_sha256=sha(dest / "report.json"),
                new_predictions={name: sha(dest / f"pred_{name}.tsv") for name in names + [ensemble]},
                selection_changed=False, AIC_public_test="unread"))
        write(output / "COMPLETE.json", dict(plan_sha256=sha(study / "EVAL_PLAN.json"),
            assessment_only=True, selection_changed=False, production_adoption="not decided"))
        state("coordinator-EVAL-complete")
        return 0
    except BaseException as error:
        write(output / "FAILED.json", dict(error=str(error), traceback=traceback.format_exc(),
            utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
        state("coordinator-EVAL-failed")
        raise
    finally:
        stop.set()
        if monitor.is_alive():
            monitor.join()
        write(study / "PROGRESS.json", status(study))


if __name__ == "__main__":
    raise SystemExit(main())
