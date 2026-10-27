"""Pinned TRAIN diagnostic: isolate parameter packing from output calibration.

No model selection, EVAL reads, fitting, performance timing or threshold changes.
Run under the shared heavy lock and run-heavy on an eligible idle host.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictions(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or [int(r["row_idx"]) for r in rows] != list(range(len(rows))):
        raise ValueError("complete ordered prediction rows required")
    values = [float(r["pred"]) for r in rows]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("finite predictions required")
    return values


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if plan["schema"] != "mlic-parameter-packing-train-v1" or plan["role"] != "TRAIN calibration":
        raise ValueError("registered TRAIN diagnostic required")
    if sha(__file__) != plan["source_sha256"]:
        raise ValueError("source changed")
    if len(plan["models"]) != 3 or {m["seed"] for m in plan["models"]} != {0, 1, 2}:
        raise ValueError("all three original full seeds required")
    for path, pin in plan["inputs"].items():
        if sha(path) != pin:
            raise ValueError(f"input changed: {path}")
    required = {plan["binary"], plan["table"]}
    required.update(m[k] for m in plan["models"] for k in ("dense", "packed", "pack_log"))
    if not required <= plan["inputs"].keys():
        raise ValueError("unbound input")
    for model in plan["models"]:
        log = Path(model["pack_log"]).read_text()
        if "prune: no dead columns found" not in log or "L0:0/104960 L1:0/128" not in log:
            raise ValueError("original pack must have no pruning or zeroed weights")
    output.mkdir(parents=True, exist_ok=False)
    (output / "PLAN.json").write_bytes(plan_path.read_bytes())
    env = dict(os.environ, TMPDIR=str(Path.home() / "tmp"), RAYON_NUM_THREADS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3",
               ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb")
    def native(args, label):
        with (output / f"{label}.log").open("w") as log:
            subprocess.run([plan["binary"], *map(str, args)], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(schema=plan["schema"], plan_sha256=sha(plan_path),
                  population="frozen TRAIN calibration; in-sample diagnostic only",
                  selection_gate=False, performance_claim=False, rows=[])
    try:
        for m in sorted(plan["models"], key=lambda m: m["seed"]):
            prefix = f"s{m['seed']}"
            f32 = output / f"{prefix}-f32-calibrated.bin"
            native(["pack", "--in", m["dense"], "--out", f32, "--dtype", "f32",
                    "--zerobias-bulk", "0", "--protect-last", "--neg-tail",
                    "--identity-knot", "100", "--tail-extend", "2", "--no-prune",
                    "--anchor", plan["table"], "--target-col", "human_score", "--verify", "none"],
                   prefix + "-f32-pack")
            models = {"original": Path(m["dense"])}
            for name, source in (("f32", f32), ("f16", Path(m["packed"]))):
                models[name] = output / f"{prefix}-{name}-no-spline.bin"
                native(["strip", "--in", source, "--out", models[name],
                        "--key", "zentrain.output_calibration_spline"], prefix + "-strip-" + name)
            values = {}
            for name, model in models.items():
                dest = output / f"{prefix}-{name}.tsv"
                native(["predict", "--bake", model, "--corpus", plan["table"],
                        "--out", dest, "--score-units"], prefix + "-predict-" + name)
                values[name] = predictions(dest)
            if values["original"] != values["f32"]:
                raise ValueError("f32 control does not exactly reproduce original score predictions")
            if len(values["f16"]) != len(values["f32"]):
                raise ValueError("paired row counts differ")
            delta = [b-a for a,b in zip(values["f32"], values["f16"])]
            absolute = sorted(map(abs, delta))
            row = dict(seed=m["seed"], rows=len(delta), f32_control_exact=True,
                       unequal_rows=sum(d != 0 for d in delta), mean_signed=sum(delta)/len(delta),
                       mean_absolute=sum(absolute)/len(delta), max_absolute=absolute[-1],
                       rmse=math.sqrt(sum(d*d for d in delta)/len(delta)),
                       p99_absolute_nearest_rank=absolute[math.ceil(.99*len(delta))-1],
                       units="score units before output spline; native f32 inference",
                       model_sha256={k: sha(v) for k,v in models.items()})
            result["rows"].append(row)
            (output / "PROGRESS.json").write_text(json.dumps(result, indent=2)+"\n")
        result["checked_utc"] = datetime.now(timezone.utc).isoformat()
        result["artifacts"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        (output / "COMPLETE.json").write_text(json.dumps(result, indent=2)+"\n")
    except BaseException as error:
        (output / "FAILED.json").write_text(json.dumps({"error": repr(error), "completed": result})+"\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
