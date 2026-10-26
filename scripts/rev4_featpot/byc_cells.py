"""By-construction exploration: run E21 LODO cells for recipe arms and score them against E34's stored control.

Exploration harness only (benchmarks/byconstruction_2026-10-22.md). Each cell is the registered `v2_lodo_mlp.py` strict
training-only route (same tables, seeds, recipe owner) with the arm's recipe tokens appended to B's C-family spec, followed
by a held-out prediction through `v2_lodo_mlp.predict` (dense bake + `bake_dial_refit predict --score-units`). Scoring
pairs every arm with E34's stored `base-c` predictions for the same fold and seed index (E34 `panel-input.json`), which
this harness reproduced bit for bit (kadid seed 0, 7,869 rows).

  run    --arm TOKENS --folds kadid,tid2013,konfig,cid22_a25 --seeds 0,1,2 --jobs N [--aic-weight W --label L]
  score  --arms NAME[,NAME...] [--seeds 0,1,2]
  run    --folds pipal,pairs,mos7 ...   (omit new-data groups, score their TRAIN rows)
  run    --folds full ...              (subset full fit; score groups already omitted)
  mix-audit / mix-select / mix-freeze --study DIR
  progress --arm NAME [--seeds 0,1,2]  (record presence and latest logged epoch; no target table opened)
  aicscore --arms NAME[,NAME...]   (AIC2026 TRAIN out-of-fold panel of --aic-weight arms)
  aiczero --arm NAME=TABLE [--arm ...] [--out JSON]
                                   (AIC2026 public_train zero-shot of every cell of arms trained without AIC: TABLE holds
                                    label-free features of pairs_train.tsv in row order, from the arm's own extraction)

Point estimates only: the E21 statistic (mean over the four D1 sources of signed SROCC, arm minus paired control,
averaged over seeds) with its seed-unit SE. Registered intervals and tests belong to the `panel e34` owner.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
import threading
from pathlib import Path

# BYC_ROOT selects a derived root (e.g. the hpm ringing root from scripts/byconstruction/ring_root.py).
ROOT = Path(os.environ.get("BYC_ROOT", "/mnt/v/output/zensim/v40r4-2026-10-08/v2e29"))
BASE_SPEC = "sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1"
E34_PANEL = Path("/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json")
FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")
MIX_ARMS = ("Mdata", "Mw025", "Mw100", "Mmos", "Mpair", "Mlic")
VALIDATE_ONLY_NEW_LEGS = ("vcl-fer", "pieapp", "fg-iqa")
HERE = Path(__file__).resolve().parent
CELLS = Path(os.environ.get("BYC_CELLS", str(Path.home() / "tmp/byc/cells")))
COLS = Path(os.environ.get("BYC_COLS", str(Path.home() / "tmp/byc/cols_c.txt")))
# v2_common resolves its root and binaries at import; this harness binds both before importing it.
os.environ.setdefault("REV4_V2_ROOT", str(ROOT))
os.environ.setdefault("REV4_V2_BIN_DIR", "/var/tmp/byconstruction-target/release")
sys.path.insert(0, str(HERE))
from v2_common import sha  # noqa: E402
from subjective_training_leg import read_train_panel  # noqa: E402
from v2_lodo_mlp import EPOCH_RE, predict, read_curve  # noqa: E402


def spec_of(tokens: str) -> str:
    """B's C-family spec (or `--base-spec`) with the arm's tokens. `H<n>` replaces H128; other tokens are appended."""
    toks = [t for t in tokens.split(":") if t]
    base = BASE_SPEC
    hidden = [t for t in toks if t.startswith("H")]
    if hidden:
        base = base.replace(":H128:", f":{hidden[0]}:")
    rest = [t for t in toks if not t.startswith("H")]
    return base + "".join(f":{t}" for t in rest)


DATASET_SCORE_LOCK = threading.Lock()
LABEL = None
# AIC2026 TRAIN (benchmarks/byconstruction_2026-10-22.md, section "AIC2026 TRAIN"): with --aic-weight W each E21 cell also
# fits the AIC2026 public_train references outside one fold (pooled rank, research replay) and predicts that fold. Fold
# of cell (E21 fold index i, seed s): (4 s + i) mod AIC_NFOLDS (7 by default), so seeds 0-2 hold out every fold.
AIC = Path(os.environ.get("AIC_FOLDS", "/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/folds_train"))
AIC_NFOLDS = int(os.environ.get("AIC_NFOLDS", "7"))
AIC_WEIGHT = None
AIC_REPLAY = "byconstruction AIC2026 TRAIN research fit (owner goal 2026-10-23); reference-held-out folds; never ship"


def aic_fold(fold: str, seed: int) -> int:
    return (4 * seed + FOLDS.index(fold)) % AIC_NFOLDS


def arm_name(tokens: str) -> str:
    return LABEL or tokens.replace(":", "-") or "control"


def predict_cell(bake: Path, table: Path, out: Path, spec: str):
    """Held-out predictions. Bakes that read research ids past zensim's registry (fx1g, the byc_globaff descriptor)
    go through `byc_predict` (zenpredict, same row layout; bit-identical to `bake_dial_refit predict --score-units` on
    fx1 bakes); every other bake through the registered owner."""
    token = next((t for t in spec.split(":") if t in ("fx1g", "fx1gs1", "fx1gg", "fx1gy", "fx1o")), None)
    if token is None:
        return predict(bake, table, out)
    import numpy as np
    import pandas as pd
    from v2_common import derived_declaration
    decl, _, _ = derived_declaration(token)
    subprocess.run([str(Path(os.environ["REV4_V2_BIN_DIR"]) / "byc_predict"), "--bake", str(bake), "--decl", str(decl),
                    "--corpus", str(table), "--out", str(out)], check=True, capture_output=True)
    pred = pd.read_csv(out, sep="\t").pred.to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError(f"{out}: nonfinite prediction")
    return pred


def aic_predict(dest: Path, fold: str, seed: int, spec: str) -> dict:
    k = aic_fold(fold, seed)
    pred = predict_cell(dest / "refit/last.bin", AIC / f"aic_k{k}_test.parquet", dest / f"pred_aic_k{k}.tsv", spec)
    return {"aic_weight": AIC_WEIGHT, "aic_fold": k, "aic_rows": int(len(pred)),
            "aic_pred_sha256": sha(dest / f"pred_aic_k{k}.tsv")}


def dataset_holdout_collections(root, fold):
    from subjective_training_leg import DATASET_FOLDS, linked_subset_exclusions
    packet = json.loads((root / "datasets/receipt.json").read_text())
    trained = {r["collection"] for r in packet["groups"]}
    if fold == "full":
        # A subset arm's full fit already excludes these labels.
        missing = tuple(c for cs in DATASET_FOLDS.values() for c in cs if c not in trained)
        if not missing:
            raise ValueError("full dataset-fold scoring requires a subset arm")
        return missing, ()
    linked = linked_subset_exclusions(root, fold)
    if fold in FOLDS:
        return (), linked
    omitted = DATASET_FOLDS[fold]
    if not trained.intersection(omitted):
        raise ValueError("do not fit a no-op dataset holdout; reuse the arm's full fit")
    return omitted, tuple(dict.fromkeys((*omitted, *linked)))


def dataset_train_statistic(refs, targets, predictions, pairwise):
    """Use canonical signed midranks for MOS; exact hard-pair ordering for pairs."""
    import numpy as np
    from collections import defaultdict
    by = defaultdict(list)
    for ref, target, pred in zip(refs, targets, predictions, strict=True):
        if not (np.isfinite(target) and np.isfinite(pred)):
            raise ValueError("nonfinite TRAIN target or prediction")
        by[ref].append((float(target), float(pred)))
    if pairwise:
        agreements = []
        for points in by.values():
            if len(points) != 2 or {p[0] for p in points} != {0., 1.}:
                raise ValueError("hard pair requires exactly one winner and one loser")
            a, b = sorted(points)
            agreements.append(float(b[1] > a[1]) + 0.5 * float(b[1] == a[1]))
        if not agreements:
            raise ValueError("empty pairwise population")
        return dict(statistic="hard-pair-agreement", value=float(np.mean(agreements)),
                    groups=len(by), defined_groups=len(by), rows=len(predictions),
                    predicted_ties=sum(v == 0.5 for v in agreements))
    sys.path.insert(0, str(HERE.parent))
    from lib.zen_stats import panel_batch
    jobs = [(str(i), [p[1] for p in points], [p[0] for p in points])
            for i, points in enumerate(by.values()) if len(points) >= 2
            and len({p[0] for p in points}) > 1 and len({p[1] for p in points}) > 1]
    vals = [v["srocc_signed"] for v in panel_batch(jobs, stats="srocc")] if jobs else []
    if not vals or not np.isfinite(vals).all():
        raise ValueError("collection has no complete defined within-reference SROCC")
    return dict(statistic="within-reference-srocc", value=float(np.mean(vals)),
                groups=len(by), defined_groups=len(vals), excluded_groups=len(by)-len(vals), rows=len(predictions))


def score_dataset_holdout(dest, collections, spec):
    import pyarrow.parquet as pq
    from subjective_training_leg import DATASET_FOLDS, log_dataset_read
    score_root = Path(os.environ.get("BYC_DATASET_SCORE_ROOT",
        "/var/tmp/datalegs-20261025/rev7data/root"))
    packet_path = score_root / "datasets/receipt.json"
    packet = json.loads(packet_path.read_text())
    frozen = json.loads((score_root / "wide/frozen.json").read_text())
    if sha(packet_path) != frozen["dataset_legs_sha256"]:
        raise ValueError("TRAIN score root is not frozen")
    groups = {r["collection"]: r for r in packet["groups"]}
    fit = json.loads((dest / "DATASET_GROUP_SELECTION.json").read_text())
    if set(collections).intersection(fit["retained_collections"]):
        raise ValueError("new-data scoring group was retained in fitting")
    stats = {}
    for collection in collections:
        group = groups[collection]
        table = Path(group["path"])
        if sha(table) != group["table_sha256"]:
            raise ValueError("TRAIN score table pin changed")
        log_dataset_read(dest, table, group["rows"], ["ref_basename", "human_score"],
                         "heldout-new-dataset-TRAIN-score", collection=collection,
                         table_sha256=group["table_sha256"])
        pred_path = dest / f"pred_dataset_{collection}.tsv"
        pred = predict_cell(dest / "refit/last.bin", table, pred_path, spec)
        labels = pq.read_table(table, columns=["ref_basename", "human_score"])
        stats[collection] = dict(dataset_train_statistic(labels["ref_basename"].to_pylist(),
            labels["human_score"].to_pylist(), pred, collection in DATASET_FOLDS["pairs"]),
            prediction_sha256=sha(pred_path), table_sha256=group["table_sha256"])
    return stats


def run_cell(tokens: str, fold: str, seed: int) -> dict:
    dest = CELLS / arm_name(tokens) / f"without_{fold}_s{seed}"
    record = dest / "byc_record.json"
    linked_pin = None
    packet_path = ROOT/"datasets/receipt.json"
    if packet_path.exists() and fold != "full":
        packet = json.loads(packet_path.read_text())
        if packet.get("subset_policy_sha256"):
            from subjective_training_leg import linked_subset_exclusions
            linked_subset_exclusions(ROOT, fold)
            linked_pin = packet["linked_subset_omissions"]["sha256"]
    if record.is_file():
        value = json.loads(record.read_text())
        if linked_pin is not None:
            fit = json.loads((dest/"result.json").read_text())
            if (value.get("linked_subset_omission_sha256") != linked_pin
                    or fit["frozen_sha256"] != sha(ROOT/"wide/frozen.json")):
                raise ValueError("cached linked-subset cell does not bind the registered omission/root")
        return value
    if dest.exists():
        raise ValueError(f"{dest}: incomplete earlier attempt; inspect and move it aside")
    scratch = Path(os.environ.get("BYC_SCRATCH", str(Path.home() / "tmp/byc/scratch"))) / arm_name(tokens) / f"{fold}_s{seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch), ZENSIM_MAX_TIER="v3")
    from subjective_training_leg import DATASET_FOLDS
    new_fold = fold in (*DATASET_FOLDS, "full")
    if fold not in (*FOLDS, *DATASET_FOLDS, "full"):
        raise ValueError(f"unknown fold: {fold}")
    cmd = [sys.executable, str(HERE / "v2_lodo_mlp.py"), "--root", str(ROOT), "--spec", spec_of(tokens),
           "--columns", COLS.read_text().strip(), "--head", "N", "--heldout", fold, "--seed-index", str(seed),
           "--strict-admission", "--data-role-decision", str(ROOT / "human_role_decision.json"),
           "--dest", str(dest), "--train-only"]
    if new_fold:
        if AIC_WEIGHT is not None:
            raise ValueError("new dataset folds do not add AIC labels")
        collections, excluded = dataset_holdout_collections(ROOT, fold)
        cmd[1] = str(HERE / "v2_confirm_fit.py")
        heldout_index = cmd.index("--heldout")
        del cmd[heldout_index:heldout_index+2]
    else:
        from subjective_training_leg import linked_subset_exclusions
        excluded = linked_subset_exclusions(ROOT, fold)
    if excluded:
        cmd += ["--exclude-datasets", ",".join(excluded)]
    if AIC_WEIGHT is not None:
        k = aic_fold(fold, seed)
        cmd += ["--research-extra-group", f"aic:{AIC}/aic_k{k}_fit.parquet:{AIC_WEIGHT!r}:0:rank",
                "--research-replay", AIC_REPLAY]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    train_s = time.time() - t0
    dest.parent.mkdir(parents=True, exist_ok=True)
    (dest.parent / f"{dest.name}.wrapper.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"{dest}: rc={proc.returncode}\n{proc.stderr[-2000:]}")
    dataset_stats = None
    if new_fold:
        # Large feature-table prediction stays serial while other fits continue.
        with DATASET_SCORE_LOCK:
            dataset_stats = score_dataset_holdout(dest, collections, spec_of(tokens))
    pred = None if new_fold else predict_cell(dest / "refit/last.bin", ROOT / "wide/main/real" / f"{fold}.parquet",
                                             dest / f"pred_{fold}.tsv", spec_of(tokens))
    rec = {"tokens": tokens, "spec": spec_of(tokens), "fold": fold, "seed": seed, "train_seconds": round(train_s, 1),
           **({} if AIC_WEIGHT is None else aic_predict(dest, fold, seed, spec_of(tokens))),
           "last_bin_sha256": sha(dest / "refit/last.bin"),
           **({"dataset_train_statistics": dataset_stats} if new_fold else
              {"pred_sha256": sha(dest / f"pred_{fold}.tsv"), "rows": int(len(pred))}),
           "curve": read_curve(dest / "train.log"), "bin_dir": os.environ["REV4_V2_BIN_DIR"],
           "trainer_sha256": sha(Path(os.environ["REV4_V2_BIN_DIR"]) / "zensim_mlp_train")}
    for line in (dest / "train.log").read_text().splitlines():
        if line.startswith("  epoch 119 "):
            rec["epoch119"] = line.split("| t=")[0].strip()
    if linked_pin is not None:
        rec.update(linked_subset_omission_sha256=linked_pin, dataset_exclusions=list(excluded))
    record.write_text(json.dumps(rec, indent=1) + "\n")
    for leftover in scratch.glob("*"):
        leftover.unlink()
    return rec


def cmd_run(args) -> int:
    global BASE_SPEC, COLS, LABEL, AIC_WEIGHT
    LABEL = args.label
    if args.aic_weight is not None:
        AIC_WEIGHT = args.aic_weight
        if not args.label:
            raise ValueError("--aic-weight needs --label (an AIC arm is a different arm)")
        LABEL = args.label
    if args.base_spec:
        BASE_SPEC, COLS, LABEL = args.base_spec, Path(args.cols), args.label
        if not (args.cols and args.label):
            raise ValueError("--base-spec needs --cols and --label")
    folds = args.folds.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = [(f, s) for s in seeds for f in folds]
    failures = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_cell, args.arm, f, s): (f, s) for f, s in cells}
        for fut in cf.as_completed(futures):
            f, s = futures[fut]
            try:
                rec = fut.result()
                print(json.dumps({"arm": arm_name(args.arm), "fold": f, "seed": s, "train_s": rec["train_seconds"]}),
                      flush=True)
            except Exception as e:  # report every cell; the run fails at the end
                failures += 1
                print(json.dumps({"arm": arm_name(args.arm), "fold": f, "seed": s, "error": str(e)[-1500:]}), flush=True)
    return 1 if failures else 0


def srocc(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def cmd_progress(args) -> int:
    from datetime import datetime, timezone
    from run_processes import bound_worker, compute_lock, process_tree
    state = json.loads(Path(args.run_state).read_text()) if getattr(args,"run_state",None) else {}
    worker = bound_worker(state.get("pid"), state.get("start_ticks"))
    cells = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for fold in args.folds.split(","):
            path = CELLS / args.arm / f"without_{fold}_s{seed}"
            record, log = path / "byc_record.json", path / "train.log"
            if record.is_file():
                r = json.loads(record.read_text())
                if (r["fold"], r["seed"]) != (fold, seed):
                    raise ValueError(f"{record}: cell identity differs")
                cell_state, epoch = "complete", 119
            elif log.is_file():
                text = log.read_text()
                epochs = [int(e) for e, _ in EPOCH_RE.findall(text)]
                # The rank trainer records a progress marker every epoch between
                # its less frequent metric rows; include those for live status.
                epochs += [int(e) for e in re.findall(r"subjective-rank epoch=(\d+)", text)]
                cell_state, epoch = "training-log-present", max(epochs) if epochs else None
            else:
                cell_state, epoch = "preparing" if path.exists() else "not-started", None
            cells.append(dict(fold=fold, seed=seed, state=cell_state, latest_logged_epoch=epoch))
    result = dict(arm=args.arm, complete=sum(r["state"] == "complete" for r in cells),
                  requested=len(cells), cells=cells, checked_utc=datetime.now(timezone.utc).isoformat())
    if state:
        lock = compute_lock(worker["pid"] if worker else None)
        result.update(run_directory=state["run_directory"], declared_stage=state["stage"],
                      source_commit=state["source_commit"], binary_sha256=state["binary_sha256"],
                      live_processes=process_tree(worker["pid"]) if worker else [], compute_lock=lock,
                      worker_alive=bool(worker), next_action=state["next_action"])
        result["failure"] = state.get("failure")
        launch_failure = Path(state["run_directory"])/"LAUNCH_FAILED.json"
        if launch_failure.is_file():
            result["failure"] = json.loads(launch_failure.read_text())
        result["status"] = ("failed" if result["failure"] else
                            "running" if worker and lock["held_by_worker_ancestor"] else
                            "cells-complete" if result["complete"] == result["requested"] else "not-running")
    text = json.dumps(result, indent=1)+"\n"
    if getattr(args,"output",None):
        output = Path(args.output)
        temporary = output.with_suffix(output.suffix+".new")
        temporary.write_text(text)
        temporary.replace(output)
    print(text,end="")
    return 0


def cmd_score(args) -> int:
    import numpy as np
    import pandas as pd
    panel = read_train_panel(E34_PANEL)
    data = {d["name"]: d for d in panel["datasets"]}
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {}
    for arm in args.arms.split(","):
        per_seed, per_ds, absolute = [], {f: [] for f in FOLDS}, {f: [] for f in FOLDS}
        for s in seeds:
            deltas = []
            for f in FOLDS:
                d = data[f]
                target = np.asarray(d["target"], dtype=float)
                orient = float(d["orientation"])
                if args.rule == "best":
                    read = lambda a: pd.read_csv(BEST / a / f"without_{f}_s{s}" / f"pred_{f}.tsv",
                                                 sep="\t").pred.to_numpy(dtype=float)
                    pa, pc = read(arm), read(args.control)
                else:
                    if arm in d["predictions"]:
                        pa = np.asarray(d["predictions"][arm][s], dtype=float)
                    else:
                        t = pd.read_csv(CELLS / arm / f"without_{f}_s{s}" / f"pred_{f}.tsv", sep="\t")
                        pa = t.pred.to_numpy(dtype=float)
                    pc = np.asarray(d["predictions"][args.control][s], dtype=float)
                if len(pa) != len(target):
                    raise ValueError(f"{arm}/{f}/s{s}: {len(pa)} predictions for {len(target)} targets")
                ra, rc = orient * srocc(pa, target), orient * srocc(pc, target)
                deltas.append(ra - rc)
                per_ds[f].append(ra - rc)
                absolute[f].append(ra)
            per_seed.append(float(np.mean(deltas)))
        n = len(per_seed)
        mean = float(np.mean(per_seed))
        se = float(np.std(per_seed, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        out[arm] = {"control": args.control, "seeds": seeds, "delta": mean, "seed_unit_se": se,
                    "per_seed_delta": per_seed,
                    "per_dataset_delta": {f: float(np.mean(v)) for f, v in per_ds.items()},
                    "per_dataset_srocc": {f: float(np.mean(v)) for f, v in absolute.items()},
                    "per_dataset_per_seed_srocc": absolute,
                    "e33_point_form": bool(mean >= -0.002 and all(np.mean(v) >= -0.005 for v in per_ds.values()))}
    if args.paired_with:
        # Each arm against another cell arm, seed by seed: the per-seed E21 deltas share the control, so their
        # difference is the arms' own paired difference; its seed-unit SE is the comparison's.
        ref = out[args.paired_with]
        for arm, res in out.items():
            if arm == args.paired_with:
                continue
            diff = [a - b for a, b in zip(res["per_seed_delta"], ref["per_seed_delta"])]
            res["paired_with"] = {
                "arm": args.paired_with, "per_seed_difference": diff, "difference": float(np.mean(diff)),
                "seed_unit_se": float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan"),
                "per_dataset_difference": {f: res["per_dataset_delta"][f] - ref["per_dataset_delta"][f] for f in FOLDS}}
            ds_stats = {}
            for f in FOLDS:
                values = [a-b for a,b in zip(res["per_dataset_per_seed_srocc"][f], ref["per_dataset_per_seed_srocc"][f], strict=True)]
                ds_stats[f] = dict(per_seed_difference=values, seed_unit_se=float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values)>1 else float("nan"))
            res["paired_with"]["per_dataset_seed_statistics"] = ds_stats
    print(json.dumps(out, indent=1))
    return 0


def cmd_mix_audit(args):
    """Verify completed cells and their exclusion inventories without target reads."""
    from subjective_training_leg import DATASET_FOLDS
    study = Path(args.study)
    output = study / "NATIVE_CELL_CONTROLS.json"
    if output.exists():
        raise ValueError("preserve the completed cell audit")
    required = {
        "Mdata": (*FOLDS,*DATASET_FOLDS),
        "Mw025": (*FOLDS,*DATASET_FOLDS),
        "Mw100": (*FOLDS,*DATASET_FOLDS),
        "Mlic": (*FOLDS,*DATASET_FOLDS),
        "Mmos": (*FOLDS,"pipal","mos7","full"),
        "Mpair": (*FOLDS,"pairs","full"),
    }
    inventory = []
    for arm,folds in required.items():
        root = study/arm/"root" if arm != "Mdata" else Path("/var/tmp/datalegs-20261025/rev7data/root")
        packet = json.loads((root/"datasets/receipt.json").read_text())
        trained = {r["collection"] for r in packet["groups"]}
        baseline_root = Path("/var/tmp/datalegs-20261025/rev7data/root")
        baseline = json.loads((baseline_root/"datasets/receipt.json").read_text())
        expected_collections = {r["collection"] for r in baseline["groups"]}
        if arm == "Mmos":
            expected_collections -= set(DATASET_FOLDS["pairs"])
        elif arm == "Mpair":
            expected_collections = set(DATASET_FOLDS["pairs"])
        elif arm == "Mlic":
            expected_collections -= set(VALIDATE_ONLY_NEW_LEGS)
        if trained != expected_collections:
            raise ValueError("mix root collection subset differs")
        nominal = {"Mw025":0.25,"Mw100":1.}.get(arm,0.5)
        baseline_groups = {r["collection"]:r for r in baseline["groups"]}
        for group in packet["groups"]:
            declaration = json.loads(Path(str(group["path"])+".manifest.json").read_text())
            if (declaration["nominal_weight"] != nominal
                    or group["table_sha256"] != baseline_groups[group["collection"]]["table_sha256"]):
                raise ValueError("mix root weight or frozen TRAIN rows differ")
        for table in (baseline_root/"wide/main/real").glob("*.parquet"):
            if sha(root/"wide/main/real"/table.name) != sha(table):
                raise ValueError("inherited tables differ across dataset mixes")
        for fold in folds:
            for seed in range(3):
                cell = CELLS/arm/f"without_{fold}_s{seed}"
                rec = json.loads((cell/"byc_record.json").read_text())
                result = json.loads((cell/"result.json").read_text())
                if (rec["seed"] != seed or rec["fold"] != fold or rec["spec"] != spec_of("mono:thr:mp32")
                        or result["selection"]["selected_epoch"] != 119 or result["epochs"] != 120
                        or result["pairs_per_epoch"] != 50000 or result["seed_index"] != seed
                        or result["frozen_sha256"] != sha(root/"wide/frozen.json")
                        or rec["last_bin_sha256"] != sha(cell/"refit/last.bin")):
                    raise ValueError(f"{cell}: native cell binding differs")
                if not (arm == "Mdata" and fold in FOLDS):
                    bound = json.loads((cell/"DATASET_GROUP_SELECTION.json").read_text())
                    retained = trained - set(DATASET_FOLDS.get(fold,()))
                    if set(bound["retained_collections"]) != retained or bound["root_sha256"] != sha(root/"wide/frozen.json"):
                        raise ValueError(f"{cell}: dataset exclusion inventory differs")
                    if not (cell/"LABEL_READS.json").is_file():
                        raise ValueError("new native fit is missing its pre-read ledger")
                if fold in FOLDS:
                    if sha(cell/f"pred_{fold}.tsv") != rec["pred_sha256"]:
                        raise ValueError("E21 predictions changed")
                else:
                    scored, _ = dataset_holdout_collections(root,fold)
                    if set(rec["dataset_train_statistics"]) != set(scored):
                        raise ValueError("new-data score coverage differs")
                    for collection,value in rec["dataset_train_statistics"].items():
                        if sha(cell/f"pred_dataset_{collection}.tsv") != value["prediction_sha256"]:
                            raise ValueError("new-data predictions changed")
                read_curve(cell/"train.log")
                inventory.append(dict(arm=arm,fold=fold,seed=seed,record_sha256=sha(cell/"byc_record.json"),
                                      model_sha256=rec["last_bin_sha256"]))
    value=dict(schema="dataset-mix-cell-controls-v1",cells=inventory,verified_cells=len(inventory),
               new_native_fits=len(inventory)-12,reused_Mdata_E21_cells=12,selected_epoch=119)
    output.write_text(json.dumps(value,indent=1)+"\n")
    print(json.dumps({k:v for k,v in value.items() if k != "cells"}))
    return 0


def mix_arm_statistics(arm, e21, records):
    """The registered old/new balanced statistic, shared by prospective comparisons."""
    import numpy as np
    from subjective_training_leg import DATASET_FOLDS
    expected = {c for cs in DATASET_FOLDS.values() for c in cs}
    if e21[arm]["seeds"] != [0,1,2]:
        raise ValueError("mix selection requires registered seed order 0,1,2")
    new, old = [], []
    per_collection = {c: [] for c in expected}
    for seed in range(3):
        values = {}
        for fold in (*DATASET_FOLDS, "full"):
            recpath = CELLS / arm / f"without_{fold}_s{seed}/byc_record.json"
            if not recpath.exists():
                continue
            record = json.loads(recpath.read_text())
            records[str(recpath)] = sha(recpath)
            stats = record["dataset_train_statistics"]
            if set(values).intersection(stats):
                raise ValueError("overlapping new-dataset score records")
            values.update(stats)
        if set(values) != expected:
            raise ValueError(f"{arm}/s{seed}: incomplete new-dataset assessment")
        mapped = []
        for c, value in sorted(values.items()):
            v = value["value"]
            if not np.isfinite(v):
                raise ValueError("undefined collection statistic blocks selection")
            per_collection[c].append(v)
            mapped.append(v if c in DATASET_FOLDS["pairs"] else (v+1)/2)
        new.append(float(np.mean(mapped)))
        old_values = [(e21[arm]["per_dataset_per_seed_srocc"][f][seed]+1)/2 for f in FOLDS]
        if not np.isfinite(old_values).all():
            raise ValueError("undefined E21 collection statistic blocks selection")
        old.append(float(np.mean(old_values)))
    balanced = [(o+n)/2 for o,n in zip(old,new,strict=True)]
    def describe(v):
        return dict(per_seed=v, mean=float(np.mean(v)), seed_unit_se=float(np.std(v,ddof=1)/np.sqrt(3)))
    return dict(old=describe(old), new=describe(new), balanced=describe(balanced),
                     eligible_under_new_leg_licence_policy=arm == "Mlic",
                     per_collection={c:describe(v) for c,v in sorted(per_collection.items())})


def cmd_mix_select(args):
    """Freeze the preregistered decision from completed TRAIN-only cell records."""
    study = Path(args.study)
    output = study / "TRAIN_SELECTION.json"
    if output.exists():
        raise ValueError("preserve the frozen TRAIN decision")
    e21_path = study / "scorecard/e21.json"
    e21 = json.loads(e21_path.read_text())
    if len(args.arms.split(",")) != len(MIX_ARMS) or set(args.arms.split(",")) != set(MIX_ARMS):
        raise ValueError("the licence addendum requires all six registered mix arms")
    policy_path = study / "LICENCE_POLICY.json"
    policy = json.loads(policy_path.read_text())
    if (set(policy["excluded_from_Mlic"]) != set(VALIDATE_ONLY_NEW_LEGS)
            or policy["eligible_new_leg_arms"] != ["Mlic"]):
        raise ValueError("the owner's new-leg licence policy differs")
    if sha(Path(policy["copied_manifest"])) != policy["source_sha256"]:
        raise ValueError("the frozen licence-policy dataset manifest changed")
    arms = {}
    records = {str(e21_path): sha(e21_path), str(policy_path): sha(policy_path),
               policy["copied_manifest"]: policy["source_sha256"]}
    for arm in args.arms.split(","):
        arms[arm] = mix_arm_statistics(arm, e21, records)
    balanced_winner = min(arms, key=lambda a:(-arms[a]["balanced"]["mean"],-arms[a]["new"]["mean"],a))
    old_winner = min(arms, key=lambda a:(-arms[a]["old"]["mean"],-arms[a]["new"]["mean"],a))
    result = dict(schema="dataset-mix-TRAIN-selection-v1", role="TRAIN", rule_commit="7155abdcf3fc",
                  finalists=list(dict.fromkeys((balanced_winner,old_winner))), assessment_seed=0,
                  balanced_winner=balanced_winner, old_winner=old_winner, arms=arms,
                  input_sha256=records, eval_labels_used=False,
                  licence_addendum="2026-10-25", selection_scope="research-comparison",
                  eligible_new_leg_arms=["Mlic"])
    output.write_text(json.dumps(result,indent=1)+"\n")
    print(json.dumps(result,indent=1))
    return 0


def neff_decision(arms, global_inversions):
    """Rule 89d3e738: paired TRAIN gain, old-data noninferiority, global monotonicity."""
    import numpy as np
    delta = {}
    for metric in ("old", "new", "balanced"):
        values = [a-b for a,b in zip(arms["MlicNeff"][metric]["per_seed"],
                                     arms["Mlic"][metric]["per_seed"], strict=True)]
        if len(values) != 3 or not np.isfinite(values).all():
            raise ValueError("three finite paired seed values required")
        delta[metric] = dict(per_seed=values, mean=float(np.mean(values)),
                            seed_unit_se=float(np.std(values,ddof=1)/np.sqrt(3)))
    if len(global_inversions) != 21 or any(type(v) is not int or v < 0 for v in global_inversions):
        raise ValueError("twenty-one global mixture cell counts required")
    checks = dict(balanced_gain=delta["balanced"]["mean"] > 0,
                  old_noninferiority=delta["old"]["mean"] >= -2*delta["old"]["seed_unit_se"],
                  zero_global_inversions=all(v == 0 for v in global_inversions))
    return dict(paired_delta=delta, checks=checks,
                finalist="MlicNeff" if all(checks.values()) else None,
                retain_control=not all(checks.values()), production_qualified=False)


def cmd_neff_select(args):
    """Bind every cell and mixture model before freezing the prospective decision."""
    from subjective_training_leg import DATASET_FOLDS
    sys.path.insert(0, str(HERE.parent/"byconstruction"))
    from mixture_test import report
    study = Path(args.study)
    output = study/"TRAIN_SELECTION.json"
    if output.exists():
        raise ValueError("preserve frozen effective-sample decision")
    baseline_path = study/"BASELINE.json"
    baseline = json.loads(baseline_path.read_text())
    for path,pin in baseline["input_sha256"].items():
        if sha(path) != pin:
            raise ValueError("frozen baseline input changed")
    e21_path = study/"scorecard/e21.json"
    e21 = json.loads(e21_path.read_text())
    inputs = {str(e21_path):sha(e21_path), str(baseline_path):sha(baseline_path)}
    inventory = []
    for arm in ("Mlic", "MlicNeff"):
        root = study/arm/"root"
        pin = sha(root/"wide/frozen.json")
        inputs[str(root/"wide/frozen.json")] = pin
        packet = json.loads((root/"datasets/receipt.json").read_text())
        trained = {g["collection"] for g in packet["groups"]}
        if trained != {"pipal","mdid","live-md","cid-iq","cid-oa","irccyn-ivc",
                       "colourlab-rating-variability-2022","bapps"}:
            raise ValueError("Mlic collection inventory differs")
        for fold in (*FOLDS,*DATASET_FOLDS):
            for seed in range(3):
                cell = CELLS/arm/f"without_{fold}_s{seed}"
                rec = json.loads((cell/"byc_record.json").read_text())
                result = json.loads((cell/"result.json").read_text())
                selected = json.loads((cell/"DATASET_GROUP_SELECTION.json").read_text())
                if (rec["fold"] != fold or rec["seed"] != seed or rec["spec"] != spec_of("mono:thr:mp32")
                        or result["selection"]["selected_epoch"] != 119 or result["epochs"] != 120
                        or result["pairs_per_epoch"] != 50000 or result["seed_index"] != seed
                        or result["frozen_sha256"] != pin or selected["root_sha256"] != pin
                        or set(selected["retained_collections"]) != trained-set(DATASET_FOLDS.get(fold,()))
                        or rec["trainer_sha256"] != baseline["trainer_sha256"]
                        or rec["last_bin_sha256"] != sha(cell/"refit/last.bin")):
                    raise ValueError(f"native cell binding differs: {cell}")
                predictions = {f"pred_{fold}.tsv":rec["pred_sha256"]} if fold in FOLDS else {
                    f"pred_dataset_{c}.tsv":v["prediction_sha256"] for c,v in rec["dataset_train_statistics"].items()}
                for relative, expected in predictions.items():
                    if sha(cell/relative) != expected:
                        raise ValueError("cell prediction changed")
                inputs[str(cell/"byc_record.json")] = sha(cell/"byc_record.json")
                inputs[str(cell/"result.json")] = sha(cell/"result.json")
                inventory.append(dict(arm=arm,fold=fold,seed=seed,model_sha256=rec["last_bin_sha256"]))
    mixtures = {}
    for name in ("mixture","mixture_global"):
        mixtures[name] = report(study/name)
        inputs[str(study/name/"manifest.json")] = sha(study/name/"manifest.json")
        for cell in inventory:
            key = f'{cell["arm"]}_{cell["fold"]}_s{cell["seed"]}'
            path = study/name/f"scores_{key}.json"
            model = json.loads(path.read_text())
            if model["bake_sha256"] != cell["model_sha256"] or key not in mixtures[name]:
                raise ValueError("mixture model binding differs")
            inputs[str(path)] = sha(path)
    arms = {arm:mix_arm_statistics(arm,e21,inputs) for arm in ("Mlic","MlicNeff")}
    arms["MlicNeff"]["eligible_under_new_leg_licence_policy"] = True
    inversions = [mixtures["mixture_global"][f'MlicNeff_{c["fold"]}_s{c["seed"]}']["inversions"]
                  for c in inventory if c["arm"] == "MlicNeff"]
    value = dict(schema="mlic-effective-sample-TRAIN-selection-v1",rule_commit="89d3e738af4e",
                 eval_labels_used=False,arms=arms,mixtures=mixtures,cells=inventory,input_sha256=inputs,
                 **neff_decision(arms,inversions))
    output.write_text(json.dumps(value,indent=1)+"\n")
    print(json.dumps({k:v for k,v in value.items() if k not in ("arms","mixtures","cells","input_sha256")},indent=1))
    return 0


def cmd_mix_freeze(args):
    """Fit or reuse the chosen seed-0 models, then freeze their exact bytes."""
    from subjective_training_leg import log_dataset_read
    study = Path(args.study)
    selection_path = study / "TRAIN_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    if selection["schema"] != "dataset-mix-TRAIN-selection-v1" or selection["eval_labels_used"]:
        raise ValueError("preregistered TRAIN-only selection required")
    output = study / "FINALISTS.json"
    if output.exists():
        raise ValueError("preserve the frozen finalist models")
    for path, digest in selection["input_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("TRAIN selection input changed")
    models = {}
    for arm in selection["finalists"]:
        root = study / arm / "root"
        if arm == "Mdata":
            root = Path("/var/tmp/datalegs-20261025/rev7data/root")
            model = Path("/var/tmp/datalegs-20261025/rev7data/full/refit/production-f16.bin")
            if sha(model) != "9fe1aa47a82730550ed1b9dd73addeacf31304485a12213c4e3e9f151bff5117":
                raise ValueError("frozen Mdata model changed")
        elif arm in ("Mmos", "Mpair"):
            dest = CELLS / arm / "without_full_s0"
            fit = json.loads((dest / "result.json").read_text())
            bake = dest / "refit/last.bin"
            if (fit["selected_bake_sha256"] != sha(bake) or fit["seed_index"] != 0
                    or fit["frozen_sha256"] != sha(root/"wide/frozen.json")
                    or (fit["epochs"],fit["pairs_per_epoch"],fit["selection"]["selected_epoch"]) != (120,50000,119)):
                raise ValueError("subset full-fit budget/seed/model binding differs")
            receipt = json.loads((root / "wide/main/real/receipt.json").read_text())
            anchor = root / receipt["legs"]["cid22"]["fit"]["rel"]
            if sha(anchor) != receipt["legs"]["cid22"]["fit"]["sha256"]:
                raise ValueError("TRAIN packing anchor changed")
            log_dataset_read(dest, anchor, receipt["legs"]["cid22"]["fit"]["rows"],
                             ["human_score"], "TRAIN-production-packing", table_sha256=sha(anchor))
            packed_receipt = dest/"PACKED_FINALIST.json"
            if packed_receipt.exists():
                packed = json.loads(packed_receipt.read_text())
                if sha(Path(packed["packed_model"])) != packed["packed_model_sha256"]:
                    raise ValueError("completed finalist pack changed")
            else:
                if (dest/"refit/production-f16.bin").exists():
                    raise ValueError("preserve and inspect the incomplete finalist pack")
                from v2_production_pack import pack_production
                packed = pack_production(bake, dest, anchor, e33_output_stage=True)
                packed_receipt.write_text(json.dumps(packed,indent=1)+"\n")
            model = Path(packed["packed_model"])
        elif arm in ("Mw025", "Mw100", "Mlic"):
            dest = study / arm / "full_s0"
            if dest.exists() and not (dest/"result.json").is_file():
                raise ValueError("inspect and preserve the earlier finalist-fit attempt")
            command = [sys.executable,str(HERE/"v2_confirm_fit.py"),"--root",str(root),
                "--spec",spec_of("mono:thr:mp32"),"--columns",COLS.read_text().strip(),"--head","N",
                "--seed-index","0","--strict-admission","--data-role-decision",str(root/"human_role_decision.json"),
                "--dest",str(dest),"--train-only","--pack-production","--e33-output-stage"]
            if not (dest/"result.json").is_file():
                with (study/f"full_{arm}_s0.log").open("w") as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            fit = json.loads((dest/"result.json").read_text())
            model = Path(fit["packed_model"])
            if (sha(model) != fit["packed_model_sha256"] or fit["seed_index"] != 0
                    or fit["spec"] != spec_of("mono:thr:mp32")
                    or fit["frozen_sha256"] != sha(root/"wide/frozen.json")
                    or (fit["epochs"],fit["pairs_per_epoch"],fit["selection"]["selected_epoch"]) != (120,50000,119)):
                raise ValueError("packed model binding differs")
        else:
            raise ValueError("unregistered mix finalist")
        with (study/f"certify_{arm}.log").open("w") as log:
            subprocess.run([str(Path(os.environ["REV4_V2_BIN_DIR"])/"bake_dial_refit"),
                            "certify-monotone","--bake",str(model)],stdout=log,stderr=subprocess.STDOUT,check=True)
        models[arm] = dict(path=str(model),sha256=sha(model),root=str(root),
                           frozen_root_sha256=sha(root/"wide/frozen.json"),seed_index=0,
                           eligible_under_new_leg_licence_policy=arm == "Mlic")
    value = dict(schema="dataset-mix-finalists-v1",TRAIN_selection_sha256=sha(selection_path),models=models,
                 eval_assessments_per_finalist=1,selection_uses_EVAL=False)
    output.write_text(json.dumps(value,indent=1)+"\n")
    print(json.dumps(value,indent=1))
    return 0


def cmd_mix_summary(args):
    """Collect the existing owners' results and input hashes for the durable record."""
    study = Path(args.study)
    output = study/"NUMERIC_SUMMARY.json"
    if output.exists():
        raise ValueError("preserve the closed numeric summary")
    inputs = {}
    def load(relative):
        path = study/relative
        inputs[relative] = sha(path)
        return json.loads(path.read_text())
    selection = load("TRAIN_SELECTION.json")
    frozen = load("FINALISTS.json")
    if frozen["TRAIN_selection_sha256"] != inputs["TRAIN_SELECTION.json"]:
        raise ValueError("finalist decision changed")
    result = dict(schema="dataset-mix-numeric-summary-v1", selection=selection, finalists=frozen,
                  native_controls=load("NATIVE_CELL_CONTROLS.json"), e21=load("scorecard/e21.json"),
                  panel=load("scorecard/panel.json"), type_bias=load("scorecard/type_bias.json"),
                  knob_inversions=load("scorecard/knobinv.json"),
                  forced_choice_controls=load("FORCED_CHOICE_CONTROLS.json"),
                  mixture_cell_controls=load("MIXTURE_CELL_CONTROLS.json"),
                  mixtures={}, all_cell_mixtures={}, eval={}, eval_paired={})
    if result["forced_choice_controls"]["comparisons"] != 54:
        raise ValueError("native hard-agreement controls require all fifty-four comparisons")
    for name in ("mixture","mixture_global"):
        result["mixtures"][name] = load(f"scorecard/{name}.json")
        result["all_cell_mixtures"][name] = load(f"scorecard/{name}_all-cells.json")
        count = result["native_controls"]["verified_cells"]
        if (len(result["all_cell_mixtures"][name]) != count
                or result["mixture_cell_controls"][name]["models"] != count):
            raise ValueError("all-cell mixture coverage differs from native model inventory")
        if any(v["cells"] != 12 for v in result["mixtures"][name].values()):
            raise ValueError("mixtures require all twelve E21 cells per arm")
    for directory in ("eval","eval-bapps"):
        result["eval"][directory] = load(f"{directory}/report.json")
    for name in frozen["models"]:
        result["eval_paired"][name] = load(f"eval/paired-{name}/report/report.json")
    result["input_sha256"] = inputs
    output.write_text(json.dumps(result,indent=1)+"\n")
    print(json.dumps(dict(path=str(output),sha256=sha(output),finalists=list(frozen["models"]),
                         native_cells=result["native_controls"]["verified_cells"])))
    return 0


def cmd_seedens(args) -> int:
    """E21 of ensembles that average held-out predictions, against the control averaged the same way. Each arm is a
    `+`-joined list of cell arms (`M64` alone, or `M64+M+T`); per fold the members' predictions over the given seeds
    are averaged into one prediction, and the control's (`--control` from the E34 panel) over the same seeds. The
    ensemble needs every member's features and one model evaluation each, so on one feature set it costs no extra
    extraction. Reported: the ensemble's E21 delta (mean over the four datasets of the oriented SROCC difference) and,
    beside it, the members' ordinary per-seed delta for reference. Seed-unit SEs do not apply to an across-seed
    mean."""
    import numpy as np
    import pandas as pd
    panel = read_train_panel(E34_PANEL)
    data = {d["name"]: d for d in panel["datasets"]}
    seeds = [int(x) for x in args.seeds.split(",")]

    def preds(arm, f, s):
        if arm in data[f]["predictions"]:
            return np.asarray(data[f]["predictions"][arm][s], dtype=float)
        t = pd.read_csv(CELLS / arm / f"without_{f}_s{s}" / f"pred_{f}.tsv", sep="\t")
        return t.pred.to_numpy(dtype=float)

    out = {}
    for spec in args.arms.split(","):
        members = spec.split("+")
        per_ds, per_seed_mean = {}, {}
        for f in FOLDS:
            d = data[f]
            target = np.asarray(d["target"], dtype=float)
            orient = float(d["orientation"])
            ens = np.mean([preds(m, f, s) for m in members for s in seeds], axis=0)
            ctl = np.mean([preds(args.control, f, s) for s in seeds], axis=0)
            per_ds[f] = orient * srocc(ens, target) - orient * srocc(ctl, target)
            per_seed_mean[f] = float(np.mean([orient * srocc(np.mean([preds(m, f, s) for m in members], axis=0), target)
                                              - orient * srocc(preds(args.control, f, s), target) for s in seeds]))
        out[spec] = {"control": args.control, "seeds": seeds, "members": members,
                     "ensemble_delta": float(np.mean(list(per_ds.values()))), "ensemble_per_dataset": per_ds,
                     "per_seed_delta": float(np.mean(list(per_seed_mean.values()))), "per_seed_per_dataset": per_seed_mean}
    print(json.dumps(out, indent=1))
    return 0


E34_CONTROL_CELLS = Path("/var/tmp/rev4-featpot/e34-controlc-results/cells/sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1__N")
DEV_RE = None


def dev_points(log: Path) -> dict:
    """{epoch: {group: srocc}} for the logged development groups of one train.log."""
    import re
    global DEV_RE
    DEV_RE = DEV_RE or re.compile(r"(\w+_development): srocc=([0-9.]+)")
    out = {}
    for line in log.read_text().splitlines():
        if line.startswith("  epoch "):
            epoch = int(line.split()[1])
            out[epoch] = {g: float(v) for g, v in DEV_RE.findall(line)}
    return out


def cmd_curves(args) -> int:
    """TRAIN-internal development SROCC (teacher and human dev groups) at epochs 0 and 119, arm vs E34 control."""
    import numpy as np
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for f in FOLDS:
        for s in seeds:
            cell = f"without_{f}_s{s}"
            a, c = dev_points(CELLS / args.arm / cell / "train.log"), dev_points(E34_CONTROL_CELLS / cell / "train.log")
            rows.append({g: (a[0][g], a[119][g], c[0][g], c[119][g]) for g in a[119]})
    summary = {}
    for g in rows[0]:
        v = np.array([r[g] for r in rows])
        summary[g] = {"arm_e0": v[:, 0].mean(), "arm_e119": v[:, 1].mean(), "control_e0": v[:, 2].mean(),
                      "control_e119": v[:, 3].mean(), "arm_minus_control_e119": (v[:, 1] - v[:, 3]).mean(),
                      "arm_change_e0_to_e119": (v[:, 1] - v[:, 0]).mean(),
                      "control_change_e0_to_e119": (v[:, 3] - v[:, 2]).mean(), "n_cells": len(v)}
    print(json.dumps({args.arm: summary}, indent=1))
    return 0


def _preds(d: dict, arm: str, seed: int):
    import numpy as np
    import pandas as pd
    if arm in d["predictions"]:
        return np.asarray(d["predictions"][arm][seed], dtype=float)
    f = d["name"]
    return pd.read_csv(CELLS / arm / f"without_{f}_s{seed}" / f"pred_{f}.tsv", sep="\t").pred.to_numpy(dtype=float)


def cmd_pairclass(args) -> int:
    """Pair-class concordance (E34 section 4.6's classes; an independent numpy reference, not the registered owner):
    same ladder (same reference and distortion type), cross-type (same reference, other type), cross-source (other
    reference). Concordance = share of target-ordered pairs the model orders alike (prediction ties count half); exact
    target ties excluded. Datasets above 4,000 rows use every within-reference pair plus a fixed-seed 2M cross-source
    sample."""
    import numpy as np
    panel = read_train_panel(E34_PANEL)
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    rng = np.random.default_rng(20261022)
    out = {}
    for d in panel["datasets"]:
        t, ref, cod = (np.asarray(d[k]) for k in ("target", "reference", "codec"))
        n = len(t)
        if n < 4000:
            ii, jj = np.triu_indices(n, 1)
        else:
            parts_i, parts_j = [], []
            for r in np.unique(ref):
                idx = np.where(ref == r)[0]
                a, b = np.triu_indices(len(idx), 1)
                parts_i.append(idx[a])
                parts_j.append(idx[b])
            si, sj = rng.integers(0, n, 2_000_000), rng.integers(0, n, 2_000_000)
            keep = ref[si] != ref[sj]
            ii, jj = np.concatenate(parts_i + [si[keep]]), np.concatenate(parts_j + [sj[keep]])
        same_ref, same_type = ref[ii] == ref[jj], cod[ii] == cod[jj]
        classes = {"same_ladder": same_ref & same_type, "cross_type": same_ref & ~same_type,
                   "cross_source": ~same_ref}
        for cname, cm in classes.items():
            dt = t[ii[cm]] - t[jj[cm]]
            ok = dt != 0
            row = {"pairs": int(ok.sum())}
            for arm in arms:
                vals = []
                for s in seeds:
                    p = _preds(d, arm, s)
                    sg = np.sign(dt[ok]) * np.sign(p[ii[cm]][ok] - p[jj[cm]][ok])
                    vals.append(float((np.sum(sg > 0) + 0.5 * np.sum(sg == 0)) / ok.sum()))
                row[arm] = float(np.mean(vals))
            out.setdefault(d["name"], {})[cname] = row
    print(json.dumps(out, indent=1))
    return 0


def cmd_bytype(args) -> int:
    """Per distortion type: mean within-reference SROCC (ladder ordering) and pooled SROCC, arm vs control."""
    import numpy as np
    panel = read_train_panel(E34_PANEL)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {}
    for d in panel["datasets"]:
        t, ref, cod = (np.asarray(d[k]) for k in ("target", "reference", "codec"))
        for typ in sorted(set(cod.tolist())):
            m = cod == typ
            acc = {"within": [], "pooled": []}
            for s in seeds:
                pa, pc = _preds(d, args.arm, s), _preds(d, args.control, s)
                wa, wc = [], []
                for r in set(ref[m].tolist()):
                    mm = m & (ref == r)
                    if mm.sum() >= 3 and np.ptp(t[mm]) > 0:
                        wa.append(srocc(pa[mm], t[mm]))
                        wc.append(srocc(pc[mm], t[mm]))
                acc["within"].append(np.nanmean(wa) - np.nanmean(wc) if wa else np.nan)
                acc["pooled"].append(srocc(pa[m], t[m]) - srocc(pc[m], t[m]))
            out.setdefault(d["name"], {})[str(typ)] = {"n": int(m.sum()),
                                                       "within_ref_delta": float(np.nanmean(acc["within"])),
                                                       "pooled_delta": float(np.nanmean(acc["pooled"]))}
    print(json.dumps(out, indent=1))
    return 0


BANK = Path("/var/tmp/rev5-featbank")


def _knobs(name: str) -> list:
    """Per E21 row of `name`: `(ladder, knob, sign)`, `sign` +1 when a larger knob means better quality. A ladder is one
    reference and distortion type with levels (KADID, TID2013, KonFiG; a larger level is worse), or one reference,
    codec and fixed other setting with quality `q` (CID22-A; larger is better, except aom's quantizer). Joined on the
    table's `pair_key` to the bank keys of its `member_set` (`ref_group`, `codec`, `knob`); no label is read.
    Directions were checked without labels: KonFiG's level raises pixel MSE in every ladder, CID22-A's `q` raises the
    encoded size except aom's (benchmarks/byconstruction_2026-10-22.md, "Ladder inversions that people and the knob
    agree on")."""
    import re
    import pyarrow.parquet as pq
    keys = pq.read_table(ROOT / "wide/main/real" / f"{name}.keys.parquet",
                         columns=["pair_key", "member_set", "ref_basename"]).to_pylist()
    bank = {}
    for ms in sorted({k["member_set"] for k in keys}):
        for r in pq.read_table(BANK / ms / "keys.parquet", columns=["pair_key", "ref_group", "codec", "knob"]).to_pylist():
            bank[(ms, r["pair_key"])] = r
    out = []
    for k in keys:
        b = bank[(k["member_set"], k["pair_key"])]
        if b["ref_group"] != k["ref_basename"]:
            raise ValueError(f"{name}: key reference {k['ref_basename']} against bank {b['ref_group']}")
        if name == "cid22_a25":
            m = re.fullmatch(r"(?:([se]\d+)_)?q([0-9.]+)", b["knob"])
            out.append(((b["ref_group"], b["codec"], m.group(1) or ""), float(m.group(2)),
                        -1 if b["codec"] == "aom" else 1))
        else:
            out.append(((b["ref_group"], b["codec"]), int(b["knob"]), -1))
    return out


def cmd_knobinv(args) -> int:
    """Ladder ordering against the encoder knob as well as people. Over pairs of one ladder with different knob values,
    a pair is `concurring` when the human order agrees with the knob's; the model's inversions are counted on those
    pairs only, each with its size on the human scale (the panel's logistic map of the model, as a share of the label
    span p1-p99). Pairs where people disagree with the knob are reported apart: how often the model follows the knob.
    `ladder_rms` is the error amount inside ladders: the logistic-map residual with each ladder's mean removed, RMS over
    rows, as a share of the span. Out of sample, seeds averaged."""
    import itertools
    import numpy as np
    sys.path.insert(0, str(HERE.parent))
    os.environ.setdefault("ZEN_PANEL_BIN", str(Path(os.environ["REV4_V2_BIN_DIR"]) / "panel"))
    from lib.zen_stats import logistic_fit
    panel = read_train_panel(E34_PANEL)
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = ["base-c", *[a for a in args.arms.split(",") if a and a != "base-c"]]
    out = {}
    for d in panel["datasets"]:
        name = d["name"]
        t = np.asarray(d["target"], dtype=float) * float(d["orientation"])
        knobs = _knobs(name)
        if len(knobs) != len(t):
            raise ValueError(f"{name}: {len(knobs)} knob rows for {len(t)} targets")
        ladders = {}
        for i, (key, knob, sign) in enumerate(knobs):
            ladders.setdefault(key, []).append((i, knob, sign))
        conc, subj = [], []
        for rows in ladders.values():
            for (i, ki, sg), (j, kj, _) in itertools.combinations(rows, 2):
                if ki == kj or t[i] == t[j]:
                    continue
                good, bad = (i, j) if (ki - kj) * sg > 0 else (j, i)  # the knob's better row first
                (conc if t[good] > t[bad] else subj).append((good, bad))
        conc, subj = np.asarray(conc), np.asarray(subj)
        lad = np.zeros(len(t), dtype=int)
        for n, rows in enumerate(ladders.values()):
            lad[[r[0] for r in rows]] = n
        span = float(np.quantile(t, 0.99) - np.quantile(t, 0.01))
        res = {"ladders": len(ladders), "pairs_concurring": int(len(conc)), "pairs_subjective_only": int(len(subj))}
        for arm in arms:
            acc = {k: [] for k in ("inversion_rate", "size_mean", "size_p90", "size_max", "follows_knob", "ladder_rms")}
            for s in seeds:
                m = np.asarray(logistic_fit(_preds(d, arm, s), t))
                inv = m[conc[:, 0]] < m[conc[:, 1]]
                size = (m[conc[:, 1]] - m[conc[:, 0]])[inv] / span
                acc["inversion_rate"].append(float(inv.mean()))
                acc["size_mean"].append(float(size.mean()) if size.size else 0.0)
                acc["size_p90"].append(float(np.quantile(size, 0.9)) if size.size else 0.0)
                acc["size_max"].append(float(size.max()) if size.size else 0.0)
                acc["follows_knob"].append(float((m[subj[:, 0]] > m[subj[:, 1]]).mean()) if len(subj) else float("nan"))
                r = m - t
                r = r - (np.bincount(lad, r) / np.bincount(lad))[lad]
                acc["ladder_rms"].append(float(np.sqrt(np.mean(r * r)) / span))
            res[arm] = {k: float(np.mean(v)) for k, v in acc.items()}
        out[name] = res
        line = "  ".join(f"{a} inv {res[a]['inversion_rate']:.4f} (p90 {res[a]['size_p90']:.3f}) rms {res[a]['ladder_rms']:.4f}"
                         for a in arms)
        print(f"== {name}: {len(ladders)} ladders, {len(conc)} concurring / {len(subj)} subjective-only pairs  {line}",
              flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


BEST = Path(os.environ.get("BYC_BEST", str(Path.home() / "tmp/byc/best")))


def cmd_aicscore(args) -> int:
    """AIC2026 TRAIN out-of-fold panel of arms run with --aic-weight: each source fold is predicted by the cell(s) that
    held it out (mean raw prediction when two cells did); pooled global SROCC, mean within-fold SROCC, within source."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(HERE.parent / "byconstruction"))
    from aic_train import panel
    pairs = pd.read_csv(AIC / "pairs_train.tsv", sep="\t")
    out = {}
    for arm in args.arms.split(","):
        by_fold = {}
        for rec_path in sorted((CELLS / arm).glob("without_*_s*/byc_record.json")):
            rec = json.loads(rec_path.read_text())
            if "aic_fold" not in rec:
                raise ValueError(f"{rec_path}: not an AIC arm")
            pr = pd.read_csv(rec_path.parent / f"pred_aic_k{rec['aic_fold']}.tsv", sep="\t").pred.to_numpy(np.float64)
            by_fold.setdefault(rec["aic_fold"], []).append(pr)
        if sorted(by_fold) != list(range(AIC_NFOLDS)):
            raise ValueError(f"{arm}: AIC folds covered {sorted(by_fold)}")
        pred = pd.Series(np.nan, index=pairs.key.astype(str))
        within = []
        for k, preds in by_fold.items():
            keys = pd.read_parquet(AIC / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
            mean = np.mean(preds, axis=0)
            pred.loc[keys] = mean
            within.append(srocc(mean, -pairs.set_index("key").loc[keys].jnd.to_numpy()))
        res = panel(pred.to_numpy(), pairs)
        res["mean_within_fold"] = float(np.mean(within))
        res["cells_per_fold"] = {int(k): len(v) for k, v in sorted(by_fold.items())}
        out[arm] = res
        print(f"{arm:12s} AIC OOF global {res['global']:.4f}  mean within-fold {res['mean_within_fold']:.4f}  "
              f"within-source {res['within_ref_mean']:.4f}  within-ladder {res['within_ladder_mean']:.4f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def codec_rank_bias(pred, q, codec) -> dict:
    """Per codec, the mean of (pooled prediction rank − pooled human rank), ranks as fractions of the set: negative = the
    model rates that codec's images worse than humans do. `spread` is the SD across codecs (equal weight), the
    codec-uniformity statistic: 0 when equal scores mean equal quality for every codec."""
    import numpy as np
    from scipy.stats import rankdata
    n = len(q)
    rp, rq = rankdata(pred) / n, rankdata(q) / n
    per = {str(c): float(np.mean(rp[codec == c] - rq[codec == c])) for c in sorted(set(codec.tolist()))}
    v = np.array(list(per.values()))
    return {"per_codec": per, "spread": float(np.std(v)), "max_abs": float(np.max(np.abs(v)))}


def cmd_aiczero(args) -> int:
    """AIC2026 public_train zero-shot (no AIC fit) of each cell of each arm: the cell's `refit/last.bin` predicts the arm's
    feature table, scored by `aic_train.panel` per cell; also the panel of the cells' mean prediction and its per-codec
    rank bias (`codec_rank_bias`). Arm `base-c` reads E34's stored control cells."""
    import tempfile
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(HERE.parent / "byconstruction"))
    from aic_train import panel
    pairs = pd.read_csv(AIC / "pairs_train.tsv", sep="\t")
    out = {}
    for spec in args.arm:
        arm, table = spec.split("=", 1)
        preds = {}
        cells_dir = E34_CONTROL_CELLS if arm == "base-c" else CELLS / arm
        # Seeds 0-2, the screen's (E34's control has ten).
        for cell in sorted(c for c in cells_dir.glob("without_*_s*") if c.name.endswith(("_s0", "_s1", "_s2"))):
            with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
                preds[cell.name] = predict(cell / "refit" / "last.bin", Path(table), Path(td) / "p.tsv")
            if len(preds[cell.name]) != len(pairs):
                raise ValueError(f"{arm}/{cell.name}: {len(preds[cell.name])} predictions for {len(pairs)} pairs")
        if len(preds) != len(FOLDS) * 3:
            raise ValueError(f"{arm}: {len(preds)} cells")
        per = {c: panel(p, pairs) for c, p in preds.items()}
        out[arm] = {"table": table, "table_sha256": sha(Path(table)),
                    "mean_cell_global": float(np.mean([r["global"] for r in per.values()])),
                    "per_cell_global": {c: r["global"] for c, r in per.items()},
                    "mean_cell_within_ref": float(np.mean([r["within_ref_mean"] for r in per.values()])),
                    "ensemble": panel(np.mean(list(preds.values()), axis=0), pairs),
                    "codec_bias": codec_rank_bias(np.mean(list(preds.values()), axis=0), -pairs.jnd.to_numpy(),
                                                  pairs.codec.to_numpy()),
                    "codec_bias_per_cell_spread": float(np.mean([
                        codec_rank_bias(p, -pairs.jnd.to_numpy(), pairs.codec.to_numpy())["spread"]
                        for p in preds.values()]))}
        print(f"{arm:12s} AIC public_train zero-shot: mean cell global {out[arm]['mean_cell_global']:.4f}, "
              f"within-source {out[arm]['mean_cell_within_ref']:.4f}; mean prediction {out[arm]['ensemble']['global']:.4f}; "
              f"codec bias spread {out[arm]['codec_bias_per_cell_spread']:.4f} per cell",
              flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def cmd_predict_best(args) -> int:
    """Held-out predictions from each cell's `refit/best.bin` (the trainer's best dev-geomean3 epoch: the harness's
    `best_dev` rule) for an arm, or for E34's stored control cells with `--arm base-c`."""
    seeds = [int(s) for s in args.seeds.split(",")]
    for f in FOLDS:
        for s in seeds:
            cell = f"without_{f}_s{s}"
            src = (E34_CONTROL_CELLS if args.arm == "base-c" else CELLS / args.arm) / cell / "refit/best.bin"
            dest = BEST / args.arm / cell
            out = dest / f"pred_{f}.tsv"
            if out.is_file():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            bake = dest / "best.bin"
            bake.write_bytes(src.read_bytes())
            predict(bake, ROOT / "wide/main/real" / f"{f}.parquet", out)
            print(json.dumps({"arm": args.arm, "cell": cell, "best_sha256": sha(bake)}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("progress")
    pr.add_argument("--arm", required=True)
    pr.add_argument("--folds", default=",".join(FOLDS))
    pr.add_argument("--seeds", default="0,1,2")
    pr.add_argument("--run-state", help="recorded worker PID/start ticks, stage and frozen identities")
    pr.add_argument("--output", help="atomically update durable progress evidence")
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, help="recipe tokens, e.g. mono:thr or H1:mono")
    r.add_argument("--folds", default=",".join(FOLDS))
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--jobs", type=int, default=12)
    r.add_argument("--base-spec", help="replace B's spec, e.g. sel:<id>@h32:H128:cv16:cf98:fx1nr")
    r.add_argument("--cols", help="comma-separated columns file for --base-spec")
    r.add_argument("--label", help="arm directory name for --base-spec runs")
    r.add_argument("--aic-weight", type=float, help="add the AIC2026 TRAIN leg (rotating held-out fold) at this weight")
    mx = sub.add_parser("mix-summary")
    mx.add_argument("--study", required=True)
    ma = sub.add_parser("mix-audit")
    ma.add_argument("--study", required=True)
    ms = sub.add_parser("mix-select")
    ms.add_argument("--study", required=True)
    ms.add_argument("--arms", default=",".join(MIX_ARMS))
    ns = sub.add_parser("neff-select")
    ns.add_argument("--study", required=True)
    mf = sub.add_parser("mix-freeze")
    mf.add_argument("--study", required=True)
    z = sub.add_parser("aiczero")
    z.add_argument("--arm", action="append", required=True, help="NAME=TABLE")
    z.add_argument("--out")
    q = sub.add_parser("aicscore")
    q.add_argument("--arms", required=True)
    q.add_argument("--out")
    s = sub.add_parser("score")
    s.add_argument("--arms", required=True)
    s.add_argument("--control", default="base-c")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--rule", choices=("last", "best"), default="last",
                   help="last = registered final epoch; best = refit/best.bin (predict-best first)")
    s.add_argument("--paired-with", help="also report every other arm minus this arm (one of --arms), seed by seed")
    se = sub.add_parser("seedens")
    se.add_argument("--arms", required=True, help="comma-separated ensembles, each a +-joined list of cell arms")
    se.add_argument("--control", default="base-c")
    se.add_argument("--seeds", default="0,1,2")
    c = sub.add_parser("curves")
    c.add_argument("--arm", required=True, help="arm directory name, e.g. mono-thr")
    c.add_argument("--seeds", default="0,1,2")
    pc = sub.add_parser("pairclass")
    pc.add_argument("--arms", required=True, help="comma list; base-c and other E34 arms come from the panel input")
    pc.add_argument("--seeds", default="0,1,2")
    bt = sub.add_parser("bytype")
    bt.add_argument("--arm", required=True)
    bt.add_argument("--control", default="base-c")
    bt.add_argument("--seeds", default="0,1,2")
    kv = sub.add_parser("knobinv")
    kv.add_argument("--arms", required=True, help="comma list of arm directories; base-c is always included")
    kv.add_argument("--seeds", default="0,1,2")
    kv.add_argument("--out")
    pb = sub.add_parser("predict-best")
    pb.add_argument("--arm", required=True, help="arm directory name, or base-c for E34's control cells")
    pb.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    return {"run": cmd_run, "score": cmd_score, "seedens": cmd_seedens, "curves": cmd_curves, "pairclass": cmd_pairclass,
            "bytype": cmd_bytype, "knobinv": cmd_knobinv, "predict-best": cmd_predict_best, "aicscore": cmd_aicscore,
            "aiczero": cmd_aiczero, "progress": cmd_progress, "mix-select": cmd_mix_select, "neff-select": cmd_neff_select,
            "mix-freeze": cmd_mix_freeze, "mix-audit": cmd_mix_audit, "mix-summary": cmd_mix_summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
