"""Reference-held-out folds on AIC2026 TRAIN through the Rust trainer (research replay; never ship).

Exploration record: benchmarks/byconstruction_2026-10-22.md, section "AIC2026 TRAIN". The fold tables come from
`aic_train.py`'s fold assignment (seed 1000): OUT/folds_train/ holds WG1's public_train split with its JND_score labels
in 7 folds (the default); OUT/folds/ the earlier 42-reference JND_curve pool in 6 folds (AIC_FOLDS, AIC_NFOLDS=6). For each fold the
trainer fits the arm on the fit references only, plus any extra groups given (the E21 legs of a by-construction root),
and `byc_predict` scores the held-out references with the exported bake. The out-of-fold panel is aic_train.panel.

  run   --arm NAME --flags "TRAINER FLAGS" [--group SPEC ...] [--aic-weight W] [--epochs 60] [--pairs 20000] [--jobs 6]
  score --arm NAME [--arm ...]

Every bake is a research fit: `--historical-replay` records that it cannot qualify a model.
"""

import argparse
import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aic_train  # noqa: E402

OUT = Path(os.environ.get("AIC_OUT", "/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train"))
BIN = Path(os.environ.get("AIC_BIN", "/var/tmp/byconstruction-bin/58a4d100ad36"))
# Fold tables: AIC_FOLDS (default OUT/folds_train: WG1 public_train, JND_score, 7 folds) with its pairs list.
FOLDS = Path(os.environ.get("AIC_FOLDS", str(OUT / "folds_train")))
NFOLDS = int(os.environ.get("AIC_NFOLDS", "7"))
PAIRS = Path(os.environ.get("AIC_PAIRS", str(FOLDS / "pairs_train.tsv")))
DECL = HERE.parent / "rev4_featpot" / "byc_fx1g_declaration.json"
REPLAY = "byconstruction AIC2026 TRAIN research fit (owner goal 2026-10-23); reference-held-out folds; never ship"


def keep_file() -> Path:
    path = OUT / "keep_fx1g.txt"
    if not path.is_file():
        path.write_text("\n".join(map(str, json.loads(DECL.read_text())["direct"])) + "\n")
    return path


def run_fold(a, k: int) -> dict:
    dest = OUT / "rust" / FOLDS.name / a.arm / f"k{k}"
    rec_path = dest / "record.json"
    if rec_path.is_file():
        return json.loads(rec_path.read_text())
    if dest.exists():
        raise ValueError(f"{dest}: incomplete earlier attempt; inspect and move it aside")
    dest.mkdir(parents=True)
    groups = [f"aic:{FOLDS}/aic_k{k}_fit.parquet:{a.aic_weight!r}:1:{a.aic_mode}", *a.group]
    cmd = [str(BIN / "zensim_mlp_train")]
    for g in groups:
        cmd += ["--group", g]
    cmd += ["--target-column", "human_score", "--target-scale", "1", "--hidden", str(a.hidden), "--epochs", str(a.epochs),
            "--pairs-per-epoch", str(a.pairs), "--init-seed", str(1101 + a.seed), "--sample-seed", str(101 + a.seed),
            "--pair-sampling", "uniform", "--max-features", "1891", "--keep-features", str(keep_file()),
            "--early-stop-patience", "0", "--val-policy", "mean", "--val-aggregate", "geomean3", "--out-dtype", "f32",
            "--log-every", str(a.log_every), "--no-auto-eval", "--historical-replay", REPLAY,
            "--out", str(dest / "best.bin"), "--derived-inputs", str(DECL), *shlex.split(a.flags)]
    t0 = time.time()
    with (dest / "train.log").open("w") as log:
        log.write("$ " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=dest,
                              env=dict(os.environ, ZENSIM_MAX_TIER="v3", RAYON_NUM_THREADS="1"))
    if proc.returncode:
        raise RuntimeError(f"{dest}: trainer rc={proc.returncode}")
    test = FOLDS / f"aic_k{k}_test.parquet"
    subprocess.run([str(BIN / "byc_predict"), "--bake", str(dest / "best.bin"), "--decl", str(DECL), "--corpus", str(test),
                    "--out", str(dest / "pred.tsv")], check=True, capture_output=True)
    rec = {"arm": a.arm, "fold": k, "flags": a.flags, "groups": groups, "train_seconds": round(time.time() - t0, 1),
           "bake_sha256": aic_train_sha(dest / "best.bin"), "trainer_sha256": aic_train_sha(BIN / "zensim_mlp_train")}
    rec_path.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def aic_train_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_run(a) -> int:
    with cf.ThreadPoolExecutor(a.jobs) as pool:
        futs = {pool.submit(run_fold, a, k): k for k in range(NFOLDS)}
        bad = 0
        for f in cf.as_completed(futs):
            try:
                print(json.dumps({"arm": a.arm, "fold": futs[f], "train_s": f.result()["train_seconds"]}), flush=True)
            except Exception as e:  # report every fold
                bad += 1
                print(json.dumps({"arm": a.arm, "fold": futs[f], "error": str(e)[-800:]}), flush=True)
    return 1 if bad else score([a.arm])


def score(arms: list[str]) -> int:
    p = pd.read_csv(PAIRS, sep="\t")
    for arm in arms:
        pred = pd.Series(np.nan, index=p.key.astype(str))
        for k in range(NFOLDS):
            test = pd.read_parquet(FOLDS / f"aic_k{k}_test.parquet", columns=["key"])
            pr = pd.read_csv(OUT / "rust" / FOLDS.name / arm / f"k{k}" / "pred.tsv", sep="\t")
            pred.loc[test.key.astype(str).to_numpy()] = pr.pred.to_numpy(np.float64)
        if pred.isna().any():
            raise ValueError(f"{arm}: missing out-of-fold predictions")
        res = aic_train.panel(pred.to_numpy(), p)
        (OUT / "rust" / FOLDS.name / arm / "oof_panel.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"{arm:24s} OOF global {res['global']:.4f} within-ref {res['within_ref_mean']:.4f} "
              f"within-ladder {res['within_ladder_mean']:.4f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--flags", required=True, help="trainer flags of the arm (e.g. the monotone construction)")
    r.add_argument("--group", action="append", default=[], help="extra trainer group spec (E21 legs)")
    r.add_argument("--aic-weight", type=float, default=1.0)
    r.add_argument("--aic-mode", default="rank", help="AIC group flags: rank (pooled across references) etc.")
    r.add_argument("--hidden", type=int, default=128)
    r.add_argument("--epochs", type=int, default=60)
    r.add_argument("--pairs", type=int, default=20000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--log-every", type=int, default=10)
    r.add_argument("--jobs", type=int, default=6)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("score")
    s.add_argument("--arm", action="append", required=True)
    s.set_defaults(fn=lambda a: score(a.arm))
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
