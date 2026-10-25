"""By-construction exploration: run E21 LODO cells for recipe arms and score them against E34's stored control.

Exploration harness only (benchmarks/byconstruction_2026-10-22.md). Each cell is the registered `v2_lodo_mlp.py` strict
training-only route (same tables, seeds, recipe owner) with the arm's recipe tokens appended to B's C-family spec, followed
by a held-out prediction through `v2_lodo_mlp.predict` (dense bake + `bake_dial_refit predict --score-units`). Scoring
pairs every arm with E34's stored `base-c` predictions for the same fold and seed index (E34 `panel-input.json`), which
this harness reproduced bit for bit (kadid seed 0, 7,869 rows).

  run    --arm TOKENS --folds kadid,tid2013,konfig,cid22_a25 --seeds 0,1,2 --jobs N [--aic-weight W --label L]
  score  --arms NAME[,NAME...] [--seeds 0,1,2]
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
import subprocess
import sys
import time
from pathlib import Path

# BYC_ROOT selects a derived root (e.g. the hpm ringing root from scripts/byconstruction/ring_root.py).
ROOT = Path(os.environ.get("BYC_ROOT", "/mnt/v/output/zensim/v40r4-2026-10-08/v2e29"))
BASE_SPEC = "sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1"
E34_PANEL = Path("/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json")
FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")
HERE = Path(__file__).resolve().parent
CELLS = Path(os.environ.get("BYC_CELLS", str(Path.home() / "tmp/byc/cells")))
COLS = Path(os.environ.get("BYC_COLS", str(Path.home() / "tmp/byc/cols_c.txt")))
# v2_common resolves its root and binaries at import; this harness binds both before importing it.
os.environ.setdefault("REV4_V2_ROOT", str(ROOT))
os.environ.setdefault("REV4_V2_BIN_DIR", "/var/tmp/byconstruction-target/release")
sys.path.insert(0, str(HERE))
from v2_common import sha  # noqa: E402
from v2_lodo_mlp import predict, read_curve  # noqa: E402


def spec_of(tokens: str) -> str:
    """B's C-family spec (or `--base-spec`) with the arm's tokens. `H<n>` replaces H128; other tokens are appended."""
    toks = [t for t in tokens.split(":") if t]
    base = BASE_SPEC
    hidden = [t for t in toks if t.startswith("H")]
    if hidden:
        base = base.replace(":H128:", f":{hidden[0]}:")
    rest = [t for t in toks if not t.startswith("H")]
    return base + "".join(f":{t}" for t in rest)


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
    token = next((t for t in spec.split(":") if t in ("fx1g", "fx1gs1", "fx1gg", "fx1o")), None)
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


def run_cell(tokens: str, fold: str, seed: int) -> dict:
    dest = CELLS / arm_name(tokens) / f"without_{fold}_s{seed}"
    record = dest / "byc_record.json"
    if record.is_file():
        return json.loads(record.read_text())
    if dest.exists():
        raise ValueError(f"{dest}: incomplete earlier attempt; inspect and move it aside")
    scratch = Path.home() / "tmp/byc/scratch" / arm_name(tokens) / f"{fold}_s{seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch), ZENSIM_MAX_TIER="v3")
    cmd = [sys.executable, str(HERE / "v2_lodo_mlp.py"), "--root", str(ROOT), "--spec", spec_of(tokens),
           "--columns", COLS.read_text().strip(), "--head", "N", "--heldout", fold, "--seed-index", str(seed),
           "--strict-admission", "--data-role-decision", str(ROOT / "human_role_decision.json"),
           "--dest", str(dest), "--train-only"]
    if AIC_WEIGHT is not None:
        k = aic_fold(fold, seed)
        cmd += ["--research-extra-group", f"aic:{AIC}/aic_k{k}_fit.parquet:{AIC_WEIGHT!r}:0:rank",
                "--research-replay", AIC_REPLAY]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    train_s = time.time() - t0
    if proc.returncode:
        raise RuntimeError(f"{dest}: rc={proc.returncode}\n{proc.stderr[-2000:]}")
    pred = predict_cell(dest / "refit/last.bin", ROOT / "wide/main/real" / f"{fold}.parquet", dest / f"pred_{fold}.tsv",
                        spec_of(tokens))
    rec = {"tokens": tokens, "spec": spec_of(tokens), "fold": fold, "seed": seed, "train_seconds": round(train_s, 1),
           **({} if AIC_WEIGHT is None else aic_predict(dest, fold, seed, spec_of(tokens))),
           "last_bin_sha256": sha(dest / "refit/last.bin"), "pred_sha256": sha(dest / f"pred_{fold}.tsv"),
           "rows": int(len(pred)), "curve": read_curve(dest / "train.log"),
           "bin_dir": os.environ["REV4_V2_BIN_DIR"],
           "trainer_sha256": sha(Path(os.environ["REV4_V2_BIN_DIR"]) / "zensim_mlp_train")}
    for line in (dest / "train.log").read_text().splitlines():
        if line.startswith("  epoch 119 "):
            rec["epoch119"] = line.split("| t=")[0].strip()
    record.write_text(json.dumps(rec, indent=1) + "\n")
    for leftover in scratch.glob("*"):
        leftover.unlink()
    return rec


def cmd_run(args) -> int:
    global BASE_SPEC, COLS, LABEL, AIC_WEIGHT
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


def cmd_score(args) -> int:
    import numpy as np
    import pandas as pd
    panel = json.loads(E34_PANEL.read_text())
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
    print(json.dumps(out, indent=1))
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
    panel = json.loads(E34_PANEL.read_text())
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
    panel = json.loads(E34_PANEL.read_text())
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
    panel = json.loads(E34_PANEL.read_text())
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
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, help="recipe tokens, e.g. mono:thr or H1:mono")
    r.add_argument("--folds", default=",".join(FOLDS))
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--jobs", type=int, default=12)
    r.add_argument("--base-spec", help="replace B's spec, e.g. sel:<id>@h32:H128:cv16:cf98:fx1nr")
    r.add_argument("--cols", help="comma-separated columns file for --base-spec")
    r.add_argument("--label", help="arm directory name for --base-spec runs")
    r.add_argument("--aic-weight", type=float, help="add the AIC2026 TRAIN leg (rotating held-out fold) at this weight")
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
    pb = sub.add_parser("predict-best")
    pb.add_argument("--arm", required=True, help="arm directory name, or base-c for E34's control cells")
    pb.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    return {"run": cmd_run, "score": cmd_score, "seedens": cmd_seedens, "curves": cmd_curves, "pairclass": cmd_pairclass,
            "bytype": cmd_bytype, "predict-best": cmd_predict_best, "aicscore": cmd_aicscore,
            "aiczero": cmd_aiczero}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
