"""Where a by-construction model's distance comes from, by feature cell (scale x channel) and family, label-free.

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Which cells can revision 7 skip"). For a dense bake
(820 fx1 inputs -> ReLU units -> one output) on feature tables of the training root, each row's score drop from the
identity input is split over the inputs with the active units' slopes (exact for a row whose active set equals the
identity's, first order otherwise), and each group's ablation is measured exactly: the group's damage inputs and their
fragility products set to their identity value 0, every other input kept. Reads feature columns only (no label).

  cell_share.py --bake DENSE.bin --table T.parquet [--table ...] [--out OUT.json]
"""
import argparse, collections, json, subprocess, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
INS = "/home/lilith/work/zen/zenanalyze/target/release/zenpredict-inspect"
FX1 = json.load(open(ROOT / "benchmarks/e33_registration_2026-10-09.json"))["fx1"]
INFO = {r[0]: (r[1], r[2], r[3]) for r in FX1["table"]}  # id -> (family, scale, channel)


def load(bake):
    d = json.loads(subprocess.run([INS, bake, "--weights"], capture_output=True, text=True, check=True).stdout)
    L = d["layers"]
    assert len(L) == 2 and L[1]["out_dim"] == 1, "expects one hidden layer and one output"
    W1 = np.array(L[0]["weights"], dtype=np.float64).reshape(L[0]["in_dim"], L[0]["out_dim"])
    b1 = np.array(L[0]["biases"], dtype=np.float64)
    w2 = np.array(L[1]["weights"], dtype=np.float64)
    b2 = float(L[1]["biases"][0])
    meta = {m["key"]: m for m in d["metadata"]}
    der = [l.split() for l in meta["zensim.derived_inputs"]["value_text"].splitlines()[1:]]
    return W1, b1, w2, b2, np.array(d["scaler_mean"]), np.array(d["scaler_scale"]), der


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--table", action="append", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    W1, b1, w2, b2, mean, scale, der = load(a.bake)
    live = np.where(w2 != 0)[0]
    W1, b1, w2 = W1[:, live], b1[live], w2[live]
    # The damage feature each input belongs to: a direct input, or the damage side of a fragility product.
    frag = set(FX1["fragility_ids"])
    owner = []
    for r in der:
        ids = [int(t) for t in r[1:]]
        dmg = [i for i in ids if i not in frag]
        assert len(dmg) == 1, r
        owner.append(dmg[0])
    cell = [(INFO[i][1], INFO[i][2]) for i in owner]
    family = [INFO[i][0] for i in owner]
    need = sorted({int(t) for r in der for t in r[1:]})
    report = {"bake": a.bake, "live_units": int(len(live)), "tables": {}}
    for path in a.table:
        tab = pq.read_table(path, columns=[f"f{i}" for i in need])
        col = {i: tab[f"f{i}"].to_numpy().astype(np.float64) for i in need}
        X = np.stack([col[int(r[1])] if r[0] == "in" else col[int(r[1])] * col[int(r[2])] for r in der], 1)
        Z = lambda x: (x - mean) / scale
        def score(x):
            return b2 + np.maximum(Z(x) @ W1 + b1, 0.0) @ w2
        s_full = score(X)
        s_id = score(np.zeros((1, X.shape[1])))[0]
        act = (Z(X) @ W1 + b1) > 0
        C = (X / scale) * ((act * w2) @ W1.T)  # per-input slope contribution (score units, <= 0 when monotone)
        drop = s_id - s_full
        rep = {"rows": int(len(X)), "identity_score": float(s_id), "mean_drop": float(drop.mean()),
               "median_drop": float(np.median(drop)), "cells": {}, "families": {}}
        groups = collections.defaultdict(list)
        for j, c in enumerate(cell):
            groups[f"s{c[0]}{c[1]}"].append(j)
        for s in range(4):
            groups[f"scale{s}"] = [j for j, c in enumerate(cell) if c[0] == s]
        groups["s0Y+s1Y"] = [j for j, c in enumerate(cell) if (c[0], c[1]) in ((0, "Y"), (1, "Y"))]
        tot = -C.sum()
        for g, js in sorted(groups.items()):
            Xa = X.copy(); Xa[:, js] = 0.0
            d_abl = score(Xa) - s_full  # score rise when the group is removed (>= 0 when monotone)
            rep["cells"][g] = {"inputs": len(js), "slope_share": float(-C[:, js].sum() / tot),
                               "ablation_mean": float(d_abl.mean()), "ablation_p90": float(np.percentile(d_abl, 90)),
                               "ablation_share_of_drop": float(d_abl.mean() / drop.mean())}
        fams = collections.defaultdict(list)
        for j, f in enumerate(family):
            fams[f].append(j)
        for f, js in fams.items():
            rep["families"][f] = float(-C[:, js].sum() / tot)
        report["tables"][Path(path).stem] = rep
        print(f"== {Path(path).stem}: {len(X)} rows, identity {s_id:.3f}, mean drop {drop.mean():.3f}")
        for g in ["s0Y", "s1Y", "s1X", "s1B", "scale0", "scale1", "scale2", "scale3", "s0Y+s1Y"]:
            c = rep["cells"][g]
            print(f"   {g:8s} slope share {c['slope_share']:.3f}  ablation mean {c['ablation_mean']:.3f}"
                  f" ({c['ablation_share_of_drop']:.3f} of the drop), p90 {c['ablation_p90']:.3f}")
        top = sorted(rep["families"].items(), key=lambda kv: -kv[1])[:8]
        print("   families:", ", ".join(f"{k} {v:.3f}" for k, v in top))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
