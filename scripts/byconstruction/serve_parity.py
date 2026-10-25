"""Served (pixel route) against table route for a by-construction research bake at formula revision 6.

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Serving the candidate"). The research bakes read the
ringhpm ringing form and the globaff descriptor; their table-route scores (`mixture_table.py score --decl`, through
`byc_predict`) come from feature tables extracted under `ZENSIM_RING_FORM=hpm` plus the research globaff sidecar. This
stamps a copy of the bake as formula revision 6, serves every mixture image through `score_pair_with_bake --bake-post
raw` with no ring-form environment, and compares the two routes image by image. No labels are read.

  python3 serve_parity.py --bin-dir DIR --mixture MIXDIR --bake NAME=PATH [--bake ...] --out OUT.json [--jobs 8]
  python3 serve_parity.py --bin-dir DIR --pairs PAIRS.tsv --pred PRED.tsv --bake NAME=PATH --out OUT.json [--jobs 12]

MIXDIR holds `manifest.json`, `img/` and the table-route `scores_<NAME>.json` of each bake. The second form serves the
pairs of PAIRS.tsv (`key`, `ref_path`, `dist_path`) and compares them row by row with the `pred` column of PRED.tsv, the
bake's table-route predictions for the same rows in the same order (e.g. `byc_predict` on public_train).
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def stamp(bin_dir: Path, src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run([str(bin_dir / "bake_stamp_revision"), str(src), "6", str(dest)], check=True,
                       capture_output=True, text=True)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--mixture")
    ap.add_argument("--pairs")
    ap.add_argument("--pred")
    ap.add_argument("--bake", action="append", required=True, help="NAME=PATH (table scores in MIXDIR/scores_NAME.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if k not in ("ZENSIM_RING_FORM", "ZENSIM_FORMULA_REV")}
    if a.pairs:
        return pairs_parity(a, env)
    if not a.mixture:
        raise SystemExit("--mixture or --pairs/--pred required")
    bin_dir, mix = Path(a.bin_dir), Path(a.mixture)
    man = json.loads((mix / "manifest.json").read_text())
    images = sorted({(r["ref"], r["base"]) for r in man["rows"]} | {(r["ref"], r["more"]) for r in man["rows"]})
    out = {}
    for spec in a.bake:
        name, path = spec.split("=", 1)
        table = json.loads((mix / f"scores_{name}.json").read_text())
        if table.get("route") != "table":
            raise SystemExit(f"{name}: scores file is not a table-route record")
        if table["bake_sha256"] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise SystemExit(f"{name}: table scores were made with a different bake")
        stamped = Path(a.out).with_suffix("") / f"{name}_rev6.bin"
        stamped_sha = stamp(bin_dir, Path(path), stamped)

        def one(pair, stamped=stamped):
            ref, dist = pair
            r = subprocess.run([str(bin_dir / "score_pair_with_bake"), "--bake", str(stamped), "--bake-post", "raw",
                                "--ref", str(mix / "img" / ref), "--dist", str(mix / "img" / dist)],
                               capture_output=True, text=True, check=True, env=env)
            return dist, float(r.stdout.strip().splitlines()[-1])

        with cf.ThreadPoolExecutor(a.jobs) as pool:
            served = dict(pool.map(one, images))
        keys = sorted(served)
        s = np.array([served[k] for k in keys])
        t = np.array([table["scores"][k] for k in keys])
        d = np.abs(s - t)
        out[name] = {"bake": path, "stamped": str(stamped), "stamped_sha256": stamped_sha, "n": len(keys),
                     "max_abs_diff": float(d.max()), "p99_abs_diff": float(np.quantile(d, 0.99)),
                     "mean_abs_diff": float(d.mean()), "n_over_1e-3": int((d > 1e-3).sum()),
                     "worst": keys[int(d.argmax())], "table_rounding": 1e-6}
        print(name, json.dumps(out[name]))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def pairs_parity(a, env) -> int:
    import pandas as pd
    if len(a.bake) != 1 or not a.pred:
        raise SystemExit("--pairs needs --pred and exactly one --bake")
    bin_dir = Path(a.bin_dir)
    name, path = a.bake[0].split("=", 1)
    p = pd.read_csv(a.pairs, sep="\t", usecols=["key", "ref_path", "dist_path"])
    pred = pd.read_csv(a.pred, sep="\t")
    if len(p) != len(pred):
        raise SystemExit(f"{len(p)} pairs against {len(pred)} predictions")
    stamped = Path(a.out).with_suffix("") / f"{name}_rev6.bin"
    stamped_sha = stamp(bin_dir, Path(path), stamped)

    def one(i):
        r = subprocess.run([str(bin_dir / "score_pair_with_bake"), "--bake", str(stamped), "--bake-post", "raw",
                            "--ref", p.ref_path[i], "--dist", p.dist_path[i]],
                           capture_output=True, text=True, check=True, env=env)
        return float(r.stdout.strip().splitlines()[-1])

    with cf.ThreadPoolExecutor(a.jobs) as pool:
        served = np.array(list(pool.map(one, range(len(p)))))
    d = np.abs(served - pred["pred"].to_numpy())
    out = {"bake": path, "bake_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "stamped": str(stamped),
           "stamped_sha256": stamped_sha, "pairs": a.pairs, "pred": a.pred, "n": len(p),
           "max_abs_diff": float(d.max()), "p99_abs_diff": float(np.quantile(d, 0.99)), "mean_abs_diff": float(d.mean()),
           "n_over_1e-3": int((d > 1e-3).sum()), "worst_key": str(p.key[int(d.argmax())])}
    print(name, json.dumps(out))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
