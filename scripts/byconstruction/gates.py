"""Label-free product gates for a by-construction candidate, through the owners E33/E34 used.

Exploration record: benchmarks/byconstruction_2026-10-22.md. Reads no human label.

  python3 gates.py --name <lowercase-alnum> --model <packed bake> --out <new dir> [--steps nearid,verdict,steer,identity]
  python3 gates.py --summary --out <dir> --name <name>

- NEARID (N1-N3) and the engineering-grid verdicts (C1-C6, G-DIAL) run through `e33_gates` with E34's evaluation
  binaries (`bake_verdict`, `serve_custom_bake`).
- G-STEER runs E34's pinned `dev96-instrument-83b13003` on E34's 135-case served packet (D12 exact gradients), exactly as
  `e34_evaluate.steering` did for the served objective.
- Identity (C5 at every SIMD tier) runs `e33_gates.identity` with the candidate as the only model.
E34's `base-c` results under `/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation` are the comparison.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rev4_featpot"))
E34 = Path("/mnt/v/output/zensim/e34-launch-2026-10-15")
STEER_BIN = E34 / "bin/dev96-instrument-83b13003"
STEER_BIN_SHA = "9478ca6fc7a94b937cdcde57764668aec33ff04adfdcf770b3ae0918dce36d9f"
STEER_JOB = Path("/mnt/v/output/zensim/e34-prep-2026-10-15/part6/cached-gate-a/job.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def steer(name, model, out, steer_bin=None, formula_rev="5"):
    if steer_bin is None and sha(STEER_BIN) != STEER_BIN_SHA:
        raise ValueError("steer instrument changed")
    steer_bin = steer_bin or STEER_BIN
    dest = out / "steer135" / name
    dest.mkdir(parents=True, exist_ok=False)
    job = json.loads(STEER_JOB.read_text())
    digest = sha(model)
    for case in job["cases"]:
        case.update(model=str(model), model_sha256=digest)
    job["linear_probe"] = False
    job["output"] = str(dest / "result.json")
    (dest / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    env = dict(os.environ, ZENSIM_FORMULA_REV=formula_rev, ZENSIM_NEIGHBOUR_EXACT="1", ZENSIM_MAX_TIER="v3",
               RAYON_NUM_THREADS="1", STEERFIX_PACKET=str(dest / "job.json"))
    with (dest / "run.log").open("x") as log:
        subprocess.run([str(steer_bin), "metric::bake::steerfix_packet::engineering_packet", "--exact", "--nocapture"],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    rows = json.loads((dest / "result.json").read_text())["rows"]
    if len(rows) != 135 or not all(r["model"] == digest and r["sensitivity_method"] == "exact-gradient-d12" for r in rows):
        raise ValueError("unexpected steer rows")
    print(f"steer135 {name}: {sum(r['pass'] for r in rows)}/135", flush=True)


def run(a):
    import e33_gates as owner
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    owner.GATES = out
    owner.BIN = Path(a.bin_dir) if a.bin_dir else E34 / "evaluation/bin"
    owner.FORMULA_REV = a.formula_rev
    if a.instruments:
        owner.INSTRUMENTS = Path(a.instruments)
    model = Path(a.model).resolve()
    owner.MODELS_OVERRIDE = {a.name: (model, sha(model), None)}
    steps = a.steps.split(",")
    if "nearid" in steps:
        owner.nearid(a.name, 0)
    if "verdict" in steps:
        for grid in ("standard", "ladder"):
            owner.verdict(a.name, 0, grid)
    if "steer" in steps:
        steer(a.name, model, out, Path(a.bin_dir) / "steer_instrument" if a.bin_dir else None, a.formula_rev)
    if "identity" in steps:
        owner.identity(0)
    return 0


def checks(path):
    g = json.loads(Path(path).read_text())
    return {c["id"]: (c["state"], c["measured"]) for c in g["checks"]}, g.get("contract"), g.get("shippable")


def summary(a):
    out, ev = Path(a.out), E34 / "evaluation"
    res = {}
    for label, root, cand in ((a.name, out, a.name), ("base-c", ev / "gates", "base-c")):
        r = {}
        near = root / "nearid-s0" / f"candidate-{cand}.GATES.json"
        if near.is_file():
            n = json.loads(near.read_text())
            r["nearid"] = {k: n[k]["pass"] for k in ("N1", "N2", "N3")} | {"monotone_ladders": n["N3"]["monotone_ladders"]}
        for grid in ("standard", "ladder"):
            p = root / f"verdict-s0-{cand}" / grid / "gaddr.json"
            if p.is_file():
                c, contract, ship = checks(p)
                r[grid] = {k: v for k, v in c.items() if k.startswith(("A", "C", "G"))} | {"contract": contract}
        steer_dir = (out / "steer135" / a.name) if label == a.name else (ev / "steer135" / "base-c-s0")
        p = steer_dir / "result.json"
        if p.is_file():
            rows = json.loads(p.read_text())["rows"]
            r["gsteer"] = f"{sum(x['pass'] for x in rows)}/135"
        res[label] = r
    print(json.dumps(res, indent=1, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="nearid,verdict,steer,identity")
    # Rebuilt gate tools (bake_verdict, serve_custom_bake, steer_instrument) instead of E34's pinned binaries, the
    # instrument tables of another extraction, and the process formula revision (the revision-7 models: 7).
    ap.add_argument("--bin-dir")
    ap.add_argument("--instruments")
    ap.add_argument("--formula-rev", default="5")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    return summary(a) if a.summary else run(a)


if __name__ == "__main__":
    sys.exit(main())
