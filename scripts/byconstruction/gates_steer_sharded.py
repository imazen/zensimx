#!/usr/bin/env python3
"""G-STEER (`gates.py` steer step) with its 135 independent cases split across processes.

Writes the same `steer135/<name>/job.json` (the full packet) and a `result.json` whose rows are the shards' rows in
the packet's case order, with `packet_sha256` = sha256 of that full job.json, as the single-process run writes it.
Each shard runs the same pinned `steer_instrument` with `RAYON_NUM_THREADS=1`, so every case is computed exactly as
before; only the cases run side by side. `gates.py --summary` reads the result unchanged.

    gates_steer_sharded.py --name NAME --model BAKE --out GATE_DIR --bin STEER_INSTRUMENT [--formula-rev 7] [--shards 9]
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

STEER_JOB = Path("/mnt/v/output/zensim/e34-prep-2026-10-15/part6/cached-gate-a/job.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bin", required=True, type=Path)
    ap.add_argument("--formula-rev", default="7")
    ap.add_argument("--shards", type=int, default=9)
    ap.add_argument("--packet", type=Path, default=STEER_JOB,
                    help="explicit relocated copy of the same frozen engineering packet")
    a = ap.parse_args()
    dest = a.out / "steer135" / a.name
    dest.mkdir(parents=True, exist_ok=False)
    # The packet exactly as gates.py builds it.
    job = json.loads(a.packet.read_text())
    digest = sha(a.model)
    for case in job["cases"]:
        case.update(model=str(a.model), model_sha256=digest)
    job["linear_probe"] = False
    job["output"] = str(dest / "result.json")
    if a.formula_rev != "5":
        job["formula_revision"] = int(a.formula_rev)
        job["repair_feature_cache"] = None
    (dest / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    keys = [c["key"] for c in job["cases"]]
    env = dict(os.environ, ZENSIM_FORMULA_REV=a.formula_rev, ZENSIM_NEIGHBOUR_EXACT="1", ZENSIM_MAX_TIER="v3",
               RAYON_NUM_THREADS="1")

    def shard(s):
        sd = dest / "shards" / f"{s:02d}"
        sd.mkdir(parents=True)
        part = dict(job, cases=job["cases"][s::a.shards], output=str(sd / "result.json"))
        (sd / "job.json").write_text(json.dumps(part, indent=2) + "\n")
        with (sd / "run.log").open("x") as log:
            subprocess.run([str(a.bin), "metric::bake::steerfix_packet::engineering_packet", "--exact", "--nocapture"],
                           check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(env, STEERFIX_PACKET=str(sd / "job.json")))
        return json.loads((sd / "result.json").read_text())["rows"]

    with ThreadPoolExecutor(max_workers=a.shards) as pool:
        parts = list(pool.map(shard, range(a.shards)))
    # Keys repeat in the packet (the same image at several cases), so rows go back by position: shard s ran
    # cases s, s + shards, s + 2 * shards, ... and returns its rows in that order.
    rows = [None] * len(keys)
    for s, part in enumerate(parts):
        positions = range(s, len(keys), a.shards)
        if len(part) != len(positions):
            raise ValueError(f"shard {s}: {len(part)} rows for {len(positions)} cases")
        for i, r in zip(positions, part):
            if r["key"] != keys[i]:
                raise ValueError(f"shard {s}: row {r['key']} at case {keys[i]}")
            rows[i] = r
    if len(rows) != 135 or not all(r["model"] == digest and r["sensitivity_method"] == "exact-gradient-d12" for r in rows):
        raise ValueError("unexpected steer rows")
    result = {"packet_sha256": sha(dest / "job.json"), "formula_revision": job["formula_revision"], "rows": rows}
    (dest / "result.json").write_text(json.dumps(result, separators=(",", ":")))
    print(f"steer135 {a.name}: {sum(r['pass'] for r in rows)}/135", flush=True)


if __name__ == "__main__":
    main()
