#!/usr/bin/env python3
"""Run all 744 identity cells as four disjoint test shards under one heavy scope."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "sde", "packet", "bake", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    assert os.uname().nodename == "i270"
    assert not a.output.exists()
    directory = a.output.with_suffix(".shards")
    directory.mkdir(exist_ok=False)
    jobs = []
    commands = []
    for i in range(4):
        target = directory / f"{i}.jsonl"
        command = [str(a.sde), "-icl", "--", str(a.binary), "fusion", "identity",
                   str(a.packet), str(a.bake), str(target)]
        env = dict(os.environ, ZENSIM_FORMULA_REV="7", A_FUSION_IDENTITY_SHARD=f"{i}/4")
        assert "A_FUSION_IDENTITY_FIRST_ONLY" not in env
        log = (directory / f"{i}.log").open("x")
        jobs.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        commands.append(dict(command=command, shard=f"{i}/4"))
    status = []
    for job, log in jobs:
        status.append(job.wait())
        log.close()
    assert status == [0] * 4, status
    rows = []
    for i in range(4):
        records = [json.loads(line) for line in (directory / f"{i}.jsonl").read_text().splitlines()]
        assert records[-1]["summary"] and records[-1]["rows"] == 186
        rows.extend(row for row in records if not row.get("summary"))
    keyed = {(r["permutation"], r["reference"]): r for r in rows}
    assert len(rows) == len(keyed) == 744
    assert len({r["permutation"] for r in rows}) == 12
    assert len({r["reference"] for r in rows}) == 62
    with a.output.open("x") as stream:
        for key in sorted(keyed):
            stream.write(json.dumps(keyed[key]) + "\n")
        stream.write(json.dumps(dict(summary=True, rows=744, sources=62)) + "\n")
    receipt = dict(commands=commands, exit_codes=status, rows=744, permutations=12,
                   binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                   files={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (a.packet, a.bake, a.output)})
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("identity complete: 744/744 cells, 12 permutations, 62 sources", flush=True)


if __name__ == "__main__":
    main()
