#!/usr/bin/env python3
"""Run each gate check under the shared host lock and persist its result."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("phase", choices=("research-clippy", "public"))
    args = parser.parse_args()
    root = args.root.resolve()
    public = args.phase == "public"
    source = root / ("public-after" if public else "research-after")
    stem = "PUBLIC" if public else "RESEARCH_RETRY"
    env = dict(os.environ, TMPDIR=str(root / "scratch"),
               CARGO_TARGET_DIR=str(root / "target-tests"),
               CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0",
               CARGO_INCREMENTAL="0", RUST_TEST_THREADS="1")
    clippy = ["cargo", "clippy", "--workspace", "--all-targets"]
    if public:
        # Reuse the already generated, ignored PNG fixtures without rewriting
        # the tracked fixture set or invoking an external image writer.
        fixtures = source / "zensim/tests/fixtures"
        for original in (root / "public-before/zensim/tests/fixtures").glob("*.png"):
            dest = fixtures / original.name
            if dest.exists():
                assert dest.read_bytes() == original.read_bytes(), dest
            else:
                os.link(original, dest)
        pinned = ["bash", "scripts/profile_test_cpuset.sh", "0-7", "env"]
        pinned += [f"{key}={env[key]}" for key in (
            "CARGO_TARGET_DIR", "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_TEST_DEBUG",
            "CARGO_INCREMENTAL", "RUST_TEST_THREADS")]
        tests = ["cargo", "test", "--workspace", "--all-targets"]
        steps = [("default", pinned + tests),
                 ("legacy", pinned + tests + ["--features", "legacy-models"]),
                 ("profiles", ["cargo", "test", "-p", "zensim", "--all-features",
                               "--test", "profiles", "--", "--test-threads=1"]),
                 ("clippy-default", clippy + ["--", "-D", "warnings"]),
                 ("clippy-legacy", clippy + ["--features", "legacy-models", "--", "-D", "warnings"]),
                 ("clippy-all", clippy + ["--all-features", "--", "-D", "warnings"]),
                 ("split", ["cargo", "run", "--release", "-p", "zensim", "--features",
                            "training,classification", "--example", "split_parity_check"])]
    else:
        steps = [("clippy", ["just", "clippy"])]
    state = {"source": str(source), "kernel_sha256": hashlib.sha256(
        (source / "zensim/src/feature_v2.rs").read_bytes()).hexdigest(),
        "launcher_pid": os.getpid(), "required": [name for name, _ in steps],
        "completed": [], "stage": "starting", "failure": None, "results": []}
    receipt = root / f"{stem}_CHECKS.json"
    assert not receipt.exists(), receipt

    def save():
        state["last_check"] = datetime.datetime.now(datetime.UTC).isoformat()
        receipt.write_text(json.dumps(state, indent=2) + "\n")

    save()
    for name, payload in steps:
        log = root / f"{stem.lower()}-{name}.log"
        assert not log.exists(), log
        command = ["flock", str(Path.home() / "tmp/heavy.lock"),
                   str(Path.home() / "work/claudehints/scripts/run-heavy"),
                   "--mem", "16G", "--jobs", "8", "--", "taskset", "-c", "0-15"] + payload
        state["stage"] = name + " queued"
        save()
        with log.open("x") as output:
            if name == "split":
                with (root / "public-after-parity.tsv").open("x") as tsv:
                    result = subprocess.run(command, cwd=source, env=env, stdout=tsv, stderr=output)
            else:
                result = subprocess.run(command, cwd=source, env=env, stdout=output, stderr=output)
        state["results"].append({"name": name, "command": command, "rc": result.returncode,
                                 "log": str(log), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        if result.returncode:
            state["stage"] = state["failure"] = f"{name} failed-{result.returncode}"
            save()
            raise SystemExit(result.returncode)
        state["completed"].append(name)
        state["stage"] = name + " passed"
        save()
        print(state["stage"], flush=True)


if __name__ == "__main__":
    main()
