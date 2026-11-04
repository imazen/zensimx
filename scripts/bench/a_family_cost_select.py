#!/usr/bin/env python3
"""Select the first sixteen clean paired calls per image from bounded attempts.

Raw attempts are immutable. A foreign rustc/cargo/aws/perf rejects its entire
image attempt; calls may be collected across admitted attempts chronologically.
The derived view deliberately carries no stale native summary statistics.
"""
import argparse
import hashlib
import json
from pathlib import Path
from a_family_cost_shared_run import cell_status, profile_valid

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("view", type=Path)
    p.add_argument("--final", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--host", choices=["r5900xt", "i270"], default="r5900xt")
    a = p.parse_args()
    root = a.study.resolve()
    epoch = (root / "QUIET_EPOCH.txt").read_text().strip()
    pairs = json.loads((root / "pairs/pairs.json").read_text())
    attempts = [root, root / "retry01", root / "retry02", root / "retry02-i270"]
    chosen, missing, profiles = [], [], []
    derived = []
    for threads in [1, 16]:
        for size in ["64", "256", "1024", "large"]:
            cell = f"{size}-{threads}t"
            states = [(attempt, *cell_status(attempt, size, threads, epoch)) for attempt in attempts]
            for pair in [r for r in pairs if r["size"] == size]:
                candidates, reasons = [], []
                attempted_rounds = flagged_rounds = 0
                for attempt, valid_images, mode in states:
                    path = attempt / f"timing/{cell}/{pair['id']}.json"
                    reason = dict(attempt=str(attempt.relative_to(root)), admission=mode)
                    if path.exists():
                        data = path.read_bytes()
                        row = json.loads(data)
                        group = row["comparisons"][0]
                        assert [b["name"] for b in group["benchmarks"]] == ["extract492", "extract336", "compute_A"]
                        assert len(group["samples"]) == 32
                        assert all(s["iterations"] == 1 and s["gate_clean"] is not None for s in group["samples"])
                        cpu = row["testbed"]["cpu_model"]
                        wanted = "5900XT" if a.host == "r5900xt" else "270K Plus"
                        if wanted not in cpu:
                            reasons.append(dict(reason, state="different host", cpu=cpu))
                            continue
                        attempted_rounds += 32
                        clean = sum(s["gate_clean"] for s in group["samples"])
                        flagged_rounds += 32 - clean
                        reason.update(timestamp=row["timestamp"], clean=clean, unreliable=row["unreliable"])
                        if pair["id"] in valid_images and not row["unreliable"]:
                            for index, sample in enumerate(group["samples"]):
                                if sample["gate_clean"]:
                                    candidates.append((row["timestamp"], index, str(path), hashlib.sha256(data).hexdigest(), sample, valid_images[pair["id"]]["admission_mode"]))
                    else:
                        reason["state"] = "not measured (contended or unexecuted)"
                    reasons.append(reason)
                candidates.sort(key=lambda r: (r[0], r[1], r[2]))
                evidence = dict(cell=cell, case=pair["id"], clean_available=len(candidates),
                                attempted_rounds=attempted_rounds, flagged_rounds=flagged_rounds, attempts=reasons)
                if len(candidates) < 16:
                    outside = a.host == "i270" and size in ["64", "256"]
                    missing.append(dict(evidence, state="not collected on this host" if outside else "not measured (contended)"))
                    continue
                selected = candidates[:16]
                origins = [dict(timestamp=r[0], source_round=r[1], source=r[2], sha256=r[3], admission_mode=r[5]) for r in selected]
                chosen.append(dict(evidence, selected=origins))
                derived.append((cell, pair["id"], dict(kind="derived_first16_clean_paired_calls", unreliable=False,
                    comparisons=[dict(group_name=f"{pair['id']}_{threads}t",
                        benchmarks=[dict(name=n) for n in ["extract492", "extract336", "compute_A"]],
                        completed_rounds=16, samples=[r[4] for r in selected])], source_rounds=origins)))
    for size in ["64", "256", "1024", "large"]:
        ready = (next((attempt / f"perf/{size}" for attempt in attempts[:3] if profile_valid(attempt, size)), None)
                 if a.host == "r5900xt" else None)
        profiles.append(dict(size=size, source=str(ready) if ready else None,
                             state="measured" if ready else "not measured (contended)"))
    record = dict(quiet_epoch=epoch, host=a.host, provisional=not a.final,
                  policy="first16 clean paired calls; per-image foreign-process rejection; primary + two retries",
                  chosen=chosen, missing=missing, profiles=profiles)
    print(f"{len(chosen)}/160 image/thread records; {sum(r['source'] is not None for r in profiles)}/4 profiles")
    for size in ["64", "256", "1024", "large"]:
        for threads in [1,16]:
            cell=f"{size}-{threads}t"
            print(cell,sum(r["cell"]==cell for r in chosen),"/20 images")
    if not a.final and not a.preview:
        return
    if a.final:
        completion = root / "BOUNDED_COMPLETE.json"
        finished = completion.exists() and json.loads(completion.read_text())["state"] == "complete"
        assert finished or "BOUNDED PASSES COMPLETE" in (root / "shared-bounded.log").read_text(), "do not finalize an active run"
    assert not a.view.exists(), "preserve an existing accepted view"
    a.view.mkdir(parents=True)
    (a.view / "pairs").symlink_to(root / "pairs", target_is_directory=True)
    for cell, case, row in derived:
        folder=a.view/f"timing/{cell}"
        folder.mkdir(parents=True,exist_ok=True)
        (folder/f"{case}.json").write_text(json.dumps(row,indent=2)+"\n")
    (a.view/"perf").mkdir()
    for row in profiles:
        if row["source"]:
            (a.view/f"perf/{row['size']}").symlink_to(row["source"],target_is_directory=True)
    (a.view/"selection.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__ == "__main__":
    main()
