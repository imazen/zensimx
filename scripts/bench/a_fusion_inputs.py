#!/usr/bin/env python3
"""Relocate frozen pixel-gate manifests, retaining original file identities.

Only metadata is opened here. The caller transfers the NUL-delimited list with
tar; native readback verifies file/pixel hashes on the measurement host.
Existing library paths remain in place rather than copying the IQA library.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--nearid", type=Path, required=True)
    parser.add_argument("--steer", type=Path, required=True)
    parser.add_argument("--eval-pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    copies = set()

    def relocate(path):
        if path.startswith("/mnt/tower/"):
            return path
        copies.add(path)
        return args.remote_root + "/tree" + path

    identity = json.loads(args.identity.read_text())
    assert len(identity) == 62
    for row in identity:
        row["ref_path"] = relocate(row["ref_path"])
    nearid = json.loads(args.nearid.read_text())
    assert len(nearid["sources"]) == 24
    for row in nearid["sources"]:
        row["ref_path"] = relocate(row["ref_path"])
    steer = json.loads(args.steer.read_text())
    assert len(steer["cases"]) == 135
    for row in steer["cases"]:
        for key in ("reference", "distorted"):
            row[key] = relocate(row[key])
    with args.eval_pairs.open() as stream:
        eval_pairs = list(csv.DictReader(stream, delimiter="\t"))
    assert len(eval_pairs) == 4678
    for row in eval_pairs:
        for key in ("ref_path", "dist_path"):
            row[key] = relocate(row[key])
    with (args.out / "eval-pairs.tsv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ref_path", "dist_path", "pair_key"], delimiter="\t")
        writer.writeheader()
        writer.writerows(eval_pairs)
    for name, doc in (("identity", identity), ("nearid", nearid), ("steer", steer)):
        (args.out / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
    (args.out / "COPY_FILES.nul").write_bytes(b"".join(str(Path(path).relative_to("/")).encode() + b"\0"
                                                       for path in sorted(copies)))
    (args.out / "ORIGINALS.json").write_text(json.dumps(dict(
        identity=str(args.identity), nearid=str(args.nearid), steer=str(args.steer),
        eval_pairs=str(args.eval_pairs), copied_files=sorted(copies),
        remote_root=args.remote_root), indent=2) + "\n")
    print(f"manifests: 62 identity sources, 24 NEARID sources, 135 steering cases, 4678 EVAL pairs; {len(copies)} file transfers")


if __name__ == "__main__":
    main()
