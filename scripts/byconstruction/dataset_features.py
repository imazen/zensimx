#!/usr/bin/env python3
"""Run the existing assessment-bank owner serially on frozen library keys.

Caller must hold shared heavy.lock and run-heavy. This driver performs no
feature arithmetic or assembly, reads no subjective payload, and never
converts image formats. Failed owner outputs remain preserved for diagnosis.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

EXTRACTOR_SHA = "ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8"
BUILD_COMMIT = "b77883be18db5933a9db4b9e2670a2ba88a9729b"
FEATURE_ID = "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args()
    assert sha(args.bin) == EXTRACTOR_SHA
    assert (args.bin.parent / "BUILD_COMMIT").read_text().strip() == BUILD_COMMIT
    scope = json.loads((args.root / "SCOPE_AUTHORITY.json").read_text())
    admitted = set(scope["existing_library_collections"])
    assert not admitted & {"ls2n-ipi-sr", "espl-synthetic"}
    selected = set(args.only) if args.only else admitted
    assert selected <= admitted
    owner = Path(__file__).resolve().parents[1] / "rev4_featpot/rev5_bank.py"
    logs = args.root / "owner-logs"
    logs.mkdir(exist_ok=True)
    progress = args.root / "EXTRACTION_PROGRESS.jsonl"
    failed = 0
    for name in sorted(selected, key=lambda n: (n == "bapps", n)):
        inputs = args.root / "inputs" / name
        prep = json.loads((inputs / "PREPARATION.json").read_text())
        assert sha(owner) == prep["bank_owner_sha256"], "assessment owner changed"
        event = {"collection": name, "time_utc": datetime.now(timezone.utc).isoformat()}
        if not prep["distinct_pairs"]:
            event |= {"state": "NO_PAIRS", "reason": prep["skip_reasons"]}
        else:
            token = prep["collection_token"]
            result = args.root / "bank" / token
            manifest_path = result / "_MANIFEST.json"
            if not result.exists():
                command = [
                    sys.executable,
                    str(owner),
                    "extract",
                    token,
                    "--bin",
                    str(args.bin),
                    "--build-commit",
                    BUILD_COMMIT,
                    "--revision",
                    "7",
                    "--era",
                    "rev7_rowlanes",
                    "--instrument-manifest",
                    str(inputs / "instrument.json"),
                    "--out",
                    str(args.root / "bank"),
                    "--threads",
                    "24",
                ]
                print(f"START {name} {prep['distinct_pairs']} pairs", flush=True)
                with (logs / f"{token}.log").open("x") as log:
                    rc = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=os.environ | {"PYTHONUNBUFFERED": "1"},
                    ).returncode
                event["returncode"] = rc
            if not manifest_path.exists():
                event |= {
                    "state": "OWNER_FAILED_PRESERVED",
                    "log": str(logs / f"{token}.log"),
                    "native_logs": str(args.root / "bank" / "_work" / token),
                }
                failed += 1
            else:
                manifest = json.loads(manifest_path.read_text())
                assert manifest["rows"] == prep["distinct_pairs"]
                assert manifest["build_commit"] == BUILD_COMMIT
                assert manifest["binary_sha256"] == EXTRACTOR_SHA
                assert manifest["feature_set_id"] == FEATURE_ID
                assert manifest["formula_revision"] == "Rev7"
                assert manifest["era_label"] == "rev7_rowlanes"
                assert manifest["assessment"]["features_only"] is True
                assert manifest["assessment"]["labels_read"] is False
                assert sha(result / "keys.parquet") == manifest["keys_sha256"]
                assert (
                    sha(result / "features.parquet")
                    == manifest["features_parquet_sha256"]
                )
                event |= {
                    "state": "OWNER_VERIFIED",
                    "pairs": manifest["rows"],
                    "manifest_sha256": sha(manifest_path),
                }
        with progress.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
    print(
        f"COMPLETE owner_failures={failed}; all outputs and logs retained", flush=True
    )
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
