#!/usr/bin/env python3
"""Verify bank receipts and account for every frozen membership, without labels.

This reads owner-produced keys/receipts only, never assembles features. Run
under the shared heavy lock and resource wrapper because artifacts are hashed.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from dataset_features import BUILD_COMMIT, EXTRACTOR_SHA, FEATURE_ID


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    scope = json.loads((args.root / "SCOPE_AUTHORITY.json").read_text())
    failures_path = args.root / "DECODE_FAILURES.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    failed_pairs = {row["pair_key"] for row in failures}
    recovery_path = args.root / "RECOVERY_BANKS.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    results = []
    for name in sorted(scope["existing_library_collections"]):
        inputs = args.root / "inputs" / name
        prep = json.loads((inputs / "PREPARATION.json").read_text())
        assert sha(inputs / "keys.parquet") == prep["keys_sha256"]
        assert sha(inputs / "membership-to-pairs.parquet") == prep["mapping_sha256"]
        source_keys = pq.read_table(inputs / "keys.parquet", columns=["pair_key"])
        expected = set(source_keys["pair_key"].to_pylist())
        available, receipts = set(), []
        for token in [prep["collection_token"], *recovery.get(name, [])]:
            bank = args.root / "bank" / token
            manifest_path = bank / "_MANIFEST.json"
            if not manifest_path.exists():
                continue
            m = json.loads(manifest_path.read_text())
            assert (
                m["build_commit"] == BUILD_COMMIT
                and m["binary_sha256"] == EXTRACTOR_SHA
            )
            assert m["formula_revision"] == "Rev7" and m["era_label"] == "rev7_rowlanes"
            assert m["feature_set_id"] == FEATURE_ID
            assert (
                m["assessment"]["features_only"] and not m["assessment"]["labels_read"]
            )
            assert sha(bank / "keys.parquet") == m["keys_sha256"]
            assert sha(bank / "features.parquet") == m["features_parquet_sha256"]
            keys = pq.read_table(bank / "keys.parquet", columns=["row_id", "pair_key"])
            assert keys.num_rows == m["rows"]
            assert keys["row_id"].to_pylist() == list(range(keys.num_rows))
            values = set(keys["pair_key"].to_pylist())
            assert len(values) == keys.num_rows and values <= expected
            assert not available & values, "duplicate delivery across recovery banks"
            available |= values
            receipts.append(
                {
                    "token": token,
                    "manifest_sha256": sha(manifest_path),
                    "pairs": len(values),
                }
            )
        mapping = pq.read_table(inputs / "membership-to-pairs.parquet")
        groups = collections.defaultdict(list)
        role_pairs = collections.defaultdict(set)
        for row in mapping.to_pylist():
            groups[
                (row["input_table"], row["membership_row_index"], row["role"])
            ].append(row)
            if row["pair_key"] in available:
                role_pairs[row["role"]].add(row["pair_key"])
        assert len(groups) == prep["frozen_membership_rows"]
        covered, skipped, errors, unresolved = (collections.Counter() for _ in range(4))
        for (_, _, role), rows in groups.items():
            pair_keys = {row["pair_key"] for row in rows if row["pair_key"] is not None}
            if not pair_keys:
                assert all(row["skip_reason"] for row in rows)
                skipped[role] += 1
            elif pair_keys <= available:
                covered[role] += 1
            elif pair_keys - available <= failed_pairs:
                errors[role] += 1
            else:
                unresolved[role] += 1
        assert expected <= available | failed_pairs or unresolved
        result = {
            "dataset_id": name,
            "frozen_memberships_by_role": prep["membership_rows_by_role"],
            "memberships_with_all_features_by_role": dict(covered),
            "no_reference_skips_by_role": dict(skipped),
            "decode_failure_memberships_by_role": dict(errors),
            "unresolved_memberships_by_role": dict(unresolved),
            "distinct_feature_pairs_by_role": {
                role: len(values) for role, values in role_pairs.items()
            },
            "distinct_feature_pairs": len(available),
            "prepared_pairs": len(expected),
            "skip_reasons": prep["skip_reasons"],
            "bank_receipts": receipts,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    unresolved = sum(sum(r["unresolved_memberships_by_role"].values()) for r in results)
    result = {
        "schema": "dataset-rev7-feature-coverage-v1",
        "response_tables_opened": False,
        "readme_example_exposure_receipt": "README_EXPOSURE.json",
        "unresolved_memberships": unresolved,
        "collections": results,
    }
    if args.write:
        assert unresolved == 0, "cannot finalize incomplete coverage"
        with (args.root / "FEATURE_COVERAGE.json").open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    return int(unresolved > 0)


if __name__ == "__main__":
    raise SystemExit(main())
