"""Production UPIQ metadata admission and zero-label-open refusal tripwires."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import upiq380 as u


class UpiqAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp", prefix="upiq-metadata-test-")
        self.root = Path(self.tmp.name)
        for ds, nc, nd, nl in [("narwaria", 10, 2, 7), ("korshunov", 20, 3, 4)]:
            for c in range(1, nc + 1):
                scene = self.root / ds / f"{c:02}"
                scene.mkdir(parents=True)
                (scene / f"i{c:02}.exr").touch()
                for d in range(1, nd + 1):
                    for level in range(1, nl + 1):
                        (scene / f"i{c:02}_{d:02}_{level}.exr").touch()
        self.a = dict(schema="upiq380-extraction-admission-v1", role="train", tier="T2", authority="D3-2026-10-07",
            formula_revision=5, input_contract="upiq-exr-bt709-nits-v1", requested_ids=u.columns("by_v2fy"),
            image_root=str(self.root), rows=u.metadata_rows(self.root), split_rule=u.RULE,
            target_transform=u.TRANSFORM, label_file_approved=str(u.LABEL), label_sha256=u.LABEL_SHA)

    def tearDown(self):
        self.tmp.cleanup()

    def refuse_without_opens(self, a):
        with patch.object(Path, "open", side_effect=AssertionError("payload opened before refusal")) as tripwire:
            # Requested IDs are the already admitted registered source metadata.
            with patch.object(u, "columns", return_value=self.a["requested_ids"]):
                with self.assertRaises(ValueError):
                    u.targets(a)
            self.assertEqual(tripwire.call_count, 0)

    def test_metadata_membership_is_380_and_only_30_HDR_references(self):
        with patch.object(Path, "open", side_effect=AssertionError("payload opened")) as tripwire:
            with patch.object(u, "columns", return_value=self.a["requested_ids"]):
                rows = u.admit_metadata(self.a)
            self.assertEqual(len(rows), 380)
            self.assertEqual(len({r["reference_rel"] for r in rows}), 30)
            self.assertEqual(sum(r["dataset"] == "narwaria" for r in rows), 140)
            self.assertEqual(sum(r["dataset"] == "korshunov" for r in rows), 240)
            self.assertEqual(tripwire.call_count, 0)

    def test_role_revision_source_and_label_inventory_refuse_before_open(self):
        for field, value in [("role", "val"), ("tier", "T0"), ("formula_revision", 4),
                ("authority", "other"), ("label_file_approved", "/mnt/v/datasets/upiq/upiq_subjective_scores.csv"),
                ("label_sha256", "0" * 64), ("target_transform", "fit-offset"), ("split_rule", "label-quantile")]:
            with self.subTest(field=field):
                a = copy.deepcopy(self.a)
                a[field] = value
                self.refuse_without_opens(a)

    def test_rev7_is_explicit_and_does_not_grant_label_ingestion(self):
        a = copy.deepcopy(self.a)
        a.update(schema="upiq380-extraction-admission-rev7-v1", formula_revision=7,
                 requested_ids=list(range(228)) + list(range(372, 720)))
        self.assertEqual(len(u.admit_metadata(a, native_rev7=True)), 380)
        self.refuse_without_opens(a)  # Existing Rev5 target reader still refuses it.
        a["rows"][-1]["dataset"] = "live"
        with self.assertRaises(ValueError):
            u.admit_metadata(a, native_rev7=True)

    def test_bad_last_member_and_duplicate_refuse_before_any_label_open(self):
        for field, value in [("dataset", "live"), ("distorted_rel", "../live/image.png"),
                ("condition_id", "l-i01-l-01-1"), ("reference_rel", "tid2013/01/i01.png")]:
            with self.subTest(field=field):
                a = copy.deepcopy(self.a)
                a["rows"][-1][field] = value
                self.refuse_without_opens(a)
        a = copy.deepcopy(self.a)
        a["rows"][-1] = a["rows"][0]
        self.refuse_without_opens(a)

    def test_unapproved_image_inventory_refuses_without_label_opens(self):
        (self.root / "narwaria/01/extra.exr").touch()
        self.refuse_without_opens(self.a)

    def test_same_width_different_ids_refuse_before_label_open(self):
        a = copy.deepcopy(self.a)
        a["requested_ids"][0] = 0
        self.assertEqual(len(a["requested_ids"]), 420)
        self.assertEqual(a["requested_ids"], sorted(set(a["requested_ids"])))
        self.refuse_without_opens(a)

    def test_all_leg_manifests_precede_human_table_and_label_access(self):
        a = copy.deepcopy(self.a)
        for row in a["rows"]:
            self.bind_row(row)
        (self.root / "extraction-admission.json").write_text(json.dumps(a))
        receipt = {"legs": {}}
        manifests = {}
        for split in ("fit", "development"):
            path = self.root / f"upiq380_{split}.parquet"
            key = path.with_suffix(".keys.parquet")
            man = Path(f"{path}.manifest.json")
            members = a["rows"] if split == "fit" else []
            manifests[split] = dict(schema="upiq380-v2-leg-v1", source="UPIQ-380", arm="uh4", role="train",
                tier="T2", split=split, authority=a["authority"], formula_revision=5,
                requested_ids=a["requested_ids"], input_contract=a["input_contract"], build_commit="fixture",
                binary_sha256="1" * 64, admission_sha256="1" * 64, target_transform=u.TRANSFORM,
                split_rule=u.RULE, member_set=[r["condition_id"] for r in members], rows=len(members),
                label_source={"sha256":u.LABEL_SHA,"path":str(u.LABEL)})
            receipt["legs"][split] = dict(table=str(path), keys=str(key), manifest=str(man), manifest_sha256="1" * 64)
        (self.root / "INGEST_RECEIPT.json").write_text(json.dumps(receipt))
        args = SimpleNamespace(dest=str(self.root), features="unopened-features", binary="fixture-binary", build_commit="fixture")
        for field, value in [("role", "val"), ("source", "UPIQ-SDR"), ("arm", "other"),
                ("requested_ids", [0]), ("member_set", ["l-i01-l-01-1"])]:
            with self.subTest(field=field):
                bad = copy.deepcopy(manifests)
                bad["development"][field] = value
                for split in bad:
                    Path(receipt["legs"][split]["manifest"]).write_text(json.dumps(bad[split]))
                with patch.object(u, "sha", return_value="1" * 64), patch.object(u, "targets", side_effect=AssertionError("labels opened")) as tripwire:
                    with self.assertRaisesRegex(ValueError, "leg manifest identity"):
                        u.verify(args)
                    self.assertEqual(tripwire.call_count, 0)

    @staticmethod
    def bind_row(row):
        row.update(split="fit", reference_sha256="1" * 64, distorted_sha256="2" * 64)
        row["pair_key"] = hashlib.sha256(("upiq380-original-byte-pair-v1\0" + row["condition_id"] + "\0"
            + row["reference_sha256"] + "\0" + row["distorted_sha256"]).encode()).hexdigest()

    def test_split_and_row_key_rewiring_refuse_before_labels(self):
        a = copy.deepcopy(self.a)
        for row in a["rows"]:
            self.bind_row(row)
        for field, value in [("split", "development"), ("pair_key", "0" * 64),
                ("reference_sha256", "not-a-hash"), ("distorted_sha256", "A" * 64)]:
            with self.subTest(field=field):
                bad = copy.deepcopy(a)
                bad["rows"][-1][field] = value
                self.refuse_without_opens(bad)
        self.refuse_without_opens(self.a)  # An unbound metadata-only list cannot read labels.


if __name__ == "__main__":
    unittest.main()
