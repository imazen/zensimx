"""Metadata controls: whole assessment-source omission, not fit-subset omission."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "byconstruction"))
from confidence_omissions import human_scope, dataset_families


def key(index, member, reference):
    return dict(pair_key=f"pixels-{index}", source_row_id=index, member_set=member,
        ref_basename=reference)


def bank_entry(row, path):
    return dict(ref_path=path, ref_group=row["ref_basename"], ref_pixels_sha256="a" * 64, width=16, height=16)


class OmissionIdentities(unittest.TestCase):
    def test_full_assessment_keeps_reference_absent_from_fit(self):
        a = key(0, "kadid_train", "kadid:I01")
        b = key(1, "kadid_select", "kadid:I02")
        c = key(2, "tid2013", "tid2013:i03")
        sources = {"kadid_train": {a["pair_key"]: bank_entry(a, "/source/I01.png")},
                   "kadid_select": {b["pair_key"]: bank_entry(b, "/source/I02.png")}}
        result = human_scope("kadid", [a, c], [c], [a, b], sources)
        self.assertEqual(result["fitting_reference_count"], 1)
        self.assertEqual(result["full_assessment_reference_count"], 2)
        self.assertEqual([r["source_paths"] for r in result["reference_identities"]],
                         [["/source/I01.png"], ["/source/I02.png"]])
        # Pixel identity does not collapse separate observation source rows.
        same_pixels = a | {"source_row_id": 3}
        result = human_scope("kadid", [a, same_pixels, c], [c], [a, same_pixels, b],
                             sources)
        self.assertEqual(result["removed_fit_rows"], 2)
        self.assertEqual(result["full_assessment_reference_count"], 2)

    def test_wrong_omission_or_source_reference_refuses(self):
        a = key(0, "kadid_train", "kadid:I01")
        b = key(1, "tid2013", "tid2013:i02")
        sources = {"kadid_train": {a["pair_key"]: bank_entry(a, "/source/I01.png")}}
        with self.assertRaisesRegex(ValueError, "exact original filter"):
            human_scope("kadid", [a, b], [a, b], [a], sources)
        with self.assertRaisesRegex(ValueError, "different omitted collection"):
            human_scope("kadid", [a, b], [b], [a, b], sources)
        with self.assertRaisesRegex(ValueError, "reference differs"):
            human_scope("kadid", [a, b], [b], [a], {"kadid_train": {
                a["pair_key"]: bank_entry(a, "/source/I99.png") | {"ref_group": "kadid:I99"}}})
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            human_scope("kadid", [a, b], [b], [a, a], sources)

    def test_member_local_row_ids_are_not_global_source_indices(self):
        a = key(0, "cid22_a25", "162520")
        b = key(0, "kadid_train", "kadid:I01")
        banks = {"cid22_a25": {a["pair_key"]: bank_entry(a, "/cid22/162520.png")},
                 "kadid_train": {b["pair_key"]: bank_entry(b, "/kadid/I01.png")}}
        cid = human_scope("cid22_a25", [a, b], [b], [a], banks)
        self.assertEqual(cid["reference_identities"][0]["source_paths"], ["/cid22/162520.png"])
        kadid = human_scope("kadid", [a, b], [a], [b], banks)
        self.assertEqual(kadid["reference_identities"][0]["source_paths"], ["/kadid/I01.png"])

    def test_dataset_family_rows_preserve_identity_and_refuse_wrong_roles(self):
        def row(i, family):
            return dict(row=i, observation=f"pipal:labels.tsv:{i}", source="r", reference="pipal:r",
                family=str(family), role="fit", tier="T2")
        rows = [row(0, 17), row(1, 17), row(2, 23)]
        self.assertEqual(dataset_families("pipal", rows, 3), [17, 23])
        for bad in [{"role": "EVAL"}, {"family": True}, {"family": "017"},
                    {"family": "1e3"}, {"family": "-1"}, {"row": 7},
                    {"observation": "other:labels.tsv:1"}, {"score": 1.}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dataset_families("pipal", [rows[0], rows[1] | bad, rows[2]], 3)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            dataset_families("pipal", rows, 4)


if __name__ == "__main__":
    unittest.main()
