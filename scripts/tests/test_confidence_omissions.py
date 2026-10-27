"""Metadata controls: whole assessment-source omission, not fit-subset omission."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "byconstruction"))
from confidence_omissions import human_scope, dataset_families


def key(index, member, reference):
    return dict(pair_key=f"pixels-{index}", source_row_id=index, member_set=member,
        ref_basename=reference)


class OmissionIdentities(unittest.TestCase):
    def test_full_assessment_keeps_reference_absent_from_fit(self):
        a = key(0, "kadid_train", "kadid:I01")
        b = key(1, "kadid_select", "kadid:I02")
        c = key(2, "tid2013", "tid2013:i03")
        sources = {0: dict(ref_path="/source/I01.png"), 1: dict(ref_path="/source/I02.png")}
        result = human_scope("kadid", [a, c], [c], [a, b], sources)
        self.assertEqual(result["fitting_reference_count"], 1)
        self.assertEqual(result["full_assessment_reference_count"], 2)
        self.assertEqual([r["source_paths"] for r in result["reference_identities"]],
                         [["/source/I01.png"], ["/source/I02.png"]])
        # Pixel identity does not collapse separate observation source rows.
        same_pixels = a | {"source_row_id": 3}
        result = human_scope("kadid", [a, same_pixels, c], [c], [a, same_pixels, b],
                             sources | {3: sources[0]})
        self.assertEqual(result["removed_fit_rows"], 2)
        self.assertEqual(result["full_assessment_reference_count"], 2)

    def test_wrong_omission_or_source_reference_refuses(self):
        a = key(0, "kadid_train", "kadid:I01")
        b = key(1, "tid2013", "tid2013:i02")
        sources = {0: dict(ref_path="/source/I01.png")}
        with self.assertRaisesRegex(ValueError, "exact original filter"):
            human_scope("kadid", [a, b], [a, b], [a], sources)
        with self.assertRaisesRegex(ValueError, "different omitted collection"):
            human_scope("kadid", [a, b], [b], [a, b], sources)
        with self.assertRaisesRegex(ValueError, "reference differs"):
            human_scope("kadid", [a, b], [b], [a], {0: dict(ref_path="/source/I99.png")})
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            human_scope("kadid", [a, b], [b], [a, a], sources)

    def test_dataset_family_rows_preserve_identity_and_refuse_wrong_roles(self):
        def row(i, family):
            return dict(row=i, observation=f"pipal:labels.tsv:{i}", source="r", reference="pipal:r",
                family=family, role="fit", tier="B")
        rows = [row(0, 17), row(1, 17), row(2, 23)]
        self.assertEqual(dataset_families("pipal", rows, 3), [17, 23])
        for bad in [{"role": "EVAL"}, {"family": True}, {"row": 7},
                    {"observation": "other:labels.tsv:1"}, {"score": 1.}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dataset_families("pipal", [rows[0], rows[1] | bad, rows[2]], 3)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            dataset_families("pipal", rows, 4)


if __name__ == "__main__":
    unittest.main()
