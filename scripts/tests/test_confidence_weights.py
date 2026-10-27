"""Original row bindings fail closed before native loss integration."""

import hashlib
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "byconstruction"))
from confidence_weights import bind, original_cell


class Bindings(unittest.TestCase):
    def test_dataset_omissions_preserve_original_full_authority_namespace(self):
        for fold in ["pipal", "pairs", "mos7", "kadid", "tid2013", "konfig", "cid22_a25"]:
            route = "full" if fold in {"pipal", "pairs", "mos7"} else fold
            authority = {"cells": {f"{route}-s1": {"seed_index": 1}}}
            self.assertEqual(original_cell(authority, fold, 1)["seed_index"], 1)
            with self.assertRaises(ValueError):
                original_cell(authority, fold, 0)

    def rows(self):
        return [
            dict(
                table_row=i,
                collection="fixture",
                reference_group="r",
                native=dict(item=str(i), reference="native-r"),
            )
            for i in range(2)
        ]

    def test_dense_first_seen_refs_and_uncovered_reference_are_preserved(self):
        rows = self.rows()
        result = bind(
            [1.0, 2.0, 3.0, 4.0],
            ["r", "r", "other", "other"],
            rows,
            {("fixture", str(i), "native-r"): [float(i), 0.1] for i in range(2)},
        )
        digest = hashlib.sha256(
            b"".join(
                struct.pack("<dI", s, r)
                for s, r in [(1.0, 0), (2.0, 0), (3.0, 1), (4.0, 1)]
            )
        ).hexdigest()
        self.assertEqual(result["scores_refs_sha256"], digest)
        self.assertEqual(result["covered_rows"], 2)
        self.assertEqual(result["points"][2:], [None, None])
        self.assertEqual(result["rows"], 4)

    def test_partial_missing_duplicated_and_wrong_order_rows_refuse(self):
        rows = self.rows()
        maps = {("fixture", str(i), "native-r"): [float(i), 0.1] for i in range(2)}
        for selected in [rows[:1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                bind([1.0, 2.0], ["r", "r"], selected, maps)
        with self.assertRaises(ValueError):
            bind([1.0, 2.0], ["r", "wrong"], rows, maps)
        with self.assertRaises(ValueError):
            bind([1.0, 2.0], ["r", "r"], rows, {})
        with self.assertRaises(ValueError):
            bind([1.0, float("nan")], ["r", "r"], rows, maps)
        with self.assertRaises(ValueError):
            bind(
                [1.0, 2.0],
                ["r", "r"],
                rows,
                maps | {("fixture", "0", "native-r"): [0.0, -1.0]},
            )


if __name__ == "__main__":
    unittest.main()
