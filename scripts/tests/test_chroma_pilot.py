"""Registered chroma ladder admission, including desaturation/tie refusals."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "chroma_pilot", Path(__file__).resolve().parents[1] / "byconstruction/chroma_pilot.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class AdmissionTests(unittest.TestCase):
    def test_margin_accumulates_without_keeping_subpoint_rungs(self):
        self.assertEqual(P.admission([100, 99.7, 99, 98.8, 98])["kept_indices"], [0, 2, 4])

    def test_first_reversal_ends_ladder_even_if_below_last_kept(self):
        r = P.admission([100, 99, 98, 97.5, 97.6, 90])
        self.assertEqual(r["kept_indices"], [0, 1, 2])
        self.assertEqual(r["stop_index"], 4)

    def test_tie_stops_and_short_ladder_drops(self):
        r = P.admission([100, 99, 99, 80])
        self.assertEqual(r["kept_indices"], [])
        self.assertEqual(r["candidate_indices"], [0, 1])
        self.assertEqual(r["stop_index"], 2)

    def test_negative_scores_are_not_clamped(self):
        self.assertEqual(P.admission([0, -1, -2])["kept_indices"], [0, 1, 2])

    def test_nonfinite_scores_refuse(self):
        for values in [[], [100, float("nan")], [float("inf")]]:
            with self.assertRaises(ValueError):
                P.admission(values)

    def test_families_have_disjoint_valid_knobs_and_fixed_luma(self):
        g = P.grids()
        for family in ["ycc444", "ycc420"]:
            tables = g[family]["plane_tables"]
            self.assertEqual(len(tables), 32)
            for table in tables:
                self.assertIn(table["m"][0], [0, .05])
                self.assertEqual(table["m"][1], table["m"][2])
            self.assertNotIn("xyb", g[family])
        self.assertNotIn("plane_tables", g["xyb"])
        self.assertEqual(g["xyb"]["chroma_distance_scales"][0], [.1, .1])
        self.assertEqual(g["xyb"]["chroma_distance_scales"][-1], [5, 5])


if __name__ == "__main__":
    unittest.main()
