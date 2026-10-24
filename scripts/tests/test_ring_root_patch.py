"""ring_root.patch: derived-root column replacement keeps every other column's bits, and AVIF rows obey the control."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pyarrow as pa

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/byconstruction"))
import ring_root as r  # noqa: E402


def table(values: np.ndarray) -> pa.Table:
    return pa.table({f"f{i}": pa.array(np.ascontiguousarray(values[:, j])) for j, i in enumerate(r.MEASURED)})


class PatchTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.old = rng.random((5, len(r.MEASURED)), dtype=np.float32)
        self.replaced = r.ARMS["linpyr"]["replaced"]
        self.pos = {i: j for j, i in enumerate(r.MEASURED)}
        self.kept = [i for i in r.MEASURED if i not in self.replaced]
        self.new = self.old.copy()
        for i in self.replaced:
            self.new[:, self.pos[i]] += 1.0

    def column(self, t: pa.Table, i: int) -> np.ndarray:
        return t.column(f"f{i}").to_numpy()

    def test_replaces_only_the_arm_columns(self):
        t = r.patch(table(self.old), self.new, "t", self.replaced)
        for i in self.replaced:
            np.testing.assert_array_equal(self.column(t, i), self.new[:, self.pos[i]])
        for i in self.kept:
            np.testing.assert_array_equal(self.column(t, i), self.old[:, self.pos[i]])

    def test_refuses_a_kept_column_that_moved(self):
        self.new[2, self.pos[self.kept[0]]] += 1e-7
        with self.assertRaisesRegex(ValueError, "differs on 1 rows"):
            r.patch(table(self.old), self.new, "t", self.replaced)

    def test_avif_rows_are_replaced_whole(self):
        free = np.array([False, True, False, False, True])
        self.new = self.old.copy()
        self.new[free] += 0.5
        t = r.patch(table(self.old), self.new, "t", [], free)
        for i in r.MEASURED:
            np.testing.assert_array_equal(self.column(t, i)[free], self.new[free, self.pos[i]])
            np.testing.assert_array_equal(self.column(t, i)[~free], self.old[~free, self.pos[i]])

    def test_avif_rows_must_match_the_reference(self):
        free = np.array([False, True, False, False, False])
        self.new[1] += 0.25
        ref = self.new.copy()
        r.patch(table(self.old), self.new, "t", self.replaced, free, ref)
        ref[1, self.pos[self.kept[3]]] += 1e-7
        with self.assertRaisesRegex(ValueError, "reference bank on AVIF rows"):
            r.patch(table(self.old), self.new, "t", self.replaced, free, ref)

    def test_a_non_avif_row_cannot_hide_behind_the_rule(self):
        free = np.array([False, True, False, False, False])
        self.new[3, self.pos[self.kept[1]]] += 1e-7
        with self.assertRaisesRegex(ValueError, "differs on 1 rows"):
            r.patch(table(self.old), self.new, "t", self.replaced, free)

    def test_scale_map_and_arm_columns(self):
        self.assertEqual(len(self.replaced), 435)
        self.assertEqual(sorted(i for i in self.kept if r.scale_of(i) != 0), [])
        self.assertEqual(sorted(set(self.replaced) - {i for i in r.MEASURED if r.scale_of(i) > 0}), r.EDGE_WIDTH_S0)
        self.assertTrue(r.is_avif("a.png", "b.AVIF") and not r.is_avif("a.png", "b.jxl"))


if __name__ == "__main__":
    unittest.main()
