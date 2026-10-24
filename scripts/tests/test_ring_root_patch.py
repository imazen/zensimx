"""ring_root.patch: a derived root replaces only its arm's columns and keeps every other column's bits."""
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
        self.old = np.random.default_rng(7).random((5, len(r.MEASURED)), dtype=np.float32)
        self.replaced = r.ARMS["linpyr"]["replaced"]
        self.pos = {i: j for j, i in enumerate(r.MEASURED)}
        self.kept = [i for i in r.MEASURED if i not in self.replaced]
        self.new = self.old.copy()
        for i in self.replaced:
            self.new[:, self.pos[i]] += 1.0

    def test_replaces_only_the_arm_columns(self):
        t = r.patch(table(self.old), self.new, "t", self.replaced)
        for i in self.replaced:
            np.testing.assert_array_equal(t.column(f"f{i}").to_numpy(), self.new[:, self.pos[i]])
        for i in self.kept:
            np.testing.assert_array_equal(t.column(f"f{i}").to_numpy(), self.old[:, self.pos[i]])

    def test_refuses_a_kept_column_that_moved(self):
        self.new[2, self.pos[self.kept[0]]] = np.nextafter(self.new[2, self.pos[self.kept[0]]], np.float32(2))
        with self.assertRaisesRegex(ValueError, "differs on 1 rows"):
            r.patch(table(self.old), self.new, "t", self.replaced)

    def test_linpyr_columns_are_the_coarse_scales_and_scale_0_edge_width(self):
        self.assertEqual(len(self.replaced), 435)
        self.assertEqual([i for i in self.kept if r.scale_of(i) != 0], [])
        self.assertEqual(sorted(set(self.replaced) - {i for i in r.MEASURED if r.scale_of(i) > 0}), r.EDGE_WIDTH_S0)


if __name__ == "__main__":
    unittest.main()
