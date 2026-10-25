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


class WidenTest(unittest.TestCase):
    def stored(self, rows=3):
        return pa.table({f"f{i}": pa.array(np.full(rows, i, np.float32)) for i in range(1853)})

    def check(self, name, first, last):
        spec = r.WIDEN[name]
        vals = np.arange(3 * len(spec["ids"]), dtype=np.float32).reshape(3, -1)
        t = r.widen(self.stored(), vals, "t", spec)
        names = [c for c in t.column_names if c.startswith("f")]
        self.assertEqual(names, [f"f{i}" for i in range(last + 1)])
        for i in range(1853, first):
            self.assertTrue(np.isnan(t.column(f"f{i}").to_numpy()).all())
        for j, i in enumerate(range(first, last + 1)):
            np.testing.assert_array_equal(t.column(f"f{i}").to_numpy(), vals[:, j])
        np.testing.assert_array_equal(t.column("f1852").to_numpy(), np.full(3, 1852, np.float32))

    def test_globaff_columns_unchanged(self):
        self.check("globaff", 1867, 1890)
        self.assertIs(r.widen.__defaults__[0], None)  # no spec: globaff, as before

    def test_orient_columns(self):
        self.check("orient", 2301, 2336)

    def test_refuses_other_widths_and_nonfinite(self):
        spec = r.WIDEN["orient"]
        vals = np.zeros((3, 36), np.float32)
        with self.assertRaisesRegex(ValueError, "width 1852"):
            r.widen(self.stored().drop(["f1852"]), vals, "t", spec)
        vals[1, 7] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            r.widen(self.stored(), vals, "t", spec)


class GatedTest(unittest.TestCase):
    """Gated damage: the plan per gated channel set and the declaration a gated root trains with."""

    DECLS = REPO / "scripts/rev4_featpot"

    def decl(self, channels: str) -> bytes:
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
            out = Path(td) / "decl.json"
            r.cmd_gated_decl(SimpleNamespace(decl=str(self.DECLS / "byc_fx1g_declaration.json"), out=str(out),
                                             channels=channels))
            return out.read_bytes()

    def test_all_channels_reproduce_the_registered_declaration(self):
        self.assertEqual(self.decl("XYB"), (self.DECLS / "byc_fx1gg_declaration.json").read_bytes())

    def test_luminance_gate_reads_only_y_cells(self):
        diffs, shares = r.gated_plan(self.DECLS / "byc_fx1g_declaration.json", "Y")
        all_diffs, all_shares = r.gated_plan(self.DECLS / "byc_fx1g_declaration.json", "XYB")
        self.assertEqual(len(diffs), 164)
        self.assertTrue(all((s - r.GLOBAFF_IDS[0]) // 2 % 3 == 1 for s in shares))
        self.assertEqual([(d, s) for d, s in zip(all_diffs, all_shares) if (s - r.GLOBAFF_IDS[0]) // 2 % 3 == 1],
                         list(zip(diffs, shares)))
        self.assertEqual(self.decl("Y"), (self.DECLS / "byc_fx1gy_declaration.json").read_bytes())

    def test_gate_scales_each_difference_by_its_cell(self):
        diffs, shares = [13, 40], [1869, 1875]
        cols = {f"f{i}": pa.array(np.zeros(2, np.float32)) for i in range(r.GATED_FIRST)}
        cols["f13"], cols["f1869"] = pa.array(np.float32([2, 4])), pa.array(np.float32([0.25, 1]))
        cols["f40"], cols["f1875"] = pa.array(np.float32([3, 5])), pa.array(np.float32([0, 0.5]))
        t = r.gate_table(pa.table(cols), diffs, shares, "t")
        self.assertEqual(t.column(f"f{r.GATED_FIRST}").to_pylist(), [1.5, 0.0])
        self.assertEqual(t.column(f"f{r.GATED_FIRST + 1}").to_pylist(), [3.0, 2.5])


if __name__ == "__main__":
    unittest.main()
