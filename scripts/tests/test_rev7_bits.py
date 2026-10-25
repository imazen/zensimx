"""Missing baselines and incomplete tier captures must fail the existing comparer."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

COMPARE = Path(__file__).resolve().parents[1] / "byconstruction/perf_bits_compare.py"


class BitComparisonTests(unittest.TestCase):
    def test_extra_dump_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            base, cand = Path(tmp) / "base", Path(tmp) / "cand"
            base.mkdir()
            cand.mkdir()
            (base / "one.hex").write_text("0000000000000000\n")
            (cand / "one.hex").write_text("0000000000000000\n")
            result = self.run_compare(base, cand)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (cand / "foreign.hex").write_text("0000000000000000\n")
            self.assertEqual(self.run_compare(base, cand).returncode, 1)

    def test_tiers_require_every_requested_tier_and_nonempty_input(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            self.assertNotEqual(self.run_compare("--tiers", root).returncode, 0)
            tiers = ["v4x_t1", "v4_t1", "v3_t1", "scalar_t1", "v4x_t8"]
            for tier in tiers[:-1]:
                (root / f"r7_pair_{tier}.hex").write_text("0000000000000000\n")
            self.assertEqual(self.run_compare("--tiers", root).returncode, 1)
            (root / f"r7_pair_{tiers[-1]}.hex").write_text("0000000000000000\n")
            self.assertEqual(self.run_compare("--tiers", root).returncode, 0)
            (root / "r7_pair_v3_t1.hex").write_text("3ff0000000000000\n")
            self.assertEqual(self.run_compare("--tiers", root).returncode, 1)

    @staticmethod
    def run_compare(*args):
        return subprocess.run(
            [sys.executable, "-B", str(COMPARE), *map(str, args)],
            capture_output=True, text=True, check=False, env={**os.environ, "TMPDIR": str(Path.home() / "tmp")},
        )


if __name__ == "__main__":
    unittest.main()
