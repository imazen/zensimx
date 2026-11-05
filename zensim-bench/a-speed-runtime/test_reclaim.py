"""Exercise cache deletion guards with temporary, invented Cargo output."""
import contextlib
import importlib.util
import io
import json
import mmap
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("reclaim", Path(__file__).with_name("reclaim_cargo.py"))
reclaim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reclaim)


class CacheReclaim(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="cargo-reclaim-control-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("target", "target-tests"):
            target = self.root / name
            target.mkdir()
            (target / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55\n")
        self.protected = self.root / "fixtures"
        self.protected.mkdir()
        (self.protected / "pixels").write_bytes(b"invented fixture")

    def run_reclaim(self):
        with patch.object(reclaim, "compute_lock", return_value={"held_by_worker_ancestor": True}), \
                patch("sys.argv", ["reclaim", str(self.root), "control"]), \
                contextlib.redirect_stdout(io.StringIO()):
            reclaim.main()

    def test_non_cargo_directory_is_preserved(self):
        (self.root / "target/CACHEDIR.TAG").write_text("protected data\n")
        with self.assertRaisesRegex(AssertionError, "not a Cargo cache"):
            self.run_reclaim()
        self.assertTrue((self.root / "target-tests").exists())
        self.assertEqual((self.protected / "pixels").read_bytes(), b"invented fixture")

    def test_live_mapping_refuses_deletion(self):
        active = self.root / "target/mapped"
        active.write_bytes(b"live cache control")
        with active.open("r+b") as stream, mmap.mmap(stream.fileno(), 0) as mapping:
            self.assertEqual(mapping[:4], b"live")
            with self.assertRaisesRegex(AssertionError, "mapped by a live process"):
                self.run_reclaim()
        self.assertTrue(active.exists())

    def test_release_control_and_non_cache_data_survive(self):
        source = self.root / "target/release/examples/control"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"\x7fELFinvented executable")
        self.run_reclaim()
        preserved = self.root / "preserved-cargo/control"
        receipt = json.loads((preserved / "RECLAIMED.json").read_text())
        self.assertEqual(len(receipt["preserved"]), 1)
        self.assertEqual((preserved / "target/release/examples/control").read_bytes(),
                         b"\x7fELFinvented executable")
        self.assertEqual((self.protected / "pixels").read_bytes(), b"invented fixture")
        self.assertFalse((self.root / "target").exists())
        self.assertFalse((self.root / "target-tests").exists())

    def test_second_target_release_control_survives(self):
        source = self.root / "target-tests/release/examples/checker"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"\x7fELFsecond target control")
        self.run_reclaim()
        destination = self.root / "preserved-cargo/control/target-tests/release/examples/checker"
        self.assertEqual(destination.read_bytes(), b"\x7fELFsecond target control")

    def test_one_remaining_target_is_supported(self):
        (self.root / "target/CACHEDIR.TAG").unlink()
        (self.root / "target").rmdir()
        source = self.root / "target-tests/release/checker"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"\x7fELFremaining target control")
        self.run_reclaim()
        destination = self.root / "preserved-cargo/control/target-tests/release/checker"
        self.assertEqual(destination.read_bytes(), b"\x7fELFremaining target control")


if __name__ == "__main__":
    unittest.main()
