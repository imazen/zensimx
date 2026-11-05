"""Check that the archive inventory covers preserved controls and refuses links."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ArtifactManifest(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="artifact-inventory-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def inventory(self):
        return subprocess.run(
            [sys.executable, "-B", str(Path(__file__).with_name("artifact_manifest.py")), str(self.root)],
            capture_output=True, text=True, check=False,
        )

    def test_preserved_controls_are_covered_while_root_caches_are_excluded(self):
        for name in ("target", "target-tests"):
            cache = self.root / name
            cache.mkdir()
            (cache / "unused").write_bytes(b"disposable build output")
            control = self.root / "preserved-cargo" / "receipt" / name / "release" / "checker"
            control.parent.mkdir(parents=True)
            control.write_bytes(b"\x7fELFretained reference executable")
        result = self.inventory()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = (self.root / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(rows), 2)
        for row in rows:
            digest, name = row.split("  ", 1)
            self.assertTrue(name.startswith("preserved-cargo/receipt/"))
            self.assertEqual(digest, hashlib.sha256((self.root / name).read_bytes()).hexdigest())

    def test_directory_link_cannot_silently_omit_fixture_content(self):
        fixture = self.root / "fixture"
        fixture.mkdir()
        (fixture / "pixels").write_bytes(b"retained pixels")
        os.symlink("fixture", self.root / "alias", target_is_directory=True)
        result = self.inventory()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("archive symlinks require an explicit policy", result.stderr)
        self.assertFalse((self.root / "SHA256SUMS").exists())


if __name__ == "__main__":
    unittest.main()
