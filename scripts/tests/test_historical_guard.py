"""Obsolete launchers must refuse before touching data or starting tools."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1] / "byconstruction"
sys.path.insert(0, str(BASE))
import historical_guard as guard


class HistoricalEntrypoints(unittest.TestCase):
    def test_every_entry_refuses_without_dependencies_or_arguments(self):
        for entry in guard.ENTRIES:
            with self.subTest(entry=entry):
                # -S makes an early accidental numpy/pandas import fail visibly.
                cmd = [sys.executable, "-S"] if entry.endswith(".py") else ["bash"]
                result = subprocess.run([*cmd, str(BASE / entry)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 64, result.stderr)
                self.assertIn("OBSOLETE ENTRY POINT", result.stderr)
                self.assertIn("No work started", result.stderr)

    def test_replay_is_explicit_first_argument_not_environment_or_trailing_flag(self):
        for args in [[], ["data", "--historical-replay"], ["--historical-replay=yes"]]:
            with self.assertRaisesRegex(ValueError, "OBSOLETE"):
                guard.require_replay(args, "serve_parity.py")
        with patch.object(guard.socket, "gethostname", return_value="i265"):
            self.assertEqual(guard.require_replay(["--historical-replay", "input"], "serve_parity.py"), ["input"])

    def test_explicit_replay_does_not_override_host_restrictions(self):
        for host in ["r7900x", "r7900x.lan", "7900x"]:
            with patch.object(guard.socket, "gethostname", return_value=host):
                for entry in guard.ENTRIES:
                    with self.assertRaisesRegex(ValueError, "HOST RESTRICTION"):
                        guard.require_replay(["--historical-replay"], entry)
        with patch.object(guard.socket, "gethostname", return_value="dev"):
            for entry in guard.BENCHMARKS:
                with self.assertRaisesRegex(ValueError, "HOST RESTRICTION"):
                    guard.require_replay(["--historical-replay"], entry)

    def test_unknown_entry_is_not_implicitly_admitted(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            guard.require_replay(["--historical-replay"], "new-unreviewed.sh")


if __name__ == "__main__":
    unittest.main()
