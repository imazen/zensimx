import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import test_shaping_continuation_finalize as fixtures
from shaping_completion import encoded, sha
from shaping_reboot_recovery import work


class CompletedContinuationRecoveryTests(unittest.TestCase):
    def fixture(self, base):
        study, root, capture, native = fixtures.ContinuationFinalizationTests().fixture(base)
        done = json.loads((native / "RESULT.json").read_text())
        cap = json.loads(capture.read_text())
        cap["native"]["cell"] = done["cell"]
        evidence = base / "predecessor.json"
        evidence.write_bytes(b"frozen predecessor")
        cap["predecessor_evidence"] = [dict(path=str(evidence), sha256=sha(evidence), name=evidence.name)]
        capture.write_bytes(encoded(cap))
        done["source_capture_sha256"] = sha(capture)
        (native / "RESULT.json").write_bytes(encoded(done))
        request = base / "request/REQUEST.json"
        request.parent.mkdir()
        request.write_bytes(encoded(dict(cell=done["cell"], host="i134")))
        request.with_name("STATUS.json").write_bytes(encoded(dict(stage="running", request_sha256=sha(request))))
        retained = root / "continuation-evidence/predecessors" / evidence.name
        retained.parent.mkdir(parents=True)
        retained.write_bytes(evidence.read_bytes())
        return study, root, capture, native, request, retained

    def test_completed_native_is_collected_without_training_and_retains_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as directory:
            study, root, capture, native, request, retained = self.fixture(Path(directory))
            pin = sha(retained)
            with patch("shaping_reboot_recovery.socket.gethostname", return_value="i134"), \
                    patch("shaping_drain_resume.resume", side_effect=AssertionError("training replay")):
                result = work(study, capture, native, request)
            self.assertEqual(result["completed_epochs_replayed"], 0)
            self.assertEqual(sha(retained), pin)
            self.assertEqual(json.loads(request.with_name("STATUS.json").read_text())["stage"], "trained")
            self.assertEqual(json.loads((root / "RESULT.json").read_text())["exit_code"], 0)

    def test_changed_predecessor_stays_protected(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as directory:
            study, root, capture, native, request, retained = self.fixture(Path(directory))
            retained.write_bytes(b"changed")
            with patch("shaping_reboot_recovery.socket.gethostname", return_value="i134"), \
                    patch("shaping_drain_resume.resume", side_effect=AssertionError("training replay")):
                with self.assertRaisesRegex(ValueError, "different predecessor"):
                    work(study, capture, native, request)
            self.assertEqual(retained.read_bytes(), b"changed")
            self.assertFalse((root / "RESULT.json").exists())

    def test_failed_native_receipt_cannot_replay_or_publish(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as directory:
            study, root, capture, native, request, retained = self.fixture(Path(directory))
            done = json.loads((native / "RESULT.json").read_text())
            done["exit_code"] = 1
            (native / "RESULT.json").write_bytes(encoded(done))
            with patch("shaping_reboot_recovery.socket.gethostname", return_value="i134"), \
                    patch("shaping_drain_resume.resume", side_effect=AssertionError("training replay")):
                with self.assertRaisesRegex(ValueError, "successful owned"):
                    work(study, capture, native, request)
            self.assertFalse((root / "RESULT.json").exists())
