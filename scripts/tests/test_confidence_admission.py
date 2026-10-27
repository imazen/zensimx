import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "byconstruction"))
import confidence_admission as admission


class AdmissionQueue(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_prior_exit_requires_matching_selection(self):
        self.save("TRAIN_SELECTION.json", {"finalist": "Mlic"})
        self.save("QUEUE_EXIT.json", {"returncode": 0})
        self.save("TRAIN_SCREEN_COMPLETE.json", {"selection_sha256": admission.sha(self.root / "TRAIN_SELECTION.json")})
        self.assertTrue(admission.after_finished(self.root))
        self.save("TRAIN_SELECTION.json", {"finalist": "changed"})
        with self.assertRaisesRegex(ValueError, "completion"):
            admission.after_finished(self.root)

    def test_stale_queue_is_not_a_live_dependency(self):
        self.save("QUEUED.json", {"queue_process": {"pid": 0, "start_ticks": 0}})
        with self.assertRaisesRegex(RuntimeError, "without a completion"):
            admission.after_finished(self.root)

    def test_source_change_refuses_native_work(self):
        source = self.root / "source.py"
        source.write_text("original")
        self.save("plan.json", {"schema": "mlic-confidence-native-admission-plan-v1",
                  "quality_responses_permitted": False,
                  "packed_binary": str(source),
                  "coordinator_sha256": admission.sha(admission.__file__),
                  "input_sha256": {str(source): admission.sha(source)}})
        admission.checked_plan(self.root / "plan.json")
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            admission.checked_plan(self.root / "plan.json")

    def test_packed_reader_is_checked_before_work(self):
        source = self.root / "native-packed"
        source.write_bytes(b"frozen binary identity")
        plan = {"schema": "mlic-confidence-native-admission-plan-v1",
                "quality_responses_permitted": False, "packed_binary": str(source),
                "coordinator_sha256": admission.sha(admission.__file__), "input_sha256": {}}
        self.save("plan.json", plan)
        with self.assertRaisesRegex(ValueError, "hash-bound"):
            admission.checked_plan(self.root / "plan.json")
        plan["input_sha256"][str(source)] = admission.sha(source)
        self.save("plan.json", plan)
        admission.checked_plan(self.root / "plan.json")
        source.rename(self.root / "preserved-native-packed")
        with self.assertRaisesRegex(ValueError, "existing"):
            admission.checked_plan(self.root / "plan.json")

    def test_finished_state_requires_actual_artifact(self):
        self.save("plan.json", {"screen": str(self.root / "screen")})
        self.save("ADMISSION_STATE.json", {
            "stage": "screen-complete-requires-membership-freeze",
            "queue_process": {"pid": 0, "start_ticks": 0}, "failure": None,
            "screen_summary_sha256": "missing", "run_directory": str(self.root),
            "completed_stages": ["decode", "screen"], "required_stages": 2,
            "next_action": "freeze"})
        with contextlib.redirect_stdout(io.StringIO()):
            admission.status(self.root / "plan.json")
        current = json.loads((self.root / "CURRENT_RUN.json").read_text())
        self.assertEqual(current["status"], "failed")
        self.assertIn("missing", current["failure"])


if __name__ == "__main__":
    unittest.main()
