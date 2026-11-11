import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import shaping_local_collection as local
import shaping_rescue_worker as worker
import shaping_control_worker as control


class LocalCollectionTests(unittest.TestCase):
    def test_remote_terminal_directory_cannot_be_sealed_as_local(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as temp:
            root = Path(temp)
            request = root / "REQUEST.json"
            request.write_text(json.dumps(dict(host="remote")))
            with patch.object(local.socket, "gethostname", return_value="local"), patch.object(worker, "verify") as verify:
                with self.assertRaisesRegex(ValueError, "coordinator-local"):
                    local.seal(root, request, root / "assignment", root)
                verify.assert_not_called()

    def test_native_gate_failure_preserves_terminal_files_without_receipt(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as temp:
            study = Path(temp)
            root = study / "runs/c/attempt-001"
            root.mkdir(parents=True)
            request = study / "REQUEST.json"
            request.write_text(json.dumps(dict(host="local", cell=dict(id="c"), execution_kind="control")))
            from shaping_completion import sha
            (root / "WORKER_EXECUTION.json").write_text(json.dumps(dict(request_sha256=sha(request))))
            with patch.object(local.socket, "gethostname", return_value="local"), patch.object(worker, "verify"), \
                 patch.object(control, "validate_result", side_effect=ValueError("native parity drift")) as gate:
                with self.assertRaisesRegex(ValueError, "native parity drift"):
                    local.seal(study, request, study / "assignment", study)
                gate.assert_called_once()
            self.assertTrue((root / "WORKER_EXECUTION.json").exists())
            self.assertFalse((root / "WORKER_COLLECTION.json").exists())


if __name__ == "__main__":
    unittest.main()
