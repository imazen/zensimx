import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import shaping_outer_prepare_worker as owner


class OuterPreparationTests(unittest.TestCase):
    def test_admission_failure_creates_no_request_or_computation(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch)
            with patch('shaping_reduced_scope.admit_compute'), patch('shaping_outer_admission.resolve', side_effect=ValueError('missing paired evidence')):
                with self.assertRaisesRegex(ValueError, 'paired evidence'):
                    owner.register(root, 0, 'r5600g', root / 'request')
            self.assertFalse((root / 'request').exists())

    def test_existing_different_input_refused_even_when_missing_inputs_allowed(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch)
            data = root / 'input'; data.write_bytes(b'frozen')
            q = dict(schema='shaping-outer-inputs-worker-v1', host='r5600g', models_fitted=0,
                     driver_sha256=owner.sha(owner.__file__), files={str(data): dict(sha256=owner.sha(data), bytes=6)})
            request = root / 'REQUEST.json'; request.write_text(json.dumps(q))
            with patch('socket.gethostname', return_value='r5600g'):
                owner.verify(request)
                data.write_bytes(b'drift!')
                with self.assertRaisesRegex(ValueError, 'preserve different'):
                    owner.verify(request, True)

    def test_worker_gate_mismatch_cannot_take_cpu_or_start_preparation(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch); admission = root / 'admission'; admission.write_text('frozen')
            q = dict(study=str(root), outer=0, admission_sha256='wrong', plan_sha256='wrong')
            with patch.object(owner, 'verify', return_value=q), patch('shaping_outer_admission.resolve', return_value=({}, admission)), \
                 patch('shaping_rescue_dispatch.reserve_worker') as reserve, patch('shaping_outer_execution.prepare') as prepare:
                with self.assertRaisesRegex(ValueError, 'admission changed'):
                    owner.prepare(root / 'REQUEST.json')
                reserve.assert_not_called(); prepare.assert_not_called()

    def test_receiver_reserves_two_independent_gib_and_verifies_after_copy(self):
        from contextlib import contextmanager
        entries = []
        @contextmanager
        def reserve(study, destination, minimum, *, bytes_required):
            entries.append((destination, bytes_required))
            yield
        with patch.object(owner, 'verify', return_value={'study': '/study'}) as verify, \
             patch('shaping_storage.reservation', side_effect=reserve), patch.object(owner.subprocess, 'run') as run:
            owner.receive(Path('/capsule/REQUEST.json'), ['--server', '-r', '.', '/'])
        self.assertEqual([n for _, n in entries], [1024**3, 1024**3])
        self.assertNotEqual(entries[0][0], entries[1][0])
        self.assertEqual(verify.call_count, 2); run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
