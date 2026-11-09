from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import shaping_terminal_transport as owner


class TerminalTransportTests(unittest.TestCase):
    def test_first_control_import_creates_its_parent_after_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch); study = root / 'study'; study.mkdir()
            result = study / 'runs/control/attempt-001'
            request = root / 'REQUEST.json'
            q = dict(study=str(study), result_output=str(result), resolution={'paired': True},
                     host='r3500', cell={'id': 'control'})
            request.write_text(json.dumps(q))
            receipt = dict(request_sha256=owner.sha(request), cell=q['cell'], pack_driver_sha256='p',
                           files={'model': {'bytes': 6}})
            @contextmanager
            def reservation(*args, **kwargs):
                yield
            def unpack(bundle, staging, pin):
                staging.mkdir(parents=True); (staging / 'model').write_bytes(b'native')
                return receipt
            def reproduce(record, *, host):
                self.assertFalse(result.parent.exists())
                return q['resolution'], []
            with patch('shaping_storage.reservation', side_effect=reservation), \
                 patch('shaping_worker_collect.unpack', side_effect=unpack):
                report = owner.collect(request, root / 'bundle', 'pin', reproduce,
                                       lambda record, path: self.assertEqual((path / 'model').read_bytes(), b'native'))
            self.assertEqual((result / 'model').read_bytes(), b'native')
            self.assertEqual(report['files'], 1)
            self.assertTrue((result / 'WORKER_COLLECTION.json').exists())


if __name__ == '__main__':
    unittest.main()
