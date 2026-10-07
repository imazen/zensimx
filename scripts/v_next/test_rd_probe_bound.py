"""Guard-only regressions: synthetic metadata, never a data population."""
import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import rd_probe_bound as bound


class FrozenModelGuard(unittest.TestCase):
    def test_wrong_pin_refuses_before_measurements(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as t:
            root = Path(t)
            (root / 'INPUTS.json').write_text(json.dumps({
                'schema': 'native-jxl-interventions-v1', 'model_sha256': '1' * 64}))
            (root / 'COMPLETE.json').write_text('{}')
            code = bound.bound_source()
            scope = {'__name__': 'guard_test', '__file__': str(bound.OWNER), 're': re}
            exec(compile(code, str(bound.OWNER), 'exec'), scope)
            with patch.object(sys, 'argv', ['test', '--interventions', str(root),
                                           '--expected-model-sha256', '2' * 64]):
                with self.assertRaisesRegex(AssertionError, 'differs from frozen pin'):
                    scope['interventions_main']()
            with patch.object(sys, 'argv', ['test', '--interventions', str(root),
                                           '--expected-model-sha256', '1' * 64]):
                with self.assertRaises(FileNotFoundError) as caught:
                    scope['interventions_main']()
                self.assertIn('measurements.jsonl', str(caught.exception))
            with patch.object(sys, 'argv', ['test', '--interventions', str(root),
                                           '--expected-model-sha256', 'invalid']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    scope['interventions_main']()


if __name__ == '__main__':
    unittest.main()
