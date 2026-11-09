from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from shaping_gradient_report import report
from shaping_completion import sha


class GradientReportTests(unittest.TestCase):
    def test_noisy_rounds_remove_both_arms_and_incomplete_batches_stay_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch); (root / 'inputs').mkdir(); binary = root / 'inputs/probe'; binary.write_bytes(b'probe')
            (root / 'inputs/BUILD.json').write_text(json.dumps(dict(binary_sha256=sha(binary), source_sha256='source')))
            for batch in range(2):
                directory = root / f'timing-v{batch + 1}'; directory.mkdir()
                terminal = dict(features=984, hidden=128, preflight_exact=True, training_changed=False,
                                unreliable=False, rounds=32, complete=False)
                (directory / 'RESULT.json').write_text(json.dumps(terminal))
                samples = [dict(iterations=1, execution_order=[i % 2, 1 - i % 2],
                    gate_clean=i < 27, elapsed_ns=[200, 100] if i < 27 else [1, 1000000],
                    compensated_ns=[200, 100] if i < 27 else [1, 1000000]) for i in range(32)]
                comparison = dict(group_name='984_features_128_hidden_two_gradients', completed_rounds=32,
                    benchmarks=[dict(name='two_passes'), dict(name='paired_pass')], samples=samples)
                (directory / 'zenbench.json').write_text(json.dumps(dict(unreliable=False, comparisons=[comparison])))
            output = root / 'report.json'
            with contextlib.redirect_stdout(io.StringIO()):report(root, binary, output)
            result = json.loads(output.read_text())
            self.assertEqual(result['retained_round_indices'], list(range(27)) + list(range(32, 37)))
            self.assertEqual(result['paired_round_time_ratio']['median'], .5)
            self.assertEqual(result['normalized_compensated_call_ns'], [[200., 100.]] * 32)
            self.assertFalse(json.loads((root / 'timing-v1/RESULT.json').read_text())['complete'])
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'binary changed'):report(root, binary, root / 'bad.json')


if __name__ == '__main__':
    unittest.main()
