from pathlib import Path
import json, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_cost_report import samples, summarize
from shaping_completion import sha


class CostReportTests(unittest.TestCase):
    def setUp(self):
        scratch = Path.home()/'tmp'; scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workload = self.root/'WORKLOAD.json'
        self.write(self.workload, dict(schema='shaping-extract-cost-v1', admission_sha256='a'*64,
                                      size=256, threads=1, raw_ids=[1, 2], pixel_sha256=['b'*64]*3))
        w = json.loads(self.workload.read_text())
        self.write(self.root/'IDENTITY.json', dict(w, workload_sha256=sha(self.workload),
                   mode='bench', preflight_strided_exact=True))
        # Each batch lists the arms in a different order. A rejected first round
        # has deliberately enormous timings and must not affect any estimate.
        for batch, order in enumerate(([2, 0, 1], [1, 2, 0])):
            rows = []
            for i in range(17 if batch == 0 else 16):
                clean = not (batch == 0 and i == 0)
                iterations = 2+i
                durations = [((j+1)*100 if clean else 9000000)*iterations for j in order]
                rows.append(dict(iterations=iterations, execution_order=[1, 2, 0],
                                 gate_clean=clean, elapsed_ns=durations, compensated_ns=durations))
            self.write(self.root/f'zenbench-{batch:03}.json', dict(unreliable=False,
                       comparisons=[dict(group_name='shaping_native_extraction', benchmarks=[dict(name=f'pair-{j}') for j in order],
                                         completed_rounds=len(rows), samples=rows)]))
        self.write(self.root/'RESULT.json', dict(complete=True, timing_measured=True,
                   postflight_exact=True, unreliable=False, batches=2,
                   gate_clean=[False]+[True]*32, retained_round_indices=list(range(1, 33))))

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def change(self, name, modify):
        p = self.root/name; d = json.loads(p.read_text()); modify(d); self.write(p, d)

    def test_remaps_arms_normalizes_actual_counts_and_drops_whole_noisy_round(self):
        names, values, vector, _, _, retained = samples(self.root, self.workload)
        self.assertEqual(names, ['pair-0', 'pair-1', 'pair-2'])
        np.testing.assert_array_equal(values, np.tile([100., 200., 300.], (32, 1)))
        self.assertFalse(vector); self.assertEqual(retained, list(range(1, 33)))
        result = summarize(names, values, True)
        self.assertEqual(result['arms']['pair-0']['call_ns']['median'], 100)
        self.assertEqual(result['arms']['pair-0']['forty_row_batch_per_row_arithmetic_mean_ns']['median'], 2.5)
        self.assertEqual(result['paired_ratios'][0]['paired_round_time_ratio']['percentile_95_interval'], [2., 2.])

    def test_rejects_receipt_dropping_or_replacing_clean_round(self):
        self.change('RESULT.json', lambda d: d.update(retained_round_indices=list(range(32))))
        with self.assertRaisesRegex(ValueError, 'thirty-two'): samples(self.root, self.workload)

    def test_rejects_raw_gate_disagreement(self):
        self.change('zenbench-001.json', lambda d: d['comparisons'][0]['samples'][0].update(gate_clean=False))
        with self.assertRaisesRegex(ValueError, 'thirty-two'): samples(self.root, self.workload)

    def test_rejects_incomplete_pair(self):
        self.change('zenbench-001.json', lambda d: d['comparisons'][0]['samples'][0].update(execution_order=[0, 0, 2]))
        with self.assertRaisesRegex(ValueError, 'paired round'): samples(self.root, self.workload)

    def test_rejects_qualification_as_timing(self):
        self.change('IDENTITY.json', lambda d: d.update(mode='qualify'))
        with self.assertRaisesRegex(ValueError, 'complete reliable'): samples(self.root, self.workload)

    def test_rejects_workload_drift(self):
        self.change('WORKLOAD.json', lambda d: d.update(threads=4))
        with self.assertRaisesRegex(ValueError, 'identity'): samples(self.root, self.workload)

    def test_rejects_raw_unreliability(self):
        self.change('zenbench-000.json', lambda d: d.update(unreliable=True))
        with self.assertRaisesRegex(ValueError, 'unreliable'): samples(self.root, self.workload)

    def test_vector_models_are_rehashed_before_reporting(self):
        model = self.root/'model.bin'; model.write_bytes(b'test-only-model')
        w = dict(schema='shaping-cost-v1', admission_sha256='a'*64,
                 arms=[dict(id=f'arm-{i}', model='model.bin', sha256=sha(model)) for i in range(16)])
        self.write(self.workload, w)
        self.change('IDENTITY.json', lambda d: d.update(workload_sha256=sha(self.workload), rows_per_call=40, preflight_exact=True))
        self.change('RESULT.json', lambda d: d.update(preflight_exact=True))
        for batch in range(2):
            def expand(d):
                c = d['comparisons'][0]; c['group_name'] = 'shaping_native_40_rows'
                c['benchmarks'] = [dict(name=f'arm-{i}') for i in range(16)]
                for row in c['samples']:
                    row.update(execution_order=list(range(15, -1, -1)),
                               elapsed_ns=[row['iterations']*(i+1) for i in range(16)],
                               compensated_ns=[row['iterations']*(i+1) for i in range(16)])
            self.change(f'zenbench-{batch:03}.json', expand)
        names, values, vector, _, _, _ = samples(self.root, self.workload)
        self.assertTrue(vector); self.assertEqual(len(names), 16)
        np.testing.assert_array_equal(values[0], np.arange(1, 17))
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'model changed'): samples(self.root, self.workload)


if __name__ == '__main__': unittest.main()
