"""Native feature declarations preserve condition membership when bytes coincide."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import rev5_bank as bank


class NativePlan(unittest.TestCase):
    def test_duplicate_pixels_keep_condition_mapping_and_bound_controls(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as td:
            root = Path(td)
            rows = []
            for i in range(96):
                family = i // 48
                digest = hashlib.sha256(str(i // 2).encode()).hexdigest()
                rows.append(dict(dataset_id='rousselot-4kdtb', role='TRAIN' if family == 0 else 'EVAL',
                                 ref_path=f'/unopened/ref{family}.hdr', dist_path=f'/unopened/dist{i}.hdr',
                                 ref_sha256=hashlib.sha256(str(family).encode()).hexdigest(), dist_sha256=digest))
            keys = root / 'memberships.json'
            keys.write_text(json.dumps(rows))
            manifest = root / '_MANIFEST.json'
            manifest.write_text(json.dumps(dict(schema='hdr-human-source-membership-v1', memberships_sha256=bank.sha256_file(keys))))
            binary, executor = root / 'native-bin', root / 'executor.py'
            binary.write_bytes(b'transport identity fixture')
            executor.write_bytes(b'executor identity fixture')
            args = SimpleNamespace(revision=7, era='rev7_rowlanes', limit=0, hdr_fleet_executor=executor,
                                   bin=binary, hdr_teacher=None, hdr_fleet_keys=keys,
                                   out=root, set='plan', build_commit='fixture-build')
            self.assertEqual(bank.cmd_hdr_fleet_plan(args), 0)
            spec = json.loads((root / 'plan/views.spec.json').read_text())
            mapping = json.loads((root / 'plan/membership-feature-map.json').read_text())
            self.assertEqual(len(spec['views']), 2)
            self.assertEqual(len(mapping), 96)
            self.assertEqual([r['source_index'] for r in mapping], list(range(96)))
            for view in spec['views']:
                controls = json.loads(view['cell']['knob_tuple_json'])
                self.assertEqual(len(controls['candidates']), 24)
                self.assertIn(hashlib.sha256(view['cell']['knob_tuple_json'].encode()).hexdigest(), view['inputs'])
                self.assertEqual(view['requires'], ['hdr-native-rev7-features-v1'])
            keys.write_text(json.dumps(rows) + ' ')
            args.set = 'forged'
            with self.assertRaisesRegex(ValueError, 'receipt differs'):
                bank.cmd_hdr_fleet_plan(args)
            self.assertFalse((root / 'forged').exists())


if __name__ == '__main__':
    unittest.main()
