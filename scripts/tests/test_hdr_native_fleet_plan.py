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
                                   bin=binary, hdr_teacher=None, hdr_fleet_keys=keys, upiq_native_admission=None,
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

    def test_ladder_plan_keeps_rung_mapping_roles_and_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as td:
            root = Path(td)
            (root / 'artifacts/zenavif').mkdir(parents=True)
            cells, sources = [], []
            for si, source in enumerate(['1001', '1002']):
                (root / 'views' / source).mkdir(parents=True)
                ref = root / 'views' / source / 'm64.png'
                ref.write_bytes(('ref' + source).encode())
                sources.append(dict(source=source, consumer_role='train' if si == 0 else 'eval'))
                for k in range(5):
                    enc = hashlib.sha256(f'{source}-{k}'.encode()).hexdigest()
                    png = ('rend' + source + str(k)).encode()
                    (root / 'artifacts/zenavif' / f'{enc}.pq.png').write_bytes(png)
                    cells.append(dict(view_id=f'{source}_m64', source=source, kind='mitchell-linear-pq16', encoded_sha256=enc,
                                      rendition_sha256=hashlib.sha256(png).hexdigest(),
                                      reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                                      chroma_q=[k, k], quality=95, kept_chroma=k != 2))
            import pyarrow as pa, pyarrow.parquet as pq
            table = root / 'CELLS.parquet'
            pq.write_table(pa.Table.from_pylist(cells), table)
            roles = root / 'ORIGIN_ROLES.json'
            roles.write_text(json.dumps(dict(schema='hdr-chroma-consumer-origin-roles-v1', sources=sources)))
            binary, executor = root / 'native-bin', root / 'executor.py'
            binary.write_bytes(b'x')
            executor.write_bytes(b'y')
            args = SimpleNamespace(revision=7, era='rev7_rowlanes', limit=0, hdr_fleet_executor=executor, bin=binary,
                                   hdr_teacher=None, hdr_fleet_keys=None, upiq_native_admission=None, hdr_ladder_cells=table,
                                   hdr_ladder_roles=roles, hdr_ladder_root=root, out=root, set='ladders', build_commit='fixture-build')
            old = dict(bank.LADDER_EXPECT)
            bank.LADDER_EXPECT.update(rungs=8, train=1, eval=1)
            try:
                self.assertEqual(bank.cmd_hdr_fleet_plan(args), 0)
                spec = json.loads((root / 'ladders/views.spec.json').read_text())
                mapping = json.loads((root / 'ladders/membership-feature-map.json').read_text())
                self.assertEqual(len(spec['views']), 2)
                self.assertEqual(len(mapping), 8)
                self.assertEqual({m['consumer_role'] for m in mapping}, {'train', 'eval'})
                self.assertFalse(any('natural' in json.dumps(m) for m in mapping))
                for view in spec['views']:
                    c = json.loads(view['cell']['knob_tuple_json'])
                    self.assertEqual(len(c['candidates']), 4)
                    self.assertTrue(all(p.endswith('.pq.png') for p in c['candidate_paths']))
                # a changed reference must refuse before any plan is written
                (root / 'views/1001/m64.png').write_bytes(b'changed')
                args.set = 'forged'
                with self.assertRaisesRegex(ValueError, 'reference changed'):
                    bank.cmd_hdr_fleet_plan(args)
                self.assertFalse((root / 'forged/views.spec.json').exists())
            finally:
                bank.LADDER_EXPECT.update(old)


if __name__ == '__main__':
    unittest.main()
