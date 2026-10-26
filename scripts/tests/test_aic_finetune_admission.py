"""D6 admission keeps the exact public_train source folds and rejects other roles."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2_human_role as role


class D6Admission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(self.tmp.cleanup)
        self.assignment = {f'source{i}': i // 4 for i in range(28)}
        path = Path(self.tmp.name) / 'assignment.json'
        path.write_text(json.dumps({'assignment': self.assignment}))
        pins = {**role.D6_INPUT_PINS, 'assignment_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.patch = patch.dict(role.D6_INPUT_PINS, pins)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.keys = [dict(row=j, key=f'{s}-{i}', source=s, split='public_train', source_fold=f)
                     for j, (s, f, i) in enumerate((s, f, i) for s, f in self.assignment.items()
                                                   if f != 2 for i in range(140))]
        self.d = dict(schema='aic2026-public-train-rev7-v1', ruling='D6', split='public_train', role='TRAIN',
                      use='fit', formula_revision=7, feature_set_id='basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af',
                      target_column='human_score', target_transform='negate-JND_score', omitted_source_fold=2,
                      rows=len(self.keys), sources=sorted({r['source'] for r in self.keys}),
                      assignment_path=str(path), **pins)

    def test_registered_fold_is_admitted(self):
        role.aic_public_train_declaration(self.d, self.keys)

    def test_other_populations_refused_before_assignment_or_keys(self):
        for field, value in [('split', 'public_test'), ('split', 'aic3'), ('role', 'EVAL'),
                             ('ruling', 'D1'), ('formula_revision', 6), ('use', 'out-of-fold-score'),
                             ('source_table_sha256', '0' * 64)]:
            with self.subTest(field=field, value=value):
                with patch.object(role, 'sha', side_effect=AssertionError('assignment opened')):
                    with self.assertRaisesRegex(ValueError, 'D6'):
                        role.aic_public_train_declaration({**self.d, field: value}, None)

    def test_key_role_and_source_exclusion_cannot_be_relabelled(self):
        for field, value in [('split', 'public_test'), ('source_fold', 2), ('row', 1),
                             ('source', 'unregistered'), ('key', self.keys[1]['key'])]:
            keys = copy.deepcopy(self.keys)
            keys[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'D6'):
                role.aic_public_train_declaration(self.d, keys)

    def test_assignment_tampering_is_refused(self):
        Path(self.d['assignment_path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'assignment pin'):
            role.aic_public_train_declaration(self.d, self.keys)

    def test_d1_default_remains_closed_to_aic(self):
        with self.assertRaisesRegex(ValueError, 'AIC'):
            role.bank_members({'bank_manifest_sha256': {'aic2026_public_train': 'a' * 64}})


if __name__ == '__main__':
    unittest.main()
