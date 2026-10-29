"""Registered native-HDR TRAIN legs: admission accepts the exact registered contract and refuses each deviation."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import subjective_training_leg as leg


def sha(b):
    return hashlib.sha256(b).hexdigest()


class HdrLeg(unittest.TestCase):
    def fixture(self, root):
        name = 'hdr_4kdtb_train'
        human, mode, loss, transform, rows, commit, extractor = leg.HDR_LEGS[name]
        bank = json.dumps(dict(schema='hdr-native-rev7-bank-v1', build_commit=commit, labels_read=False,
                               research=dict(feature_set_id=leg.HDR_FEATURES))).encode()
        (root / 'bank.json').write_bytes(bank)
        refs = ['4kdtb:S%d' % (i // 12) for i in range(rows)]
        keys = [dict(row=i, observation=f'o{i}', source=refs[i], reference=refs[i], family='hdr:' + refs[i], role='fit', tier='T2')
                for i in range(rows)]
        kb = json.dumps(keys).encode()
        (root / 'keys.json').write_bytes(kb)
        counts = {}
        for r in refs:
            counts[r] = counts.get(r, 0) + 1
        nominal = 0.5
        weight = nominal / (sum(1.0 - 1.0 / n for n in counts.values()) / len(counts))
        decl = dict(schema='hdr-native-leg-v1', leg=name, loss=loss, nominal_weight=nominal, target_orientation='quality',
                    target_transform=transform, target_column='human_score', selection='fixed-final-epoch', formula_revision=7,
                    tier='v3', feature_ids=[1, 2], feature_set_id=leg.HDR_FEATURES, producer_commit=commit, extractor_sha256=extractor,
                    native_role='TRAIN', rows=rows, table_sha256='t', keys_sha256=sha(kb), keys_path=str(root / 'keys.json'),
                    feature_bank_manifest_path=str(root / 'bank.json'), feature_bank_manifest_sha256=sha(bank))
        rec = dict(name=name, teacher=not human, subjective=human, train_weight=weight, val_weight=0.0, mode=mode, table_sha256='t',
                   keys_sha256=sha(kb), rows=rows, source_counts=counts)
        return rec, decl

    def test_registered_leg_admits_and_each_deviation_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rec, decl = self.fixture(root)
            leg.admit_hdr_leg(rec, decl, [1, 2], set())
            cases = {
                'loss': lambda r, d: d.update(loss='within-source-rank-and-scale'),
                'producer': lambda r, d: d.update(producer_commit='0' * 40),
                'extractor': lambda r, d: d.update(extractor_sha256='0' * 64),
                'features': lambda r, d: d.update(feature_ids=[1]),
                'teacher flag': lambda r, d: r.update(teacher=True),
                'mode': lambda r, d: r.update(mode='withinref,both'),
                'weight': lambda r, d: r.update(train_weight=r['train_weight'] * 1.01),
                'val weight': lambda r, d: r.update(val_weight=0.1),
                'rows': lambda r, d: (r.update(rows=47), d.update(rows=47)),
                'unregistered': lambda r, d: r.update(name='hdr_other'),
            }
            for label, mutate in cases.items():
                r, d = copy.deepcopy(rec), copy.deepcopy(decl)
                mutate(r, d)
                with self.subTest(label), self.assertRaises(ValueError):
                    leg.admit_hdr_leg(r, d, [1, 2], set())
            with self.assertRaises(ValueError):
                leg.admit_hdr_leg(rec, decl, [1, 2], {'hdr:4kdtb:S0'})


if __name__ == '__main__':
    unittest.main()
