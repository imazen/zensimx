"""Construction and native export controls; no scientific dataset access."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import tree_study as t
import numpy as np


class TreeStudyTests(unittest.TestCase):
    def test_source_membership_and_alias(self):
        refs = np.array([t.canonical_ref(f'src{i}_part{p}') for i in range(9) for p in 'ab'])
        folds = t.fold_ids(np.full(18, 'konfig'), refs)
        self.assertEqual(folds[::2].tolist(), folds[1::2].tolist())
        self.assertEqual(set(folds), {0, 1, 2})
        for fold in range(3):
            self.assertFalse(set(refs[folds == fold]) & set(refs[folds != fold]))

    def test_reference_balanced_targets(self):
        y = np.array([0., 0., 1., 2.])
        refs = np.array(['a', 'a', 'a', 'b'])
        w, target, maps = t.weights_and_targets(y, np.full(4, 'human'), refs, 'cdf', 1.)
        self.assertAlmostEqual(w[:3].sum(), w[3])
        np.testing.assert_allclose(target, [1 / 6, 1 / 6, 5 / 12, 3 / 4])
        self.assertEqual(maps['human']['references'], 2)
        _, ranged, _ = t.weights_and_targets(y, np.full(4, 'human'), refs, 'range', 1.)
        np.testing.assert_array_equal(ranged, [0., 0., .5, 1.])

    def test_registered_search(self):
        c = t.candidates()
        self.assertEqual(len(c), 38)
        self.assertEqual(len({r['id'] for r in c}), len(c))
        for r in c:
            model = t.estimator(r, 11, 3)
            if r['family'].startswith('hgb'):
                self.assertFalse(model.early_stopping)
            if r['family'].endswith('_mono'):
                self.assertEqual(r['features'], 'direct')
                self.assertEqual(model.monotonic_cst, [-1] * 3)

    def test_native_export_and_rejected_wrong_reference(self):
        binary = Path(os.environ['TREE_PARITY_BIN']).resolve()
        root = Path(os.environ['TMPDIR'])
        with tempfile.TemporaryDirectory(prefix='tree-study-controls-', dir=root) as temporary:
            root = Path(temporary)
            x = np.random.default_rng(27).uniform(0, 2, (256, 3)).astype(np.float32)
            y = 1 - x[:, 0] / 4 - x[:, 1] ** 2 / 8 + np.sin(x[:, 2]) / 20
            for family in ('rf', 'extra', 'hgb', 'rf_mono', 'hgb_mono'):
                config = dict(family=family, params=dict(min_samples_leaf=4))
                model = t.estimator(config, 13, 3)
                if family.startswith('hgb'):
                    model.set_params(max_iter=8)
                else:
                    model.set_params(n_estimators=8)
                model.fit(x, y)
                # f32 neighbors of thresholds exercise branch boundaries.
                thresholds = []
                if hasattr(model, 'estimators_'):
                    for tree in model.estimators_:
                        for feature, threshold in zip(tree.tree_.feature, tree.tree_.threshold):
                            if feature >= 0:
                                for side in (-np.inf, np.inf):
                                    row = x[0].copy()
                                    row[feature] = np.nextafter(np.float32(threshold), np.float32(side))
                                    thresholds.append(row)
                check = np.concatenate([np.zeros((1, 3), dtype=np.float32), x, np.array(thresholds).reshape(-1, 3)])
                for anchored in (False, True):
                    path = root / (family + str(anchored))
                    values, _ = t.native_gate(model, check, path, binary, dict(synthetic=True), anchored)
                    if anchored:
                        self.assertEqual(values[0], 1.)
                    else:
                        self.assertLessEqual(np.max(np.abs(values - model.predict(check))), 1e-12)
                # Corrupting a reference value must fail the unchanged zero-ULP gate.
                path = root / (family + 'False')
                with np.load(path.with_suffix('.parity.npz')) as z:
                    fields = dict(z)
                fields['test_raw'][0] += .01
                bad = root / (family + '-negative.npz')
                np.savez_compressed(bad, **fields)
                proc = subprocess.run([str(binary), '--head', str(path.with_suffix('.opaque-vector.zcth')),
                    '--parity', str(bad), '--set', 'test'], capture_output=True, text=True)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn('corrhead_parity: FAIL', proc.stderr)


if __name__ == '__main__':
    unittest.main()
