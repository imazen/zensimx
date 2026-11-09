from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import shaping_variant_pipeline as owner
from shaping_completion import matrix


class VariantPipelineTests(unittest.TestCase):
    def admission(self, outer):
        return dict(outer=outer, outer_evaluation_admitted=True, reduced_scope_sha256='approved',
                    finalist_roles=dict(relu='relu', best_analytic='softplus',
                                        best_product='product_joint_identity',
                                        best_separate='separate_pretrained_frozen',
                                        extra_trees='extra_trees', shaped_additive='shaped_additive'))

    def test_exact_registered_reduced_coverage_with_no_omitted_diagnostics(self):
        cells = matrix()
        for fold in range(3):
            admission = self.admission(fold)
            with patch('shaping_execution.load_registration', return_value=({}, cells, {})), \
                 patch('shaping_outer_admission.resolve', return_value=(admission, Path('admission'))):
                selected = [c for stage in owner.STAGES for c in owner.selected(Path('study'), fold, stage)[0]]
            self.assertEqual(len(selected), 64)
            self.assertEqual({c['stage'] for c in selected}, {'human_only', 'source_curve', 'shuffle'})
            self.assertTrue(all(c['fraction'] in (.25, .5) and c['order_seed'] == 20261107
                                for c in selected if c['stage'] == 'source_curve'))
            self.assertTrue(all(c['permutation'] == 'population' and c['permutation_seed'] == 20261107
                                for c in selected if c['stage'] == 'shuffle'))
            self.assertEqual(len({c['id'] for c in selected}), 64)

    def test_missing_paired_seed_blocks_registration(self):
        cells = matrix()
        missing = next(c for c in cells if c['stage'] == 'human_only' and c['partition'] == 'outer-0')
        cells.remove(missing)
        with patch('shaping_execution.load_registration', return_value=({}, cells, {})), \
             patch('shaping_outer_admission.resolve', return_value=(self.admission(0), Path('admission'))):
            with self.assertRaisesRegex(ValueError, 'complete registered variant matrix'):
                owner.selected(Path('study'), 0, 'human_only')

    def test_no_registration_before_inner_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch)
            args = SimpleNamespace(study=root / 'study', output=root / 'out', outer=0)
            with patch.object(owner, 'register_fits') as register, \
                 patch.object(owner.time, 'sleep', side_effect=RuntimeError('observed waiting')):
                with self.assertRaisesRegex(RuntimeError, 'observed waiting'):
                    owner.watch(args)
                register.assert_not_called()


if __name__ == '__main__':
    unittest.main()
