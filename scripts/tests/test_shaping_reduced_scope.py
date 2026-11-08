import copy
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from shaping_completion import FAMILIES, encoded, matrix, sha
import shaping_reduced_scope as reduced
from shaping_rescue_selection import aggregate as rescue_aggregate
from shaping_selection import HUMANS


def decisions(outer):
    return {f: dict(family=f, outer=outer, setting=0, optimizer_choice='base',
                    inner_score=.8, outer_results_read=False) for f in FAMILIES}


class ReducedScopeTests(unittest.TestCase):
    def test_evaluation_bootstrap_and_cost_refuse_before_native_work(self):
        from types import SimpleNamespace
        from shaping_evaluation import evaluate
        from shaping_reexport import reexport
        from shaping_bootstrap import run
        from shaping_cost_finalists import prepare
        args=SimpleNamespace(study=Path('not-admitted'))
        with patch('shaping_reduced_scope.admit_compute',side_effect=ValueError('host outside approved pool')),patch('subprocess.run') as launch,patch('numpy.load') as load:
            for operation in (lambda:evaluate(args),lambda:reexport(args),lambda:run(args),lambda:prepare(args.study,None,None)):
                with self.assertRaisesRegex(ValueError,'host outside'):operation()
            launch.assert_not_called();load.assert_not_called()

    def test_reduced_rescue_selection_keeps_every_fold_and_seed(self):
        cells = matrix()
        selected = [c for c in cells if c['family'] == 'relu'
                    and c['partition'] in ('inner-0-0', 'inner-0-1')
                    and ((c['stage'] == 'inner' and c['setting'] == 2)
                         or (c['stage'] == 'rescue' and c['rescue'] in reduced.RESCUES))]
        reports = {}
        for cell in selected:
            score = .82 if cell.get('rescue') == 'constant_lr' else .8
            reports[cell['id']] = dict(cell=cell, outer_results_read=False,
                inner_validation=dict(selection_score=score,
                    summary={h: dict(pooled=score, within=score) for h in HUMANS}))
        result = rescue_aggregate(cells, reports, 'relu', 0, 2, rescue_choices=reduced.RESCUES)
        self.assertEqual(result['required_reports'], 18)
        self.assertEqual(result['selected_choice'], 'constant_lr')
        self.assertEqual(set(result['choice_scores']), {'base', 'init_half', 'constant_lr'})
        with self.assertRaisesRegex(ValueError, 'all paired'):
            rescue_aggregate(cells, reports, 'relu', 0, 2)
        missing = dict(reports)
        del missing[selected[0]['id']]
        with self.assertRaisesRegex(ValueError, 'all paired'):
            rescue_aggregate(cells, missing, 'relu', 0, 2, rescue_choices=reduced.RESCUES)
        exposed = copy.deepcopy(reports)
        exposed[selected[0]['id']]['outer_results_read'] = True
        with self.assertRaisesRegex(ValueError, 'outer-exposed'):
            rescue_aggregate(cells, exposed, 'relu', 0, 2, rescue_choices=reduced.RESCUES)
        with self.assertRaisesRegex(ValueError, 'registered rescue choices'):
            rescue_aggregate(cells, reports, 'relu', 0, 2, rescue_choices=('constant_lr',))

    def test_only_fixed_relu_can_rescue_before_complete_shortlisting(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory)
            (root / 'REDUCED_SCOPE.json').write_text('{}')
            with patch.object(reduced, 'coverage', return_value={}):
                choices, pins = reduced.rescue_choices(root, 'relu', 0)
                self.assertEqual(choices, reduced.RESCUES)
                self.assertIn(str(root / 'REDUCED_SCOPE.json'), pins)
                with self.assertRaises(FileNotFoundError):
                    reduced.rescue_choices(root, 'softplus', 0)

    def test_exact_1281_original_cells_and_no_mutation(self):
        cells = matrix()
        before = copy.deepcopy(cells)
        selected = reduced.fixed_cells(cells)
        self.assertEqual(len(selected), 1089)
        for outer in range(3):
            result = reduced.fold_cells(cells, decisions(outer), outer)
            selected += result['cells']
            self.assertEqual(len(result['roles']), 6)
            self.assertFalse(result['outer_results_read'])
        self.assertEqual(len({c['id'] for c in selected}), 1281)
        counts = Counter(c['stage'] for c in selected)
        self.assertEqual({stage: counts[stage] for stage in reduced.TARGETS}, reduced.TARGETS)
        self.assertTrue(all(c in cells for c in selected))
        self.assertEqual(cells, before)

    def test_shortlists_are_base_only_complete_unexposed_and_paired(self):
        cells = matrix()
        d = decisions(1)
        d['silu_monotone']['inner_score'] = .81
        result = reduced.fold_cells(cells, d, 1)
        self.assertEqual(result['roles']['best_analytic'], 'silu_monotone')
        self.assertFalse(any(c['family'] == 'softplus' for c in result['cells']))
        for field, value in [('outer_results_read', True), ('optimizer_choice', 'constant_lr'),
                             ('outer', 2), ('family', 'relu')]:
            bad = copy.deepcopy(d)
            bad['silu_monotone'][field] = value
            with self.assertRaises(ValueError):
                reduced.fold_cells(cells, bad, 1)
        del d['softplus']
        with self.assertRaisesRegex(ValueError, 'all twelve'):
            reduced.fold_cells(cells, d, 1)
        missing = result['cells'][0]['id']
        with self.assertRaisesRegex(ValueError, 'paired fold coverage'):
            reduced.fold_cells([c for c in cells if c['id'] != missing], decisions(1), 1)
        with self.assertRaisesRegex(ValueError, 'fixed reduced matrix'):
            reduced.fixed_cells([c for c in cells if c != reduced.fixed_cells(cells)[0]])

    def test_approval_pins_registration_and_preserves_originals(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory)
            for name in ('PLAN.json', 'MATRIX.json', 'PARTITIONS.json'):
                (root / name).write_text('{}\n')
            original = {p: p.read_bytes() for p in root.iterdir()}
            proposal = root / 'PROPOSAL.json'
            proposal.write_bytes(encoded(dict(
                proposed_by_stage=reduced.TARGETS, proposed_core_cells=1281,
                rescue=dict(roles=list(reduced.RESCUE_ROLES), choices=list(reduced.RESCUES)),
                outer_roles=list(reduced.ROLES), original_plan_sha256=sha(root / 'PLAN.json'))))
            approval = root / 'APPROVAL.json'
            record = dict(proposal=str(proposal), proposal_sha256=sha(proposal),
                          answer='Approve reduced plan (recommended)', core_fit_target=1281,
                          interrupted_replays_authorized=False, active_fits_must_finish=True,
                          old_registration_preserved=True,
                          allowed_new_compute_hosts=['r5600g', 'i134', 'r3500', 'r3800x'])
            approval.write_bytes(encoded(record))
            with patch.object(reduced, 'PROPOSAL_SHA256', sha(proposal)), \
                    patch.object(reduced, 'load_registration', return_value=({}, matrix(), {})):
                actual = reduced.register(root, approval)
                self.assertEqual(len(actual['fixed_cells']), 1089)
                self.assertEqual(actual['unresolved_fold_cells'], 192)
                self.assertEqual(json.loads((root / 'REDUCED_SCOPE.json').read_text()), actual)
                coverage = reduced.coverage(root)
                self.assertEqual(coverage['core_fit_target'], 1281)
                self.assertFalse(coverage['completion_claim'])
                self.assertEqual(coverage['unresolved_fold_cells'], 192)
                frozen_policy=root/'REDUCED_PROTOCOL.md';frozen_policy.write_bytes(reduced.POLICY.read_bytes())
                self.assertEqual(reduced.coverage(root)['registration_sha256'],sha(root/'REDUCED_SCOPE.json'))
                frozen_policy.write_text('changed')
                with self.assertRaisesRegex(ValueError,'evidence drift'):reduced.coverage(root)
                frozen_policy.write_bytes(reduced.POLICY.read_bytes())
                cell = next(c for c in matrix() if c['stage'] == 'rescue'
                            and c['family'] == 'relu' and c['rescue'] == 'init_half')
                reduced.admit_rescue(root, cell, host='r5600g')
                with self.assertRaisesRegex(ValueError, 'replacement compute pool'):
                    reduced.admit_rescue(root, cell, host='dev')
                with self.assertRaisesRegex(ValueError, 'outside owner-approved reduced scope'):
                    reduced.admit_rescue(root, {**cell, 'rescue': 'init_double'}, host='r5600g')
                for stage in ('row_curve', 'full_fit_selection', 'full_fit'):
                    removed = next(c for c in matrix() if c['stage'] == stage)
                    with self.assertRaisesRegex(ValueError, 'outside owner-approved reduced scope'):
                        reduced.admit_cell(root, removed, host='r5600g')
                admitted = next(c for c in matrix() if c['stage'] == 'human_only')
                reduced.admit_cell(root, admitted, host='r5600g')
                with self.assertRaises(FileExistsError):
                    reduced.register(root, approval)
                record['interrupted_replays_authorized'] = True
                approval.write_bytes(encoded(record))
                with self.assertRaisesRegex(ValueError, 'coverage evidence drift'):
                    reduced.coverage(root)
                with self.assertRaisesRegex(ValueError, 'approval scope'):
                    reduced.register(root, approval)
                proposal.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'checksum differs'):
                    reduced.register(root, approval)
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
