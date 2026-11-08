"""Hash-bound reduced coverage over the immutable original shaping registration."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from shaping_completion import REPO, encoded, sha
from shaping_execution import load_registration
from shaping_outer_plan import finalist_roles

PROPOSAL_SHA256 = 'fce3b7ff3f7ad32ae49cee5d1a506b67fea885a1e5b4ac2fc28b756a97368dde'
POLICY = REPO / 'docs/SHAPING_REDUCED_SCOPE.md'
TARGETS = dict(inner=816, fresh_control_inner=72, fresh_control_outer=9,
               rescue=144, outer=48, source_curve=96, row_curve=0, shuffle=48,
               human_only=48, full_fit_selection=0, full_fit=0)
ROLES = ('relu', 'extra_trees', 'shaped_additive', 'best_analytic',
         'best_product', 'best_separate')
RESCUE_ROLES = ('relu', 'best_analytic', 'best_product', 'best_separate')
RESCUES = ('init_half', 'constant_lr')


def fixed_cells(cells):
    """No fitted scores are needed for these 1,089 original cells."""
    selected = [c for c in cells if
                c['stage'] in ('inner', 'fresh_control_inner', 'fresh_control_outer', 'human_only')
                or (c['stage'] == 'source_curve' and c['fraction'] in (.25, .5)
                    and c['order_seed'] == 20261107)
                or (c['stage'] == 'shuffle' and c['permutation'] == 'population'
                    and c['permutation_seed'] == 20261107)]
    expected = {k: TARGETS[k] for k in ('inner', 'fresh_control_inner',
                                      'fresh_control_outer', 'human_only', 'source_curve', 'shuffle')}
    if dict(Counter(c['stage'] for c in selected)) != expected:
        raise ValueError('fixed reduced matrix coverage differs')
    if len({c['id'] for c in selected}) != len(selected):
        raise ValueError('duplicate original cell')
    return selected


def fold_cells(cells, decisions, outer):
    """Resolve original IDs only after all twelve base decisions are available.

    The caller verifies the decision evidence; this pure function checks scope,
    exposure and exact paired coverage without reading any model responses.
    """
    if outer not in range(3):
        raise ValueError('registered outer fold required')
    for family, decision in decisions.items():
        if (decision['family'] != family or decision['outer'] != outer
                or decision['outer_results_read'] or decision['optimizer_choice'] != 'base'):
            raise ValueError('unexposed base decisions required before rescues')
    roles = finalist_roles(decisions)
    if tuple(roles) != ROLES:
        raise ValueError('finalist roles differ')
    neural = {roles[role] for role in RESCUE_ROLES}
    finalists = set(roles.values())
    if len(neural) != 4 or len(finalists) != 6:
        raise ValueError('distinct role families required')
    selected = [c for c in cells if
                (c['stage'] == 'rescue' and c['partition'] in
                 (f'inner-{outer}-0', f'inner-{outer}-1') and c['family'] in neural
                 and c['rescue'] in RESCUES)
                or (c['stage'] == 'outer' and c['partition'] == f'outer-{outer}'
                    and c['family'] in finalists)]
    expected_rescues = {(family, f'inner-{outer}-{fold}', seed, choice)
                        for family in neural for fold in range(2)
                        for seed in range(3) for choice in RESCUES}
    actual_rescues = {(c['family'], c['partition'], c['seed'], c['rescue'])
                      for c in selected if c['stage'] == 'rescue'}
    expected_outer = {(family, None if family == 'shaped_additive' else seed)
                      for family in finalists
                      for seed in ([None] if family == 'shaped_additive' else range(3))}
    actual_outer = {(c['family'], c['seed']) for c in selected if c['stage'] == 'outer'}
    if (actual_rescues != expected_rescues or actual_outer != expected_outer
            or len(selected) != 64 or len({c['id'] for c in selected}) != 64):
        raise ValueError('complete reduced paired fold coverage required')
    return dict(outer=outer, roles=roles, cells=selected, outer_results_read=False,
                shortlist_stage='original_inner_base_before_rescues')


def register(root, approval_path):
    approval = json.loads(approval_path.read_text())
    proposal_path = Path(approval['proposal'])
    if sha(proposal_path) != PROPOSAL_SHA256 or approval['proposal_sha256'] != PROPOSAL_SHA256:
        raise ValueError('approved proposal checksum differs')
    proposal = json.loads(proposal_path.read_text())
    if (approval['answer'] != 'Approve reduced plan (recommended)'
            or approval['core_fit_target'] != 1281
            or approval['interrupted_replays_authorized'] is not False
            or approval['active_fits_must_finish'] is not True
            or approval['old_registration_preserved'] is not True):
        raise ValueError('owner approval scope differs')
    if (proposal['proposed_by_stage'] != TARGETS or proposal['proposed_core_cells'] != 1281
            or proposal['rescue']['roles'] != list(RESCUE_ROLES)
            or proposal['rescue']['choices'] != list(RESCUES)
            or proposal['outer_roles'] != list(ROLES)):
        raise ValueError('approved targets or roles differ')
    _, cells, _ = load_registration(root)
    if proposal['original_plan_sha256'] != sha(root / 'PLAN.json'):
        raise ValueError('original registration differs')
    fixed = fixed_cells(cells)
    record = dict(schema='shaping-reduced-scope-v1', core_fit_target=1281,
                  required_by_stage=TARGETS, fixed_cells=sorted(c['id'] for c in fixed),
                  unresolved_fold_cells=192, completed_and_active_attempts_preserved=True,
                  interrupted_fit_replays_authorized=False, outer_results_read=False,
                  role_selection='complete_original_inner_base_before_rescues',
                  evidence_sha256={str(approval_path): sha(approval_path),
                                   str(proposal_path): sha(proposal_path), str(POLICY): sha(POLICY),
                                   **{str(root / name): sha(root / name)
                                      for name in ('PLAN.json', 'MATRIX.json', 'PARTITIONS.json')}},
                  driver_sha256=sha(__file__))
    path = root / 'REDUCED_SCOPE.json'
    with path.open('xb') as stream:
        stream.write(encoded(record))
    return record


def coverage(root):
    """Expose approved targets without counting unresolved roles as completed."""
    path = root / 'REDUCED_SCOPE.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record['schema'] != 'shaping-reduced-scope-v1'
            or record['required_by_stage'] != TARGETS or record['core_fit_target'] != 1281
            or len(set(record['fixed_cells'])) != 1089
            or record['unresolved_fold_cells'] != 192
            or record['outer_results_read'] is not False
            or record['interrupted_fit_replays_authorized'] is not False):
        raise ValueError('reduced coverage registration differs')
    for name, pin in record['evidence_sha256'].items():
        if sha(name) != pin:
            raise ValueError('reduced coverage evidence drift')
    return dict(registration_sha256=sha(path), core_fit_target=1281,
                required_by_stage=TARGETS, fixed_cell_count=1089,
                unresolved_fold_cells=192, actual_counts_include_preserved_original_cells=True,
                completion_claim=False)


def shortlist(root, outer, *, register=False):
    """Reproduce the original inner-only screen before allocating rescue trials."""
    if coverage(root) is None:
        raise ValueError('approved reduced registration required')
    from shaping_completion import FAMILIES
    from shaping_outer_admission import base_decision
    _, cells, _ = load_registration(root)
    decisions = {}
    evidence = {str(root / 'REDUCED_SCOPE.json'): sha(root / 'REDUCED_SCOPE.json')}
    for family in FAMILIES:
        actual, _, pins = base_decision(root, family, outer, cells)
        setting = actual['selected_setting']
        decisions[family] = dict(family=family, outer=outer, setting=setting,
                                 optimizer_choice='base', outer_results_read=False,
                                 inner_score=actual['setting_scores'][setting])
        evidence.update(pins)
    result = fold_cells(cells, decisions, outer)
    result.update(evidence_sha256=evidence,
                  reduced_scope_sha256=sha(root / 'REDUCED_SCOPE.json'))
    path = root / 'reduced_scope' / f'outer-{outer}' / 'SHORTLIST.json'
    if register:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(encoded(result))
    else:
        if json.loads(path.read_text()) != result:
            raise ValueError('reduced shortlist cannot be reproduced')
    return result, path


def rescue_choices(root, family, outer):
    """Keep legacy scope unless a verified owner amendment is present."""
    if coverage(root) is None:
        from shaping_completion import RESCUES as original
        return original, {}
    pins = {str(root / 'REDUCED_SCOPE.json'): sha(root / 'REDUCED_SCOPE.json')}
    # ReLU's role is fixed in advance and does not depend on the candidate pools.
    if family != 'relu':
        record, path = shortlist(root, outer)
        if family not in {record['roles'][role] for role in RESCUE_ROLES}:
            raise ValueError('family is outside the frozen rescue shortlist')
        pins[str(path)] = sha(path)
    return RESCUES, pins


def neural_families(root, outer):
    from shaping_completion import NEURAL
    if coverage(root) is None:
        return NEURAL
    record, _ = shortlist(root, outer)
    selected = {record['roles'][role] for role in RESCUE_ROLES}
    return tuple(family for family in NEURAL if family in selected)


def _admit_host(root, host):
    scope = coverage(root)
    if scope is None:
        return False
    import socket
    record = json.loads((root / 'REDUCED_SCOPE.json').read_text())
    approvals = []
    for name in record['evidence_sha256']:
        if Path(name).suffix == '.json':
            row = json.loads(Path(name).read_text())
            if isinstance(row, dict) and row.get('answer') == 'Approve reduced plan (recommended)':
                approvals.append(row)
    if len(approvals) != 1:
        raise ValueError('one pinned owner approval required')
    if (host or socket.gethostname()) not in approvals[0]['allowed_new_compute_hosts']:
        raise ValueError('host is outside owner-approved replacement compute pool')
    return True


def admit_cell(root, cell, *, host=None):
    """Guard new attempts; historical verification retains original coverage."""
    if not _admit_host(root, host):
        return
    stage = cell['stage']
    if stage in ('row_curve', 'full_fit_selection', 'full_fit'):
        raise ValueError('new fit is outside owner-approved reduced scope')
    if stage == 'rescue':
        outer = int(cell['partition'].split('-')[1])
        choices, _ = rescue_choices(root, cell['family'], outer)
        if cell['rescue'] not in choices:
            raise ValueError('new rescue is outside owner-approved reduced scope')
        return
    if stage in ('outer', 'exposure_extension', 'fresh_control_outer_matched'):
        outer = int(cell['partition'].split('-')[1])
        record, _ = shortlist(root, outer)
        if stage == 'outer':
            if cell not in record['cells']:
                raise ValueError('outer fit is outside frozen reduced shortlist')
        elif cell['family'] not in set(record['roles'].values()):
            raise ValueError('conditional fit is outside frozen reduced shortlist')
        # The existing owner must also resolve the exact conditional registration.
        return
    record = json.loads((root / 'REDUCED_SCOPE.json').read_text())
    if cell['id'] not in record['fixed_cells']:
        raise ValueError('new fit is outside owner-approved reduced scope')


def admit_rescue(root, cell, *, host=None):
    if cell['stage'] != 'rescue':
        raise ValueError('registered rescue cell required')
    admit_cell(root, cell, host=host)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    args = parser.parse_args()
    result = register(args.study, args.approval)
    print(json.dumps({k: v for k, v in result.items() if k != 'fixed_cells'}), flush=True)
