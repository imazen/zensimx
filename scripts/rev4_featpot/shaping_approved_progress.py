"""Count approved work separately from preserved original evidence for the viewer."""
from collections import Counter
import json
from pathlib import Path

from shaping_completion import sha


def approved_progress(study, rows):
    path = study / 'REDUCED_SCOPE.json'
    if not path.exists():
        return None
    scope = json.loads(path.read_text())
    selected = set(scope['fixed_cells'])
    registered = {row['cell']: row for row in rows}
    # Activated primary rows already replace the superseded threshold conditions.
    selected.update(row['cell'] for row in rows if row['stage'] == 'inner')
    folds = []
    for outer in range(3):
        shortlist = study / 'reduced_scope' / f'outer-{outer}' / 'SHORTLIST.json'
        if not shortlist.exists():
            continue
        record = json.loads(shortlist.read_text())
        if (record['outer'] != outer or record['outer_results_read'] is not False
                or record['reduced_scope_sha256'] != sha(path)
                or Counter(c['stage'] for c in record['cells']) != dict(rescue=48, outer=16)
                or len({c['id'] for c in record['cells']}) != 64):
            raise ValueError('frozen reduced viewer shortlist differs')
        for cell in record['cells']:
            row = registered.get(cell['id'])
            if row is None or row['stage'] != cell['stage'] or row['family'] != cell['family']:
                raise ValueError('viewer shortlist contains an unregistered cell')
        selected.update(c['id'] for c in record['cells'])
        folds.append(outer)
    counts = {}
    for stage, target in scope['required_by_stage'].items():
        chosen = [row for row in rows if row['cell'] in selected and row['stage'] == stage]
        counts[stage] = dict(required=target, assigned_cells=len(chosen),
            successful_fit_cells=sum(any(a['fit_success'] for a in row['attempts']) for row in chosen),
            archive_receipt_cells=sum(row['archive_receipt_verified'] for row in chosen))
    return dict(counts=counts, frozen_shortlist_folds=folds, unresolved_fold_cells=64 * (3 - len(folds)),
                interpretation='Approved cells only; historical fits outside this subset remain in preserved counts. Conditional extensions are separate.')
