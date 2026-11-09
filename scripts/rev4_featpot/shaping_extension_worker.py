"""Resolve an existing registered exposure extension for native worker transport."""
import json
from pathlib import Path
from shaping_completion import sha


def admit(args, *, host=None):
    from shaping_execution import load_registration
    from shaping_extension import resolve
    from shaping_reduced_scope import admit_cell
    from shaping_corrected_view import require_downstream_build
    from shaping_rescue import admitted_families
    from shaping_rescue_execution import primary_source
    plan, cells, _ = load_registration(args.study)
    resolution = resolve(args.study, args.cell)
    cell = resolution['cell']
    admit_cell(args.study, cell, host=host)
    build = json.loads((args.binary_dir / 'BUILD.json').read_text())
    for name, pin in build['binary_sha256'].items():
        if sha(args.binary_dir / name) != pin:
            raise ValueError('extension native binary drift')
    require_downstream_build(args.study, cell['family'], sha(args.binary_dir / 'BUILD.json'))
    if cell['family'] in ('softplus', 'silu_monotone', 'silu_diagnostic'):
        q = json.loads((args.neutral_qualification / 'RESULT.json').read_text())
        if not q['match'] or q['epochs'] != 240 or q['build_sha256'] != sha(args.binary_dir / 'BUILD.json'):
            raise ValueError('full analytic neutral replay required')
    record = json.loads((args.study / 'comparators' / cell['partition'] / 'GROUPS.json').read_text())
    admitted_families(args.study, record)
    for path, pin in record['file_sha256'].items():
        if sha(path) != pin:
            raise ValueError('extension TRAIN input drift')
    templates = [primary_source(args.study, cells, cell, cell['setting'])]
    if '_pretrained_' in cell['family']:
        parent = cell['family'].split('_pretrained_')[0] + '_joint_identity'
        templates.append(primary_source(args.study, cells, cell, cell['setting'], parent))
    schedule = dict(setting=cell['setting'], optimizer_choice=cell['optimizer_choice'],
                    phase_epochs=cell['phase_epochs'], total_pair_draws=cell['total_pair_draws'])
    return dict(cell=cell, resolution=resolution, templates=templates, schedule=schedule)
