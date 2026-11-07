"""Count final-phase neural parameters from saved native structure and manifests."""
import argparse, json, subprocess
from pathlib import Path
from shaping_completion import encoded, sha


def counts(inputs, first, pairs, head=None, shaping=None):
    if any(type(v) is not int for v in (inputs, first, pairs)) or inputs <= 0 or first <= 0 or not 0 <= 2*pairs <= first:
        raise ValueError('invalid native dimensions or pair count')
    diagnostic = head is not None and head['family'] == 'silu_diagnostic'
    second = max(first//2, 8)
    temperatures = 0
    if head is not None:
        if (head['schema'] != 'monotone-head-v1' or head['family'] not in ('softplus', 'silu_monotone', 'silu_diagnostic')
                or head['first_width'] != first or head['max_pairs'] != pairs
                or len(head['widths']) != 1 or bool(head.get('unrestricted_first', False)) != diagnostic):
            raise ValueError('unsupported or inconsistent analytic owner')
        second = head['widths'][0]
        if type(second) is not int or second <= 0 or (diagnostic and pairs):
            raise ValueError('invalid analytic width or diagnostic ties')
        if type(head['analytic']['learn_temperature']) is not bool:
            raise ValueError('explicit temperature policy required')
        temperatures = int(head['analytic']['learn_temperature'])
    result = dict(first_weights=inputs*first, first_biases=first-pairs,
                  intermediate_weights=(first-pairs)*second, intermediate_biases=second,
                  output_weights=second, output_biases=int(diagnostic),
                  learned_temperatures=temperatures, damage_slopes=0, reference_gate_values=0)
    frozen_shape_values = 0
    if shaping is not None:
        terms = shaping['terms']
        if shaping['schema'] != 'zensim-input-shaping-v1' or type(shaping['trainable']) is not bool:
            raise ValueError('explicit shaping training policy required')
        columns = [t['column'] for t in terms]
        # Shape columns are logical feature IDs, not compact exported positions.
        # The native loader binds them to the kept-column map before training.
        if len(columns) > inputs or len(set(columns)) != len(columns) or any(type(c) is not int or c < 0 for c in columns):
            raise ValueError('invalid shaped columns')
        for term in terms:
            if (len(term['knots']) != 2 or len(term['slopes']) != 3 or len(term['gate']) != 3
                    or type(term['separate']) is not bool or (term['separate'] and term['reference'] is None)):
                raise ValueError('unsupported shape term')
        slopes = 3*len(terms)
        gates = 3*sum(t['separate'] for t in terms)
        if shaping['trainable']:
            result.update(damage_slopes=slopes, reference_gate_values=gates)
        else:
            frozen_shape_values = slopes+gates
    return dict(trainable_scalar_entries=sum(result.values()), components=result,
                frozen_response_shape_values=frozen_shape_values,
                interpretation='Nominal adjustable scalar entries after explicit ties and fixed constants; not effective rank, active capacity, optimizer storage, or identifiable functional degrees of freedom. Fixed knots/scalers/routing and unused product-gate slots excluded.')


def count_run(run, inspector, output):
    if output.exists():
        raise ValueError('preserve existing parameter report')
    pins = {str(p): sha(p) for p in (run/'RUN.json', run/'RESULT.json', inspector)}
    identity = json.loads((run/'RUN.json').read_text())
    result = json.loads((run/'RESULT.json').read_text())
    from shaping_execution import load_registration
    study = run.parents[2]
    _, cells, _ = load_registration(study)
    if identity['cell'] not in cells or identity['plan_sha256'] != sha(study/'PLAN.json'):
        raise ValueError('registered unchanged study cell required')
    pins[str(study/'PLAN.json')] = sha(study/'PLAN.json')
    if result['stage'] != 'trained' or result['exit_code'] != 0 or result['cell'] != identity['cell']:
        raise ValueError('completed matching neural fit required')
    fit = run/identity.get('fit_directory', '.')
    family = identity.get('resolved_family', identity['cell']['family'])
    if family not in ('relu', 'softplus', 'silu_monotone', 'silu_diagnostic',
                      'product_joint_identity', 'product_pretrained_joint', 'product_pretrained_frozen',
                      'separate_joint_identity', 'separate_pretrained_joint', 'separate_pretrained_frozen'):
        raise ValueError('registered neural family required')
    expected_head = family in ('softplus', 'silu_monotone', 'silu_diagnostic')
    expected_shape = family.startswith(('product_', 'separate_'))
    if (fit/'HEAD.json').exists() != expected_head or (fit/'SHAPING.json').exists() != expected_shape:
        raise ValueError('required parameter manifests missing or unexpected')
    command = result['phases'][-1]['command'] if result.get('phases') else identity['command']
    def option(key, default=None):
        positions = [i for i, v in enumerate(command) if v == key]
        if not positions and default is not None:
            return default
        if len(positions) != 1:
            raise ValueError('one command setting required: '+key)
        return int(command[positions[0]+1])
    if option('--n-hidden-layers') != 2 or any(flag not in command for flag in ('--nonneg-distance', '--monotone-cbc', '--nonneg-thresholds')):
        raise ValueError('registered shared-depth owner required')
    model = fit/'checkpoints'/f"ckpt_epoch{identity['epochs']-1:03}.bin"
    pins[str(model)] = sha(model)
    if pins[str(model)] != result['bake_sha256']:
        raise ValueError('final checkpoint differs from training result')
    structure = json.loads(subprocess.check_output([str(inspector), 'inspect', '--in', str(model)], text=True))
    first = option('--hidden'); pairs = option('--max-pairs', 0)
    if structure['layers'][0]['out_dim'] != first or structure['layers'][0]['in_dim'] != structure['inputs']:
        raise ValueError('native first layer differs from training command')
    configs = {}
    for file, key in [('HEAD.json', 'zentrain.monotone_research'), ('SHAPING.json', 'zentrain.input_shaping')]:
        path = fit/file
        if path.exists():
            pins[str(path)] = sha(path)
            initial = json.loads(path.read_text())
            final = json.loads(subprocess.check_output([str(inspector), 'read-meta', '--in', str(model), '--key', key], text=True))
            if file == 'HEAD.json':
                if initial != final:
                    raise ValueError('head manifest differs from native export')
            else:
                from shaping_pretraining import structural_terms
                if initial['trainable'] != final['trainable'] or structural_terms(initial) != structural_terms(final):
                    raise ValueError('shape structure or training policy changed')
            configs[file] = final
    if expected_head and configs['HEAD.json']['family'] != family:
        raise ValueError('analytic family differs from registered run')
    counted = counts(structure['inputs'], first, pairs, configs.get('HEAD.json'), configs.get('SHAPING.json'))
    for path, pin in pins.items():
        if sha(path) != pin:
            raise ValueError('parameter evidence changed')
    counted.update(cell=identity['cell'], scope='final training phase only; original final checkpoint before export repair',
                   structure=structure, native_configs=configs, source_pins=pins, driver_sha256=sha(__file__),
                   scientific_eligibility_asserted=False, training_replayed=False)
    output.write_bytes(encoded(counted))
    return counted


def qualify(study, inspector, output):
    from shaping_execution import load_registration
    if output.exists():
        raise ValueError('preserve existing qualification')
    _, cells, _ = load_registration(study)
    families = ('relu', 'softplus', 'silu_monotone', 'silu_diagnostic',
                'product_joint_identity', 'separate_joint_identity',
                'product_pretrained_frozen', 'product_pretrained_joint',
                'separate_pretrained_frozen', 'separate_pretrained_joint')
    output.mkdir(parents=True)
    rows = []; missing = []
    for family in families:
        matches = [c for c in cells if c['stage'] == 'inner' and c['partition'] == 'inner-0-0'
                   and c['family'] == family and c['seed'] == 0 and c['setting'] == 0]
        if len(matches) != 1:
            raise ValueError('one fixed qualification cell required')
        cell = matches[0]
        attempts = [p.parent for p in (study/'runs'/cell['id']).glob('attempt-*/RESULT.json')
                    if json.loads(p.read_text()).get('stage') == 'trained'
                    and json.loads(p.read_text()).get('exit_code') == 0]
        if not attempts:
            missing.append(cell); continue
        if len(attempts) != 1:
            raise ValueError('ambiguous completed qualification attempt')
        record = count_run(attempts[0], inspector, output/(family+'.json'))
        rows.append(dict(family=family, cell=cell, run=str(attempts[0]),
                         count=record['trainable_scalar_entries'],
                         components=record['components'], frozen_shape_values=record['frozen_response_shape_values']))
    result = dict(rows=rows, missing=missing, complete=not missing, training_replayed=False)
    (output/'SUMMARY.json').write_bytes(encoded(result))
    if missing:
        raise ValueError('fixed qualification fits not all available; partial evidence preserved')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', type=Path)
    mode.add_argument('--qualify-study', type=Path)
    for name in ('inspector', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(qualify(a.qualify_study, a.inspector, a.output) if a.qualify_study
                     else count_run(a.run, a.inspector, a.output)))
