"""Qualify native structural inventory against existing inspection/parity evidence."""
import argparse, copy, json, re, subprocess
from pathlib import Path
from shaping_completion import encoded, sha


def check(binary, inspector, workload, tree_log, output):
    if output.exists():
        raise ValueError('preserve previous inventory qualification')
    pins = {str(p): sha(p) for p in (binary, inspector, workload, tree_log)}
    original = json.loads(workload.read_text())
    for arm in original['arms']:
        model = (workload.parent/arm['model']).resolve()
        if sha(model) != arm['sha256']:
            raise ValueError('fixture model hash differs')
        pins[str(model)] = arm['sha256']; arm['model'] = str(model)
    output.mkdir(parents=True)
    source = output/'WORKLOAD.json'; source.write_bytes(encoded(original))
    result = output/'INVENTORY.json'
    with (output/'native.log').open('w') as log:
        subprocess.run([str(binary), str(source), str(result)], check=True, stdout=log, stderr=subprocess.STDOUT)
    inventory = json.loads(result.read_text())
    if inventory['workload_sha256'] != sha(source) or inventory['binary_sha256'] != pins[str(binary)]:
        raise ValueError('native inventory identity differs')
    observed = {row['id']: row for row in inventory['models']}
    if set(observed) != {a['id'] for a in original['arms']}:
        raise ValueError('inventory model coverage differs')
    for i, arm in enumerate(original['arms']):
        row = observed[arm['id']]; model = Path(arm['model'])
        if row['model_sha256'] != arm['sha256'] or row['model_bytes'] != model.stat().st_size:
            raise ValueError('inventory model identity differs')
        if arm['kind'] == 'tree':
            match = re.findall(r'caller width (\d+), reads (\d+), (\d+) trees / (\d+) nodes', tree_log.read_text())
            if len(match) != 1 or 'corrhead_parity: PASS' not in tree_log.read_text():
                raise ValueError('one passed original tree parity log required')
            width, fields, trees, nodes = map(int, match[0])
            s = row['structure']
            if (s['caller_input_width'], len(s['declared_feature_ids']), s['trees'], s['nodes']) != (width, fields, trees, nodes):
                raise ValueError('tree inventory differs from original native parity')
        else:
            data = subprocess.check_output([str(inspector), 'inspect', '--in', str(model)], text=True)
            (output/f'inspect-{i:02}.json').write_text(data)
            reference = json.loads(data); structure = row['structure']
            if structure['expanded_stored_coefficients'] != reference['stored_coefficients'] or structure['inputs'] != reference['inputs']:
                raise ValueError('stored coefficient or input count differs')
            if len(structure['layers']) != len(reference['layers']):
                raise ValueError('layer count differs')
            for actual, expected in zip(structure['layers'], reference['layers']):
                for key in ('in_dim', 'out_dim', 'activation', 'stored_weights', 'stored_biases'):
                    if actual[key] != expected[key]:
                        raise ValueError('native layer structure differs')
                if actual['nonzero_weights'] != expected['stored_weights']-expected['zero_weights']:
                    raise ValueError('active stored weight count differs')
    wrong = copy.deepcopy(original); wrong['arms'][0]['sha256'] = '0'*64
    wrong_path = output/'wrong-hash.json'; wrong_path.write_bytes(encoded(wrong))
    wrong_result = output/'wrong-result.json'
    failure = subprocess.run([str(binary), str(wrong_path), str(wrong_result)], capture_output=True, text=True)
    (output/'wrong-hash.log').write_text(failure.stdout+failure.stderr)
    if not failure.returncode or 'model hash differs' not in failure.stderr or wrong_result.exists():
        raise ValueError('model corruption not refused')
    for name, pin in pins.items():
        if sha(name) != pin:
            raise ValueError('qualification evidence changed')
    receipt = dict(models=len(observed), native_inventory_verified=True,
                   training_parameter_accounting_complete=False, inventory_sha256=sha(result), source_pins=pins)
    (output/'QUALIFICATION.json').write_bytes(encoded(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'inspector', 'workload', 'tree-log', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(check(a.binary, a.inspector, a.workload, a.tree_log, a.output)))
