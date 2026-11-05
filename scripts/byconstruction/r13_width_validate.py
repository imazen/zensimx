"""Check R13's actual frozen artifacts and read coverage without changing any gate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import r13_width as owner


def load(path):
    return json.loads(Path(path).read_text())


def main():
    root = owner.O
    assert owner.sha(root / 'FROZEN_FITS.json') == '7cbcb2e7e5726e4b31738da1de6ac480a7c58786cd3c6b12970969a4ea12e9b1'
    fits = load(root / 'FROZEN_FITS.json')['models']
    expected = {f"{v['arm']}_s{v['seed']}": v['packed_sha256'] for v in fits}
    assert set(expected) == {f'W{w}_s0' for w in (4,16,32,64,256,512)}
    models = load(root / 'MODELS.json')
    assert set(models) == set(expected)
    pins = load(root / 'reads/MODEL_PINS.json')
    assert set(pins) == set(expected) | {'F_s0','F_s1','F_s2'}
    for name, row in pins.items():
        assert owner.sha(row['path']) == row['sha256']
        if name in expected:
            assert row['sha256'] == expected[name] == owner.sha(models[name])
    for fit in fits:
        path = root / 'fits' / f"{fit['arm']}-s0"
        for stage, epoch in (('base',119),('continuation',59)):
            assert load(path / stage / 'result.json')['selection']['selected_epoch'] == epoch
    assert not list((root / 'logs').glob('*-s[12]-*.log'))
    source = load(root / 'READ_SOURCE_PINS.json')
    effective = dict(source['repairs'][-1]['source_sha256'])
    effective['scripts/byconstruction/r13_width.py'] = source['repairs'][-1]['owner_sha256']
    for path, digest in effective.items():
        snapshot = root / 'read-source' / path
        if not snapshot.exists():
            assert owner.sha(owner.X / path) == digest, path
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(owner.X / path,snapshot)
        assert owner.sha(snapshot) == digest, path
    tools = dict(source['native_tool_sha256'])
    tools.update(load(root / 'GATE_TOOL_PINS.json')['tools'])
    tools.update({str(owner.HEAD):'aa409c8e689323f7ed0f8ca4f07fa34565d673efa18e26806f98572677af3a33',
                  str(owner.EXPORT):'d9e1ca3e9624036f567c29dbc5cd79668a697fe4a63b4bb33c9cf30fdaed2f6f'})
    archive = {}
    for path, digest in tools.items():
        assert owner.sha(path) == digest, path
        dest = root / 'read-tools' / f'{digest[:12]}-{Path(path).name}'
        if not dest.exists():
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(path,dest)
        assert owner.sha(dest) == digest
        archive[path] = dict(snapshot=str(dest.relative_to(root)),sha256=digest)
    owner.write(root / 'READ_TOOL_ARCHIVE.json',dict(pinned_files=archive,
        role='execution tools plus the complete historical gate manifest bundle; byc_score_timing is preserved provenance and is not the R13 head instrument'))
    assert owner.sha(root / 'bin/zensim_mlp_train') == '267a3492e14bd2c2988e49a056b9a1840b188aada796804974f635b733c670cf'
    native = load(root / 'W4_SOURCE_PINS.json')
    assert owner.sha(native['native_binary']) == native['native_sha256']
    cap = load(root / 'CAPACITY.json')
    assert set(cap) == {f'{bank}/{name}' for bank in owner.BANKS for name in expected}
    for bank, (path, digest) in owner.BANKS.items():
        assert owner.sha(path) == digest
        for name in expected:
            row = cap[f'{bank}/{name}']
            assert row['bake_sha256'] == expected[name] and row['bank_sha256'] == digest
            assert row['dead'] + row['residual'] + row['live'] == row['hidden']
            assert row['rows'] == {'cid22':12163,'hdr':7390}[bank]
    timing = load(root / 'heads/SUMMARY.json')
    assert set(timing) == set(expected)
    for name in expected:
        raw = load(root / f'heads/timing-{name}.json')
        comparison, = raw['comparisons']
        assert not raw['unreliable'] and len(comparison['samples']) == 30
        assert timing[name]['clean'] == sum(s['gate_clean'] for s in comparison['samples']) > 0
    names = {'eval-score','readout','ceval-compression','ceval-noncodec','e36-score',
             'e36-inversions','primary','chroma-score','chroma-level','chroma-report','hdr-panel','hdr-upiq'}
    names |= {f'{stage}-{name}' for name in expected for stage in ('steer','dial','gate-summary','nearid')}
    receipts = list((root / 'reads').glob('*.receipt.json'))
    assert len(receipts) == 36 and {p.name.removesuffix('.receipt.json') for p in receipts} == names
    assert all(load(p)['rc'] == 0 for p in receipts)
    assert load(root / 'reads/PRIMARY.json')['meta'] == dict(sources=40,pairs=646677,draws=10000,seed=20261101,unit='selection source (origin)')
    anchor = load(owner.H / 'tmp/e36test/gates/mixF3_s0/steer135/mixF3_s0/job.json')
    def cases(packet):
        return [{k:v for k,v in c.items() if k not in ('model','model_sha256')} for c in packet['cases']]
    results = {}
    for name in expected:
        gate = root / 'reads/gates' / name
        packet = load(gate / f'steer135/{name}/job.json')
        assert len(packet['cases']) == 135 and cases(packet) == cases(anchor)
        assert all(c['model_sha256'] == expected[name] for c in packet['cases'])
        steer = load(gate / f'steer135/{name}/result.json')
        assert steer['formula_revision'] == 7 and len(steer['rows']) == 135
        assert all(r['model'] == expected[name] and r['sensitivity_method'] == 'exact-gradient-d12' for r in steer['rows'])
        identity = [json.loads(line) for line in (gate / 'identity-s0.jsonl').read_text().splitlines()]
        identity = [r for r in identity if not r.get('summary')]
        assert len(identity) == 744
        contracts = {}
        for grid in ('standard','ladder'):
            paths = list(gate.glob(f'verdict-s0-*/{grid}/fulleval.json'))
            assert len(paths) == 1
            verdict = load(paths[0])
            assert verdict['bake_sha256'] == expected[name]
            checks = [c for c in verdict['dial']['addressability']['checks'] if c['id'] in ('C1','C2','C3','C4','C5','C6')]
            assert len(checks) == 6 and len({c['id'] for c in checks}) == 6
            contracts[grid] = {c['id']:c['state'] for c in checks}
        near = load(root / f'reads/nearid/{name}/candidate-{name.replace("_", "").lower()}.GATES.json')
        assert near['rows'] == 648 and near['N3']['total_ladders'] == 144 and len(near['references']) == 24
        results[name] = dict(steer_successes=sum(r['pass'] for r in steer['rows']),contracts=contracts,
            identity_rows=744,identity_pass=all(r['candidate_score_bits'] == [4636737291354636288] and all(r['candidate_read_set_bits_equal_first_permutation']) for r in identity),
            nearid={k:near[k]['pass'] for k in ('N1','N2','N3')},nearid_monotone=near['N3']['monotone_ladders'])
    result = dict(checked=datetime.now(timezone.utc).isoformat(),fit_freeze_sha256=owner.sha(root / 'FROZEN_FITS.json'),
                  models=6,capacity_cells=12,head_pairs=6,head_rounds_each=30,read_receipts=36,
                  read_source_commit=source['repairs'][-1]['source_commit'],
                  gate_results=results,role='artifact and population validation; negative gate outcomes remain results')
    owner.write(root / 'FINAL_VALIDATION.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
