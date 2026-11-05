"""Describe R14 beside the R13 seed-zero cells already read by their owner.

Only completed artifacts are read. An absent width-lane gate/panel is reported
as not measured here; it never becomes a pass or an invitation to rerun it.
"""
import hashlib
import json
from pathlib import Path
from statistics import mean


O = Path('/var/tmp/byc-transforms-20261104')
W = Path('/var/tmp/byc-width-20261104')
X = Path(__file__).resolve().parents[2]


def main():
    sources = {}

    def load(path):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    ours, widths = load(O/'reads/readout.json'), load(W/'reads/readout.json')
    primary = {root: load(root/'reads/PRIMARY.json') for root in (O, W)}
    pins = {root: load(root/'reads/MODEL_PINS.json') for root in (O, W)}
    capacity = {root: load(root/'CAPACITY.json') for root in (O, W)}
    for seed in range(3):
        if ours['per'][f'F@{seed}'] != widths['per'][f'F@{seed}']:
            raise ValueError('R13/R14 complete A control dictionaries differ')
    if any('aic' in c.lower() and '2026' in c for d in (ours, widths)
           for cols in d['per'].values() for c in cols):
        raise ValueError('AIC-2026 quality cannot enter the comparison')
    inventory = [('A0', 'F', O), *[(arm, arm, W) for arm in ('W4','W16','W32','W64','W256','W512')],
                 *[(arm, arm, O) for arm in ('T1a','T1b','T1c')]]
    rows = []
    for label, arm, root in inventory:
        name = f'{arm}_s0'
        pin = pins[root][name]
        if hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError(f'{name}: packed model differs')
        per = (ours if root == O else widths)['per'][f'{arm}@0']
        row = dict(label=label, packed_sha256=pin['sha256'],
            E36_percent=100*primary[root][arm]['seeds'][name],
            EVAL9_percent=100*mean(v['med'] for v in per.values()),
            EVAL8_percent=100*mean(v['med'] for k,v in per.items() if k!='mdid'),
            per_collection_median_percent={k:100*v['med'] for k,v in per.items()})
        row['capacity'] = {}
        for bank in ('cid22','hdr'):
            key = f'{bank}/{name}'
            if key in capacity[root]:
                values = capacity[root][key]
                reference = capacity[O][f'{bank}/T1a_s0']
                if (values['bake_sha256'] != pin['sha256'] or values['rows'] != reference['rows']
                        or values['bank_sha256'] != reference['bank_sha256']):
                    raise ValueError('width/transform capacity bank or packed identity differs')
                row['capacity'][bank] = values
        chroma = root/'reads/chroma/G_CHROMA_L.json'
        row['chroma'] = 'reference A0' if arm=='F' else (
            'pass' if load(chroma)['candidates'][name]['passes'] else 'fail') if chroma.exists() else None
        gate = Path.home()/'tmp/e36test/gates/mixF3_s0' if arm=='F' else root/f'reads/gates/{name}'
        steer_name = 'mixF3_s0' if arm=='F' else name
        path = gate/f'steer135/{steer_name}/result.json'
        row['steer'] = sum(r['pass'] for r in load(path)['rows']) if path.exists() else None
        dial_root = Path.home()/'tmp/e36test/gates/mixF3_s0-vi' if arm=='F' else gate
        row['C1_C6'] = {}
        for grid in ('standard','ladder'):
            paths = list(dial_root.glob(f'verdict-s0-*/{grid}/fulleval.json'))
            if len(paths)>1:
                raise ValueError('ambiguous width contract')
            if paths:
                checks = [r for r in load(paths[0])['dial']['addressability']['checks']
                          if r['id'] in ('C1','C2','C3','C4','C5','C6')]
                if len(checks)!=6:raise ValueError('incomplete contract artifact')
                row['C1_C6'][grid] = all(c['state']=='pass' for c in checks)
        row['HDR'] = {}
        for sub in ('hdr-panel','hdr-upiq'):
            path = root/f'reads/{sub}/panel.json'
            if path.exists():
                for pop, models in load(path).items():
                    if not pop.startswith('_') and name in models:
                        row['HDR'][pop] = {k:models[name][k] for k in
                            ('pooled_srocc','within_group_spearman_mean','inversions')}
        rows.append(row)
    for name,digest in sources.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('source artifact changed during comparison')
    receipt = dict(role='descriptive frozen-model comparison, no fit or selection', rows=rows,
        artifact_sha256=sources, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = O/'R13_COMPARISON.json'
    payload = json.dumps(receipt,indent=2)+'\n'
    if path.exists() and path.read_text()!=payload:
        number=1
        while path.with_name(f'R13_COMPARISON.bak{number}.json').exists():number+=1
        path.rename(path.with_name(f'R13_COMPARISON.bak{number}.json'))
    path.write_text(payload)
    lines=['# R14 beside completed R13 seed-zero reads', '',
        'All rows use seed 0. A0 is the frozen F control. The three A EVAL dictionaries are identical',
        'between the R13 and R14 reads. Missing width-lane artifacts are not measured here; no job',
        'was started in that lane. These descriptive EVAL reads select neither fits nor repeats.', '',
        '| Model | E36 inversions (%) | EVAL9 error (%) | EVAL8 excluding MDID (%) | Chroma | Steer /135 | C1–C6 standard / ladder |',
        '|---|---:|---:|---:|---|---:|---|']
    show = lambda value: 'not measured' if value is None else str(value)
    for row in rows:
        contracts=' / '.join('not measured' if grid not in row['C1_C6'] else
            'pass' if row['C1_C6'][grid] else 'fail' for grid in ('standard','ladder'))
        lines.append(f"| {row['label']} | {row['E36_percent']:.6f} | {row['EVAL9_percent']:.5f} | "
            f"{row['EVAL8_percent']:.5f} | {show(row['chroma'])} | {show(row['steer'])} | {contracts} |")
    lines.extend(['', 'EVAL error is the equal-collection mean normalized median residual; lower is better.',
        'EVAL8 excludes training-source MDID. E36 is a selection-source diagnostic, not held-out evidence.', '',
        '## Collection median residual / target 1–99% span (%)', '',
        '| Collection | '+' | '.join(r['label'] for r in rows)+' |', '|---|'+'---:|'*len(rows)])
    for collection in sorted(rows[0]['per_collection_median_percent']):
        lines.append('| '+collection+' | '+' | '.join(f"{r['per_collection_median_percent'][collection]:.5f}" for r in rows)+' |')
    lines.extend(['', '## Capacity on the identical complete TRAIN banks', '',
        '| Model | Bank | Dead / residual / live | Live pairs | Largest pair share (%) | Inputs at 50/90/99% mass |',
        '|---|---|---|---:|---:|---|'])
    for row in rows:
        for bank,v in row['capacity'].items():
            share='undefined' if v['largest_pair_share'] is None else f"{100*v['largest_pair_share']:.6f}"
            concentration=' / '.join(str(v['concentration'][k]) for k in ('0.5','0.9','0.99'))
            lines.append(f"| {row['label']} | {bank} | {v['dead']} / {v['residual']} / {v['live']} | "
                f"{v['live_pairs']} | {share} | {concentration} |")
    lines.extend(['', 'The completed R13 W512 fit retains two live pairs (four live units); the earlier',
        'one-pair description is not universal. A0 capacity is documented in the A weight audit.', ''])
    lines.extend(['', '## HDR: pooled SROCC / within-reference SROCC / inversions', '',
        '| Population | '+' | '.join(r['label'] for r in rows)+' |', '|---|'+'---:|'*len(rows)])
    for pop in ('4kdtb_eval','ladder_eval','hdrteach_val','upiq_dev','upiq_all'):
        values=[]
        for row in rows:
            v=row['HDR'].get(pop)
            values.append('not measured' if v is None else
                f"{v['pooled_srocc']:.6f} / {v['within_group_spearman_mean']:.6f} / {v['inversions']}")
        lines.append('| '+pop+' | '+' | '.join(values)+' |')
    lines.extend(['', 'Artifact/model hashes and exact extracted values: `R13_COMPARISON.json` in the R14 archive.',
        'This snapshot reports the width cells available when assembled; it does not claim the width lane is finished.', ''])
    (X/'benchmarks/r14_r13_comparison_2026-11-04.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
