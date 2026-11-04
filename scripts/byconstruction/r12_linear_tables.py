"""R12 descriptive tables from frozen-model read owners; no fits or selection.

Three seed means are descriptive, not pooled correlations. Original JSON retains
per-seed statistics and intervals. The older experiment-5 arm-name classifier
does not recognize R12 names, so its `heldout` summary is not used here.
"""
import argparse
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--with-extra', action='store_true')
    ap.add_argument('--arms', help='explicit frozen arm inventory, including F control')
    ap.add_argument('--title', default='R12 measured read tables')
    ap.add_argument('--candidate-seeds', default='0,1,2', help='registered candidate seeds; F control remains 0,1,2')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root = a.root
    arms = ['F', 'L1', 'L2', 'L3']
    if a.with_extra:
        arms = ['F', 'L0', 'L1', 'L2', 'L3', 'L4']
    if a.arms:
        arms = a.arms.split(',')
    candidate_seeds=[int(s) for s in a.candidate_seeds.split(',')]
    def reads(arm):
        return root / ('extra/reads' if arm in ('L0', 'L4') else 'reads')
    def names(arm, sep='_s'):
        return [f'{arm}{sep}{s}' for s in (range(3) if arm=='F' else candidate_seeds)]
    lines = [f'# {a.title}', '',
             'A is the registered F baseline. Cells are seed means unless seed values are separated by `/`.',
             'No across-seed interval is implied. Per-seed whole-reference/family intervals remain in the native read JSON.', '']
    if candidate_seeds!=[0,1,2]:
        lines[2]=f'A is the registered F baseline, seeds 0–2. Candidate seeds: {a.candidate_seeds}; means use only those registered seeds.'
        lines.insert(4,'Candidate seeds 1–2: not run (owner cut on cost, 2026-11-04). No candidate across-seed stability is measured.')
    def table(title, labels, rows):
        lines.extend([f'## {title}', '', '| Read | ' + ' | '.join(labels) + ' |',
                      '|---|' + '---:|' * len(labels)])
        for name, values in rows:
            lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
        lines.append('')
    labels = ['A' if arm == 'F' else arm for arm in arms]
    primary = {arm: load(reads(arm) / 'PRIMARY.json')[arm] for arm in arms}
    rows = [('E36 inversion share (%)', [f"{100 * primary[arm]['share' if arm == 'F' else 'arm_share']:.6f}" for arm in arms]),
            ('E36 change vs A (percentage points), 95% source bootstrap',
             ['baseline' if arm == 'F' else f"{100 * primary[arm]['diff']:+.6f} [{100 * primary[arm]['ci95'][0]:+.6f}, {100 * primary[arm]['ci95'][1]:+.6f}]" for arm in arms])]
    table('E36: 40 origins, 646,677 selection pairs, 10,000 draws', labels, rows)
    original = load(root / 'reads/readout.json')
    collections = sorted(original['per']['F@0'])
    readouts = {arm: load(reads(arm) / 'readout.json') for arm in arms}
    def eval_mean(arm, collection, metric):
        return mean(readouts[arm]['per'][name][collection][metric] for name in names(arm, '@'))
    for metric, title, scale in [('med', 'EVAL MOS median residual / target 1–99% span (%)', 100),
                                 ('p90', 'EVAL MOS p90 residual / target 1–99% span (%)', 100),
                                 ('srocc', 'EVAL MOS absolute SROCC', 1)]:
        rows = [(f"{c} (n={original['per']['F@0'][c]['n']})", [f'{scale * eval_mean(arm, c, metric):.5f}' for arm in arms]) for c in collections]
        if metric != 'p90':
            for title_row, cols in [('all nine, equal collection mean', collections),
                                    ('eight collections excluding training-source MDID', [c for c in collections if c != 'mdid']),
                                    ('common three: IQA-ODI, NITS-IQA, VCL-FER', ['iqa-odi', 'nits-iqa', 'vcl-fer'])]:
                rows.append((title_row, [f'{scale * mean(eval_mean(arm, c, metric) for c in cols):.5f}' for arm in arms]))
        table(title, labels, rows)
    for mode in ('compression', 'noncodec'):
        data = {arm: load(reads(arm) / f'ceval-{mode}.json') for arm in arms}
        cols = [c for c in data['F']['F@0'] if c != 'summary']
        rows = []
        for c in cols:
            values = []
            for arm in arms:
                ss = [data[arm][n][c] for n in names(arm, '@')]
                values.append(f"{mean(v['srocc'] for v in ss):.5f} / {mean(v['within'] for v in ss):.5f} / {100 * mean(v['err'] for v in ss):.3f}%")
            rows.append((f"{c} (n={data['F']['F@0'][c]['n']})", values))
        rows.append(('row-weighted SROCC / equal-collection error',
                     [f"{mean(data[arm][n]['summary']['srocc_rows'] for n in names(arm, '@')):.5f} / {100 * mean(data[arm][n]['summary']['err_mean'] for n in names(arm, '@')):.3f}%" for arm in arms]))
        table(f'EVAL {mode}: signed SROCC / within-reference SROCC / normalized median error', labels, rows)
    chroma = {arm: load(reads(arm) / 'chroma/G_CHROMA_L.json') for arm in arms}
    rows = []
    for family in ('ycc444', 'ycc420', 'xyb-capped', 'xyb-extended', 'band-ac'):
        values = []
        for arm in arms:
            if arm == 'F':
                v = next(iter(chroma[arm]['candidates'].values()))['per_family'][family]['mae_baseline']
            else:
                v = mean(chroma[arm]['candidates'][n]['per_family'][family]['mae_candidate'] for n in names(arm))
            values.append(f'{v:.6f}')
        rows.append((family, values))
    rows.append(('level pass / rank pass (seeds)', ['reference A0' if arm == 'F' else
                 f"{sum(chroma[arm]['candidates'][n]['passes'] for n in names(arm))}/{len(names(arm))} / {sum(chroma[arm]['candidates'][n]['rank_non_regression']['passes'] for n in names(arm))}/{len(names(arm))}" for arm in arms]))
    table('G-CHROMA-L: dev views >64 px; MAE in score points against A0', labels, rows)
    gate_rows = {arm: [] for arm in arms}
    h = Path.home() / 'tmp/e36test'
    for arm in arms:
        for name in names(arm):
            s=int(name.rsplit('_s',1)[1])
            gate = h / f'gates/mixF3_s{s}' if arm == 'F' else reads(arm) / f'gates/{name}'
            steer_name = f'mixF3_s{s}' if arm == 'F' else name
            steer = load(gate / f'steer135/{steer_name}/result.json')['rows']
            dial_root = h / f'gates/mixF3_s{s}-vi' if arm == 'F' else gate
            contracts = []
            for grid in ('standard', 'ladder'):
                paths = list(dial_root.glob(f'verdict-s0-*/{grid}/fulleval.json'))
                if len(paths) != 1:
                    raise ValueError(f'{name}: missing or ambiguous {grid} verdict')
                checks = load(paths[0])['dial']['addressability']['checks']
                cc = [c['state'] == 'pass' for c in checks if c['id'] in ('C1','C2','C3','C4','C5','C6')]
                assert len(cc) == 6
                contracts.append(all(cc))
            near = load(h / f'nearid/out-mixf3s{s}/candidate-mixf3s{s}.GATES.json' if arm == 'F' else reads(arm) / f'nearid/{name}/candidate-{name.replace("_", "").lower()}.GATES.json')
            identity = [json.loads(line) for line in (dial_root / 'identity-s0.jsonl').read_text().splitlines()]
            identity = [v for v in identity if not v.get('summary')]
            assert len(identity) == 744
            identity_ok = all(v['candidate_score_bits'] == [4636737291354636288]
                              and all(v['candidate_read_set_bits_equal_first_permutation']) for v in identity)
            gate_rows[arm].append(dict(steer=sum(v['pass'] for v in steer), contracts=all(contracts),
                                       N1=near['N1']['pass'], N2=near['N2']['pass'],
                                       N3=near['N3']['monotone_ladders'], identity=identity_ok))
    rows = [('G-STEER successes /135', [' / '.join(str(v['steer']) for v in gate_rows[arm]) for arm in arms]),
            ('C1–C6, both standard and ladder grids', [f"{sum(v['contracts'] for v in gate_rows[arm])}/{len(names(arm))} pass" for arm in arms]),
            ('Identity: 744 scores exactly 100; read sets invariant', [f"{sum(v['identity'] for v in gate_rows[arm])}/{len(names(arm))} pass" for arm in arms]),
            ('NEARID N1 / N2', [f"{sum(v['N1'] for v in gate_rows[arm])}/{len(names(arm))} / {sum(v['N2'] for v in gate_rows[arm])}/{len(names(arm))} pass" for arm in arms]),
            ('NEARID N3 monotone ladders /144 (bar 122)', [' / '.join(str(v['N3']) for v in gate_rows[arm]) for arm in arms])]
    table('Label-free pixel and dial checks', labels, rows)
    hdr = {arm: load(reads(arm) / 'hdr-panel/panel.json') for arm in arms}
    for arm in arms:
        hdr[arm]['upiq_all'] = load(reads(arm) / 'hdr-upiq/panel.json')['upiq_all']
    for metric, title in [('pooled_srocc','pooled signed SROCC'), ('pooled_plcc','raw pooled PLCC'),
                          ('within_group_spearman_mean','mean within-reference Spearman'), ('inversions','within-reference inversion count')]:
        rows = []
        for pop in hdr['F']:
            exemplar = hdr['F'][pop]['F_s0']
            values = [f"{mean(hdr[arm][pop][n][metric] for n in names(arm)):.6f}" for arm in arms]
            if metric == 'inversions':
                values = [' / '.join(str(hdr[arm][pop][n][metric]) for n in names(arm)) for arm in arms]
            rows.append((f"{pop} (n={exemplar['rows']}, refs={exemplar['groups']})", values))
        table(f'HDR cube-root:203 — {title}', labels, rows)
    lines.extend(['Pooled ladder correlations are descriptive only; the registered ladder outcome is within-ladder rank.',
                  '4Kdtb has one family, so its family bootstrap interval is undefined. HDR is report-only; no HDR training leg was added.',
                  'The EVAL logistic fit is an assessment transform, never a fitted served-model update.', ''])
    a.out.write_text('\n'.join(lines).replace('/ nan /', '/ undefined /'))


if __name__ == '__main__':
    main()
