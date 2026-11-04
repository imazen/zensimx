"""Summarize frozen R14 reads and the registered seed-repeat eligibility."""
import json
from pathlib import Path
from statistics import mean

import r14_transforms as fixed

O=fixed.O


def load(path):
    return json.loads(Path(path).read_text())


def main():
    fits=load(O/'FROZEN_FITS.json')['models']
    models=load(O/'MODELS.json')
    release=load(O/'T2_RELEASE.json')
    if len(fits)!=len(models) or len(fits)!=3+int(release['run']):
        raise ValueError('all released fits required')
    receipt=load(O/'READ_TABLE_RECEIPT.json')
    if receipt['frozen_sha256']!=fixed.owner.sha(O/'FROZEN_FITS.json'):
        raise ValueError('complete read-table receipt required')
    development=load(O/'FINAL_DEVELOPMENT.json')
    primary=load(O/'reads/PRIMARY.json')
    readout=load(O/'reads/readout.json')
    chroma=load(O/'reads/chroma/G_CHROMA_L.json')['candidates']
    capacity=load(O/'CAPACITY.json')
    rows=[];eligibility={}
    for arm,seed in [('F',s) for s in range(3)]+[(r['arm'],r['seed']) for r in fits]:
        name=f'{arm}_s{seed}'
        errors=readout['per'][f'{arm}@{seed}']
        row=dict(arm=arm,seed=seed,development_MAE=development['pooled'][name],
            E36_inversion_percent=100*primary[arm]['seeds'][name],
            EVAL_nine_collection_mean_median_error_percent=100*mean(v['med'] for v in errors.values()),
            EVAL_eight_excluding_mdid_mean_median_error_percent=100*mean(v['med'] for k,v in errors.items() if k!='mdid'))
        if arm!='F':
            fit=next(r for r in fits if r['arm']==arm and r['seed']==seed)
            base=O/f'reads/gates/{name}'
            steer=load(base/f'steer135/{name}/result.json')['rows']
            if len(steer)!=135:raise ValueError('complete steering population required')
            contracts={};addressability={};dial={}
            for grid in ('standard','ladder'):
                path,=list(base.glob(f'verdict-s0-*/{grid}/fulleval.json'))
                g=load(path)['dial']['addressability']
                checks=[c for c in g['checks'] if c['id'] in ('C1','C2','C3','C4','C5','C6')]
                if len(checks)!=6:raise ValueError('complete C1–C6 contract required')
                contracts[grid]=all(c['state']=='pass' for c in checks)
                addressability[grid]=dict(regression=g['regression'],shippable=g['shippable'])
                values=g['measured']['grid']
                dial[grid]=values['p5']<=25 and values['p95']>=85 and values['mono']>=.93
            row.update(chroma_pass=chroma[name]['passes'],steering_successes=sum(r['pass'] for r in steer),
                contracts=contracts,addressability=addressability,G_DIAL=dial,
                cid22_capacity=capacity[f'cid22/{name}'],hdr_capacity=capacity[f'hdr/{name}'],
                fit_wall_seconds=fit['wall_seconds'],packed_sha256=fit['packed_sha256'])
            # Existing E33 gate owner: 135 cases, at least 128 successes.
            conditions=dict(chroma=bool(row['chroma_pass']),steer=row['steering_successes']>=128,
                C1_C6=all(contracts.values()),G_ADDR=all(v['shippable'] for v in addressability.values()),
                development_strict_improvement=row['development_MAE']<development['pooled']['F_s0'])
            eligibility[name]=dict(conditions=conditions,steering_bar=128,eligible=all(conditions.values()),
                disposition='requires owner cost decision before any repeat' if all(conditions.values()) else
                    'not eligible under registration; no seed repeats',
                failed_or_unmeasured=[k for k,v in conditions.items() if not v])
        rows.append(row)
    fixed.owner.write(O/'REPORT_ROWS.json',rows)
    fixed.owner.write(O/'SEED_ELIGIBILITY.json',dict(registration=release['registration'],arms=eligibility,
        source_sha256=fixed.owner.sha(__file__),role='registered label-free/development eligibility only; EVAL does not select repeats'))
    lines=['| Arm | Seed | TRAIN dev MAE | E36 inversions (%) | EVAL all-nine error (%) | EVAL eight excluding MDID (%) | Chroma | Steer /135 | Dead / residual / live | Fit wall (s) |',
           '|---|---:|---:|---:|---:|---:|---|---:|---|---:|']
    for r in rows:
        if r['arm']=='F':chroma_value='reference A0' if r['seed']==0 else 'not rescored';steer='see full reads';cap='see A audit';wall='not refitted'
        else:
            c=r['cid22_capacity'];cap=f"{c['dead']}/{c['residual']}/{c['live']}"
            chroma_value='pass' if r['chroma_pass'] else 'fail';steer=str(r['steering_successes']);wall=f"{r['fit_wall_seconds']:.2f}"
        lines.append(f"| {'A' if r['arm']=='F' else r['arm']} | {r['seed']} | {r['development_MAE']:.6f} | {r['E36_inversion_percent']:.6f} | {r['EVAL_nine_collection_mean_median_error_percent']:.5f} | {r['EVAL_eight_excluding_mdid_mean_median_error_percent']:.5f} | {chroma_value} | {steer} | {cap} | {wall} |")
    (O/'SUMMARY_TABLE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    print(json.dumps(eligibility,indent=2),flush=True)


if __name__=='__main__':main()
