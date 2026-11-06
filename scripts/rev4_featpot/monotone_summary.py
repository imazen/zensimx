#!/usr/bin/env python3
"""Reduce completed native studies without choosing an arm or a checkpoint.

All correlations and scatter statistics come from the Rust panel. Python only
aggregates independent seeds, verifies provenance and compares event receipts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean

from activation_curriculum import sha, write


def spread(values):
    return dict(mean=mean(values), minimum=min(values), maximum=max(values), values=values)


def records(log, prefix):
    return [json.loads(line.removeprefix(prefix)) for line in log.splitlines() if line.startswith(prefix)]


def summarize(root, breakdowns, output, control_root=None):
    plan_path=root/'PLAN.json';plan=json.loads(plan_path.read_text())
    complete=json.loads((root/'SUMMARY.json').read_text())
    if complete['complete']!=complete['required'] or complete['failures']:
        raise ValueError('summary requires every registered cell and no unresolved failures')
    index=json.loads((breakdowns/'INDEX.json').read_text())
    if index['plan_sha256']!=sha(plan_path):raise ValueError('breakdown plan drift')
    cells={};rows=[];events={};diagnostics={}
    for arm in plan['arms']:
        for seed_index,seed_pair in enumerate(plan['seed_pairs']):
            cell=f'{arm}/s{seed_index}';folder=root/cell
            result_path=folder/'RESULT.json';result=json.loads(result_path.read_text())
            breakdown_path=breakdowns/cell/'FAMILIES.json';detail=json.loads(breakdown_path.read_text())
            if sha(breakdown_path)!=index['cells'][cell] or sha(result_path)!=detail['source_result_sha256']:
                raise ValueError('result or breakdown drift: '+cell)
            if result['plan_sha256']!=sha(plan_path) or [result['init_seed'],result['sample_seed']]!=seed_pair:
                raise ValueError('result plan/seed mismatch: '+cell)
            bake=folder/f"ckpt/ckpt_epoch{plan['epochs']-1:03d}.bin"
            if sha(bake)!=result['bake_sha256']:raise ValueError('bake drift: '+cell)
            metrics={}
            for population,panel in result['panels'].items():
                if sha(folder/(population+'.panel.parquet'))!=detail['prediction_table_sha256'][population]:
                    raise ValueError('prediction table drift: '+cell)
                q=panel['signed-quality'];raw=panel['scatter']['raw'];geo=panel['scatter']['geo']
                metric=dict(pooled_srocc=q['srocc_signed'],raw_plcc=q['plcc_raw'],logistic_plcc=q['plcc'],
                    raw_tail_p99=raw['p99'],raw_tail_max=raw['max'],geo_tail_p99=geo['p99d'],geo_tail_max=geo['maxd'],
                    prediction_min=raw['pred_min'],prediction_max=raw['pred_max'])
                for scope,owner in [('source','sources'),('family','panels'),('source_family','source_families')]:
                    groups=detail[owner][population]
                    values=[g['srocc_signed'] for g in groups if g['srocc_signed'] is not None]
                    if not values:raise ValueError('no defined subgroup correlations: '+cell+'/'+population+'/'+scope)
                    metric[scope+'_mean_srocc']=mean(values)
                    metric[scope+'_min_srocc']=min(values)
                    metric[scope+'_defined']=len(values)
                    metric[scope+'_undefined']=len(groups)-len(values)
                metrics[population]=metric
                rows.append(dict(arm=arm,seed_index=seed_index,population=population,**metric))
            log=(folder/'train.log').read_text()
            if not re.search(r'run-heavy: done rc=0\b',log):raise ValueError('successful resource receipt missing: '+cell)
            native_events=records(log,'TRAINING_EVENTS ')
            if plan.get('training',False):
                if len(native_events)!=1:raise ValueError('native event receipt missing/duplicated: '+cell)
                event=native_events[0]
                if event['events']!=plan['epochs']*plan['pairs_per_epoch'] or not event['multiset_equal']:
                    raise ValueError('native event count/multiset failure: '+cell)
                events[cell]=event
                diag=records(log,'TRAINING_DIAGNOSTICS ')
                if [d['epoch'] for d in diag]!=list(range(plan['epochs'])):
                    raise ValueError('incomplete diagnostic epochs: '+cell)
                diagnostics[cell]=dict(first=diag[0],final=diag[-1],
                    note='sampled training responses; no claim of global inactivity or a local minimum')
            cells[cell]=dict(result_sha256=sha(result_path),bake_sha256=sha(bake),metrics=metrics,
                probes=result['feature_probes'])
    if len(cells)!=complete['required'] or set(cells)!=set(index['cells']):raise ValueError('cell coverage mismatch')
    event_checks={}
    if plan.get('training',False):
        control_root=control_root or root
        control_plan=json.loads((control_root/'PLAN.json').read_text())
        for key in ('prepared_sha256','epochs','pairs_per_epoch','seed_pairs'):
            if plan[key]!=control_plan[key]:raise ValueError('unmatched event control: '+key)
        for cell,event in events.items():
            arm,seed=cell.split('/')
            control_log=(control_root/'control'/seed/'train.log').read_text()
            control=records(control_log,'TRAINING_EVENTS ')[0]
            options=plan['training_options'][arm]
            changed_mass=options.get('teacher_mass',1.)!=1.
            checks=dict(teacher_mass_changed=changed_mass,
                original_matches_control=event['original_digest']==control['original_digest'],
                tail_matches_control=event['tail_digest']==control['tail_digest'],
                mass_counts_match_control=event['mass']==control['mass'],
                ordered_matches_control=event['ordered_digest']==control['ordered_digest'])
            if not changed_mass and not all(checks[k] for k in ('original_matches_control','tail_matches_control','mass_counts_match_control')):
                raise ValueError('matched exposure/tail failure: '+cell)
            if not changed_mass and options.get('order','random')=='random' and not checks['ordered_matches_control']:
                raise ValueError('random-event order drift: '+cell)
            if options.get('teacher_mass')==0 and any(m['pair_events'] for m in event['mass'] if m['group']=='cid22'):
                raise ValueError('zero-mass teacher still sampled: '+cell)
            event_checks[cell]=checks
    aggregates={}
    for arm in plan['arms']:
        selected=[cells[f'{arm}/s{i}']['metrics'] for i in range(len(plan['seed_pairs']))]
        aggregates[arm]={population:{metric:spread([cell[population][metric] for cell in selected])
            for metric in selected[0][population]} for population in selected[0]}
    output.mkdir(parents=True,exist_ok=False)
    write(output/'SUMMARY.json',dict(schema='native-monotone-summary-v1',plan_sha256=sha(plan_path),
        breakdown_index_sha256=sha(breakdowns/'INDEX.json'),completed=len(cells),required=complete['required'],
        selection='none; fixed final checkpoints, every registered arm',
        interpretation='seed ranges are descriptive, not confidence intervals; development is already exposed',
        arms=aggregates,cells=cells,events=events,event_checks=event_checks,diagnostics=diagnostics))
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    populations=['human_kadid_development','human_tid2013_development','human_konfig_development','human_cid22_a25_development']
    text=['# Fixed-final development comparison','',f"Completed {len(cells)}/{complete['required']} cells; {plan['epochs']} epochs x {plan['pairs_per_epoch']} draws.",
        '','Three-seed means; ranges and all fitting panels are in SUMMARY.json. No model selection.']
    for metric in ('pooled_srocc','source_mean_srocc','family_min_srocc','geo_tail_p99','raw_tail_p99'):
        text+=['',f'## {metric}','','| Arm | KADID | TID2013 | KonFiG F | CID22 human |','|---|---:|---:|---:|---:|']
        for arm in plan['arms']:
            text.append('| '+arm+' | '+' | '.join(f"{aggregates[arm][p][metric]['mean']:.6f}" for p in populations)+' |')
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(completed=len(cells),summary_sha256=sha(output/'SUMMARY.json'),event_checks=len(event_checks))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','breakdowns','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--control-root',type=Path)
    args=parser.parse_args()
    summarize(args.root,args.breakdowns,args.output,args.control_root)


if __name__=='__main__':main()
