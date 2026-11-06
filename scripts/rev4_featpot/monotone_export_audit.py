#!/usr/bin/env python3
"""Audit frozen exports, retaining failed feature gates; no optimizer steps.

Original models and native panels are copied verbatim into a separate evaluation
record. The only new candidates are all three registered spline re-exports.
Evaluation completion is distinct from scientific gate acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from activation_curriculum import sha, write


def passes(probes):
    return probes['identity']==100. and not probes['violations_over_1e_5'] and not probes['ceiling_violations']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','baseline-plan','binary-dir','repair-binary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    original_plan=json.loads((args.original/'PLAN.json').read_text())
    terminal=json.loads((args.original/'SUMMARY.json').read_text())
    if terminal['complete']+len(terminal['failures'])!=terminal['required'] or terminal['required']!=24:
        raise ValueError('all 24 original fits must terminate')
    if any('native feature gate failed:' not in f['error'] for f in terminal['failures']):
        raise ValueError('unresolved execution failure, distinct from measured feature failure')
    base=json.loads(args.baseline_plan.read_text());prepared_path=Path(base['output'])/'PREPARED.json'
    if sha(args.baseline_plan)!=original_plan['base_plan_sha256'] or sha(prepared_path)!=original_plan['prepared_sha256']:
        raise ValueError('input plan drift')
    prepared=json.loads(prepared_path.read_text())
    for group in prepared['groups']:
        p=Path(group['path'])
        for path,key in [(p,'sha256'),(Path(str(p)+'.manifest.json'),'manifest_sha256'),(p.with_suffix('.keys.parquet'),'keys_sha256')]:
            if sha(path)!=group[key]:raise ValueError('prepared input drift')
    for name,digest in original_plan['binary_sha256'].items():
        if sha(args.binary_dir/name)!=digest:raise ValueError('original native binary drift')
    probe=args.original/'feature-probes.parquet'
    if sha(probe)!=json.loads((args.original/'PROBES.json').read_text())['table_sha256']:
        raise ValueError('probe table drift')
    os.environ.update(REV4_V2_ROOT=base['root'],REV4_V2_BIN_DIR=str(args.binary_dir),
        ZEN_PANEL_BIN=str(args.binary_dir/'panel'),RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import v2_lodo_mlp as owner
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'DRIVER.py')
    plan={**original_plan,'arms':original_plan['arms']+['shared_spline_repaired'],
        'scope':'frozen export evaluation; original failed gates retained; zero new fits',
        'original_plan_sha256':sha(args.original/'PLAN.json'),'original_terminal_sha256':sha(args.original/'SUMMARY.json'),
        'repair_binary_sha256':sha(args.repair_binary),'evaluation_driver_sha256':sha(__file__),
        'original_fits':24,'new_fits':0,'new_spline_exports':3,'workers':1}
    write(args.output/'PLAN.json',plan)
    epoch=plan['epochs']-1;outcomes={};differences={};repairs={};pins={}
    inherited={(f['arm'],f['seed_index']) for f in terminal['failures']}
    for arm in plan['arms']:
        for i,pair in enumerate(plan['seed_pairs']):
            repaired=arm=='shared_spline_repaired';source_arm='shared_spline' if repaired else arm
            source=args.original/source_arm/f's{i}';dest=args.output/arm/f's{i}'
            dest.mkdir(parents=True);(dest/'ckpt').mkdir()
            source_bake=source/f'ckpt/ckpt_epoch{epoch:03d}.bin';bake=dest/f'ckpt/ckpt_epoch{epoch:03d}.bin'
            log=(source/'train.log').read_text()
            if not re.search(r'run-heavy: done rc=0\b',log):raise ValueError('original optimizer did not finish')
            shutil.copy2(source/'train.log',dest/'train.log')
            pins[str(source_bake)]=sha(source_bake)
            old_probes=json.loads((source/'PROBES.json').read_text())
            if passes(old_probes)==((source_arm,i) in inherited):raise ValueError('recorded gate outcome mismatch')
            if (source/'RESULT.json').exists():
                original_result=json.loads((source/'RESULT.json').read_text())
                if original_result['bake_sha256']!=sha(source_bake):raise ValueError('original bake drift')
            if repaired:
                command=[str(args.repair_binary),'repair-research-spline','--in',str(source_bake),'--out',str(bake)]
                result=subprocess.run(command,capture_output=True,text=True)
                (dest/'repair.log').write_text(result.stdout+'\n'+result.stderr);result.check_returncode()
                repairs[f'{arm}/s{i}']=json.loads(result.stdout)
                if repairs[f'{arm}/s{i}']['input_sha256']!=sha(source_bake) or repairs[f'{arm}/s{i}']['output_sha256']!=sha(bake):
                    raise ValueError('repair receipt bake drift')
            else:shutil.copy2(source_bake,bake)
            panels={};delta_by_population={}
            for group in prepared['groups']:
                name=group['name'];table=dest/(name+'.panel.parquet')
                if repaired:
                    prediction=owner.predict(bake,Path(group['path']),dest/(name+'.tsv'))
                    old=pq.read_table(source/(name+'.panel.parquet')).to_pydict()
                    targets=pq.read_table(group['path'],columns=['human_score','ref_basename']).to_pydict()
                    if old['target']!=targets['human_score'] or old['band']!=targets['ref_basename']:
                        raise ValueError('original native target/source order differs')
                    delta=np.abs(prediction-np.asarray(old['predicted']))
                    delta_by_population[name]=dict(rows=len(delta),changed=int(np.count_nonzero(delta)),
                        max_absolute=float(delta.max()),mean_absolute=float(delta.mean()))
                    pq.write_table(pa.table(dict(predicted=prediction,target=old['target'],band=old['band'])),table,compression='zstd')
                else:shutil.copy2(source/(name+'.panel.parquet'),table)
                pins[str(source/(name+'.panel.parquet'))]=sha(source/(name+'.panel.parquet'))
                panels[name]={}
                for mode in ('signed-quality','per-group','scatter'):
                    output=dest/(name+'.'+mode+'.log')
                    if repaired:
                        proc=subprocess.run([str(args.binary_dir/'panel'),'--input',str(table),'--json','--'+mode],capture_output=True,text=True)
                        output.write_text(proc.stdout+'\n'+proc.stderr);proc.check_returncode()
                    else:shutil.copy2(source/(name+'.'+mode+'.log'),output)
                    panels[name][mode]=json.loads(output.read_text())
                quality=panels[name]['signed-quality']
                if quality['n']!=pq.read_metadata(table).num_rows or quality['n_dropped'] or panels[name]['scatter']['status']!='MEASURED':
                    raise ValueError('incomplete native panel')
            probes=old_probes
            if repaired:
                prediction=owner.predict(bake,probe,dest/'feature-probes.tsv');delta=prediction[128:256]-prediction[:128]
                probes=dict(identity=float(prediction[256]),near_identity_ray=prediction[256:].tolist(),coordinate_pairs=128,
                    positive_deltas=int(np.sum(delta>0)),violations_over_1e_5=int(np.sum(delta>1e-5)),
                    max_increase=float(max(0.,delta.max())),ceiling_violations=int(np.sum(prediction>100.)))
                differences[f'{arm}/s{i}']=delta_by_population
            write(dest/'PROBES.json',probes)
            outcomes[f'{arm}/s{i}']='PASS' if passes(probes) else 'FAIL'
            write(dest/'RESULT.json',dict(arm=arm,seed_index=i,init_seed=pair[0],sample_seed=pair[1],
                bake_sha256=sha(bake),plan_sha256=sha(args.output/'PLAN.json'),panels=panels,feature_probes=probes,
                feature_gate_status=outcomes[f'{arm}/s{i}'],original_fit=str(source),original_fit_bake_sha256=sha(source_bake),
                fitting_steps_in_this_evaluation=0))
            print(json.dumps(dict(cell=f'{arm}/s{i}',gate=outcomes[f'{arm}/s{i}'],new_fit=False)),flush=True)
    write(args.output/'AUDIT.json',dict(original_gate_failures=terminal['failures'],feature_gates=outcomes,
        original_artifact_sha256=pins,repair_receipts=repairs,prediction_differences=differences,new_fits=0))
    write(args.output/'SUMMARY.json',dict(required=len(outcomes),complete=len(outcomes),failures=[],
        scope='evaluation completed; this is not an all-gates-pass claim',feature_gate_failures=[k for k,v in outcomes.items() if v=='FAIL'],
        original_fits=24,new_fits=0,audit_sha256=sha(args.output/'AUDIT.json')))
    if any(v=='FAIL' for k,v in outcomes.items() if k.startswith('shared_spline_repaired/')):
        raise ValueError('repaired export still fails the unchanged feature gate')


if __name__=='__main__':main()
