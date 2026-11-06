#!/usr/bin/env python3
"""Exact final replay first, then TRAIN-only intermediate source/family panels."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

from activation_curriculum import compare_neutral, sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('original','replays','baseline-plan','binary-dir','audit-binary','metadata','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    plan=json.loads((args.replays/'PLAN.json').read_text());original=json.loads((args.original/'PLAN.json').read_text())
    base=json.loads(args.baseline_plan.read_text());prepared_path=Path(base['output'])/'PREPARED.json'
    if sha(args.baseline_plan)!=plan['base_plan_sha256'] or sha(prepared_path)!=plan['prepared_sha256']:
        raise ValueError('stage input plan drift')
    if not plan['stage_replay'] or plan['checkpoint_interval']!=1:raise ValueError('not a registered checkpoint replay')
    for key in ('prepared_sha256','epochs','pairs_per_epoch','seed_pairs','binary_sha256'):
        if plan[key]!=original[key]:raise ValueError('stage replay differs: '+key)
    for name,digest in plan['binary_sha256'].items():
        if sha(args.binary_dir/name)!=digest:raise ValueError('stage binary drift: '+name)
    if sha(args.metadata)!=plan['source_audit']['metadata_sha256']:raise ValueError('stage metadata drift')
    os.environ.update(REV4_V2_ROOT=base['root'],REV4_V2_BIN_DIR=str(args.binary_dir),
        ZEN_PANEL_BIN=str(args.binary_dir/'panel'),TMPDIR=str(Path.home()/'tmp'),
        RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import v2_lodo_mlp as owner
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from lib.zen_stats import panel_batch_indexed
    prepared=json.loads(prepared_path.read_text())
    metadata=pq.read_table(args.metadata,columns=['pair_key','codec','ref_pixels_sha256']).to_pydict()
    offset=0;groups={}
    for group in prepared['groups']:
        table=Path(group['path']);keys=table.with_suffix('.keys.parquet')
        if sha(keys)!=group['keys_sha256']:raise ValueError('stage key drift')
        pair_keys=pq.read_table(keys,columns=['pair_key'])['pair_key'].to_pylist();end=offset+len(pair_keys)
        if metadata['pair_key'][offset:end]!=pair_keys:raise ValueError('stage key order mismatch')
        if group['split']=='fit':
            if sha(table)!=group['sha256'] or sha(Path(str(table)+'.manifest.json'))!=group['manifest_sha256']:
                raise ValueError('stage fitting table drift')
            target=pq.read_table(table,columns=['human_score'])['human_score'].to_numpy()
            labels={name:metadata[name][offset:end] for name in ('codec','ref_pixels_sha256')}
            jobs=[('0','p','t',None)];descriptions={'0':dict(scope='population',label='ALL')}
            for scope,key in [('family','codec'),('source','ref_pixels_sha256')]:
                buckets={}
                for i,label in enumerate(labels[key]):buckets.setdefault(label,[]).append(i)
                for label,ids in sorted(buckets.items()):
                    name=str(len(jobs));jobs.append((name,'p','t',ids));descriptions[name]=dict(scope=scope,label=label)
            groups[group['name']]=dict(table=table,target=target,jobs=jobs,descriptions=descriptions)
        offset=end
    if offset!=len(metadata['pair_key']) or len(groups)!=5:raise ValueError('stage TRAIN group coverage mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'PLAN.json',dict(replay_plan_sha256=sha(args.replays/'PLAN.json'),original_plan_sha256=sha(args.original/'PLAN.json'),
        script_sha256=sha(__file__),audit_binary_sha256=sha(args.audit_binary),epochs=[0,29,59,89,119],
        scope='TRAIN-only diagnostic; exact original final model replay required; no checkpoint selection'))
    completed={}
    while True:
        for arm in plan['arms']:
            if arm in completed:continue
            if not all((args.replays/arm/f's{i}'/'RESULT.json').exists() for i in range(3)):continue
            for i in range(3):
                if sha(args.replays/arm/f's{i}'/'HEAD.json')!=sha(args.original/arm/f's{i}'/'HEAD.json'):
                    raise ValueError('stage HEAD manifest differs')
                trajectories=[];event_receipts=[]
                for root in (args.replays,args.original):
                    cell=root/arm/f's{i}';result=json.loads((cell/'RESULT.json').read_text())
                    if sha(cell/'ckpt/ckpt_epoch119.bin')!=result['bake_sha256']:raise ValueError('stage final bake drift')
                    log=(cell/'train.log').read_text()
                    trajectories.append({int(m[1]):m[2] for m in re.finditer(r'epoch\s+(\d+)\s*\|(.+?)\| t=',log)})
                    event_receipts.append([json.loads(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('TRAINING_EVENTS ')])
                if trajectories[0]!=trajectories[1] or set(trajectories[0])!=set(range(120)):
                    raise ValueError('retained checkpoints changed a logged training trajectory')
                if event_receipts[0]!=event_receipts[1] or len(event_receipts[0])!=1:
                    raise ValueError('retained checkpoints changed native events')
                for epoch in range(120):
                    if not (args.replays/arm/f's{i}'/f'ckpt/ckpt_epoch{epoch:03d}.bin').is_file():
                        raise ValueError('missing retained epoch')
            compare_neutral(args.original/arm,args.replays/arm,args.audit_binary,args.output/arm/'replay-proof',epoch=119,strip_research=False)
            rows=[];bakes={}
            for i in range(3):
                for epoch in (0,29,59,89,119):
                    destination=args.output/arm/f's{i}'/f'e{epoch:03d}';destination.mkdir(parents=True)
                    bake=args.replays/arm/f's{i}'/f'ckpt/ckpt_epoch{epoch:03d}.bin';bakes[f's{i}/e{epoch}']=sha(bake)
                    for name,group in groups.items():
                        prediction=owner.predict(bake,group['table'],destination/(name+'.tsv'))
                        panel=panel_batch_indexed({'p':prediction,'t':group['target']},group['jobs'],stats='srocc')
                        if any(row['n_dropped'] or not np.isfinite(row['srocc_signed']) for row in panel):
                            raise ValueError('stage panel has missing/nonfinite correlation')
                        for row in panel:
                            rows.append(dict(arm=arm,seed_index=i,epoch=epoch,population=name,
                                **group['descriptions'][row['label']],n=row['n'],srocc_signed=row['srocc_signed']))
                        if epoch==119:
                            result=json.loads((args.replays/arm/f's{i}'/'RESULT.json').read_text())
                            if abs(panel[0]['srocc_signed']-result['panels'][name]['signed-quality']['srocc_signed'])>1e-12:
                                raise ValueError('stage final panel differs from original evaluation')
            pq.write_table(pa.Table.from_pylist(rows),args.output/arm/'PANELS.parquet',compression='zstd')
            write(args.output/arm/'STAGES.json',dict(bake_sha256=bakes,panels=rows,scope='TRAIN only'))
            completed[arm]=sha(args.output/arm/'STAGES.json')
            write(args.output/'PROGRESS.json',dict(required=len(plan['arms']),complete=len(completed),arms=completed))
            print(json.dumps(dict(stage_arms_complete=len(completed),required=len(plan['arms']),arm=arm)),flush=True)
        if len(completed)==len(plan['arms']):
            write(args.output/'SUMMARY.json',dict(required=len(plan['arms']),complete=len(completed),arms=completed,
                exact_final_replay_cells=3*len(completed),checkpoint_coverage=120*3*len(completed)))
            break
        if not args.watch:raise ValueError('replay cells still pending; use --watch or wait for completion')
        if (args.replays/'SUMMARY.json').exists():
            terminal=json.loads((args.replays/'SUMMARY.json').read_text())
            if terminal['failures']:raise ValueError('checkpoint replay failed; preserve all evidence')
        time.sleep(5)


if __name__=='__main__':main()
