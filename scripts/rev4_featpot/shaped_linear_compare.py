"""Matched-data Extra Trees and shared ReLU controls for the registered linear study."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shaped_linear_study import (load,sha,write,digest,clean,command,HEAVY,BASE,PREP,SEEDS,
                                permuted,weights_and_targets,metrics,RAW_IDS)


def prepare_neural(root,cell,d,plan):
    from v2_teacher import row_keys_sha,selection_sha
    from v2_common import acceptance_weight
    dest=root/'comparators'/cell['id'];admission=dest/'admission';tables=admission/'wide/main/real'
    tables.mkdir(parents=True,exist_ok=True)
    original=json.loads(PREP.read_text());base=json.loads(BASE.read_text())
    decision=Path(base['output'])/'root/human_role_decision.json'
    shutil.copy2(decision,admission/'human_role_decision.json')
    train=np.asarray(cell['train']);y=permuted(d['y'],d,train,cell['mode'],cell['seed']) if cell['mode']!='true' else np.array(d['y'])
    receipt=dict(schema='shaped-linear-comparator-receipt-v1',study_plan_sha256=sha(root/'PLAN.json'),
                 original_prepared_sha256=sha(PREP),admission_view={'source_receipt_sha256':json.loads(decision.read_text())['source_receipt_sha256']},legs={})
    groups=[];offset=0;records=[]
    for g in original['groups']:
        source=Path(g['path']);n=g['rows'];global_ids=np.arange(offset,offset+n);offset+=n
        if g['split']=='dev':
            groups.append((g['name'],source,0.,1.,'withinref,rank'));continue
        selected=np.flatnonzero(np.isin(global_ids,train));ids=global_ids[selected]
        if not len(ids):raise ValueError('empty comparator population')
        source_keys=source.with_suffix('.keys.parquet');manifest=Path(str(source)+'.manifest.json')
        for p,k in ((source,'sha256'),(source_keys,'keys_sha256'),(manifest,'manifest_sha256')):
            if sha(p)!=g[k]:raise ValueError('comparator original input drift')
        keys=pq.read_table(source_keys).take(selected);t=pq.read_table(source).take(selected)
        original_labels=t['human_score'].to_numpy()
        if not np.array_equal(original_labels,d['y'][ids]):raise ValueError('label mapping alignment differs')
        t=t.set_column(t.schema.get_field_index('human_score'),'human_score',pa.array(y[ids],type=t['human_score'].type))
        out=tables/source.name;kp=out.with_suffix('.keys.parquet')
        pq.write_table(keys,kp,compression='zstd');pq.write_table(t,out,compression='zstd')
        old=json.loads(manifest.read_text());m=dict(old)
        m.update(source_table_sha256=sha(source),source_manifest_sha256=sha(manifest),table_sha256=sha(out),keys_sha256=sha(kp),row_keys_sha256=row_keys_sha(keys),rows_in=n,rows_kept=len(ids),
                 row_selection_sha256=selection_sha(selected),targets_changed=int(np.sum(y[ids]!=original_labels)),immutable_input_roots=[*old['immutable_input_roots'],str(admission)],
                 composite='Revision-7 source-subset memorization diagnostic; measured features unchanged; declared training-label permutation',
                 shaped_linear_comparator=dict(study_plan_sha256=sha(root/'PLAN.json'),mode=cell['mode'],seed=cell['seed'],outer=cell['outer'],training_indices_sha256=selection_sha(ids),original_manifest_sha256=sha(manifest)),
                 human_ssim2_trial={**old.get('human_ssim2_trial',{}),'target_origin':'original labels' if cell['mode']=='true' else 'permutation of admitted training labels','target_transform':cell['mode']})
        write(Path(str(out)+'.manifest.json'),m)
        receipt['legs'][g['name']]={'fit':dict(rel=str(source),sha256=sha(source),manifest_sha256=sha(manifest),rows=n)}
        refs=keys['ref_basename'].to_pylist()
        groups.append((g['name'],out,acceptance_weight(1.,refs),0.,'withinref,both' if g['name']=='cid22' else 'withinref,rank'))
        records.append(dict(name=g['name'],path=str(out),table_sha256=sha(out),manifest_sha256=sha(str(out)+'.manifest.json'),keys_sha256=sha(kp),rows=len(ids)))
    write(tables/'receipt.json',receipt)
    write(dest/'ADMISSION.json',dict(cell=cell,groups=records,receipt_sha256=sha(tables/'receipt.json'),decision_sha256=sha(admission/'human_role_decision.json'),originals_unchanged=True))
    return groups,admission/'human_role_decision.json',y


def comparison_cell(rootstr,cell):
    root=Path(rootstr);plan,d=load(root);dest=root/'comparators'/cell['id'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return json.loads((dest/'COMPLETE.json').read_text())
    os.environ.update(REV4_V2_BIN_DIR=str(root/'bin'),ZEN_PANEL_BIN=str(root/'bin/panel'))
    train=np.asarray(cell['train']);test=np.asarray(cell['test']);n=len(d['y'])
    y=permuted(d['y'],d,train,cell['mode'],cell['seed']) if cell['mode']!='true' else np.array(d['y'])
    w,target,maps=weights_and_targets(y[train],d['pop'][train],d['component'][train],'range',1.)
    write(dest/'CELL.json',cell)
    if cell['model']=='extra':
        from tree_study import estimator,native_gate
        config=plan['comparison']['extra'];model=estimator(config,cell['seed'],576)
        model.fit(d['raw'][train],target,sample_weight=w)
        pred,gate=native_gate(model,np.array(d['raw']),dest/'model',root/'bin/corrhead_parity',dict(study_plan_sha256=sha(root/'PLAN.json'),cell=cell['id']))
        pred=pred*100
        np.savez_compressed(dest/'predictions.npz',native=pred)
        provenance=dict(native_gate=gate,training_objective='reference-balanced squared error on training-only range map')
    else:
        import v2_common
        import v2_lodo_mlp as owner
        v2_common.FITBIN=owner.FITBIN=root/'bin/bake_dial_refit'
        owner.EPOCHS,owner.PAIRS_PER_EPOCH,owner.LOG_EVERY=120,5000,1
        groups,decision,y=prepare_neural(root,cell,d,plan)
        keep=dest/'keep.txt';keep.write_text('\n'.join(map(str,plan['declaration']['direct']))+'\n')
        recipe={**json.loads(BASE.read_text())['recipe'],'shared_hidden_activation':'relu','mse_weight':1}
        argv=owner.train_command(groups,cell['seed'],cell['seed']+100000000,1853,keep,'N',dest/'best.bin',recipe,strict_admission=True,data_role_decision=decision)
        ckpt=dest/'ckpt';ckpt.mkdir(exist_ok=True)
        argv+=['--l2','1e-5','--no-sample-coverage','--dump-checkpoints-every','119','--dump-checkpoints-dir',str(ckpt)]
        env=os.environ.copy()
        for k in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256'):env.pop(k,None)
        with (dest/'train.log').open('w') as log:
            import subprocess
            proc=subprocess.Popen([str(HEAVY),'--mem','3G','--jobs','1','--',*argv],stdout=log,stderr=subprocess.STDOUT,env=env)
            write(dest/'RUN.json',dict(command=argv,pid=proc.pid,stage='training',cell=cell['id'],started=datetime.now(timezone.utc).isoformat()))
            rc=proc.wait()
        if rc:raise RuntimeError(f'native comparator rc={rc}: {dest}/train.log')
        curve=owner.read_curve(dest/'train.log')
        if set(curve)!=set(range(120)):raise ValueError('incomplete fixed neural budget')
        bake=ckpt/'ckpt_epoch119.bin'
        preds=[]
        for g in json.loads(PREP.read_text())['groups']:
            preds.append(owner.predict(bake,Path(g['path']),dest/(g['name']+'.tsv')))
        pred=np.concatenate(preds);np.savez_compressed(dest/'predictions.npz',native=pred)
        # Reuse the independently admitted feature-space probes, not labels or images.
        raw=d['x'][n:];positions={v:i for i,v in enumerate(RAW_IDS)}
        fields={f'f{i}':pa.array(raw[:,positions[i]] if i in positions else np.zeros(len(raw),np.float32)) for i in range(1853)}
        fields.update(human_score=pa.array(np.zeros(len(raw))),ref_basename=pa.array(['probe']*len(raw)))
        probe=dest/'probes.parquet';pq.write_table(pa.table(fields),probe,compression='zstd');write(Path(str(probe)+'.manifest.json'),dict(formula_revision=7,scope='feature-space diagnostic only',rows=len(raw)))
        p=owner.predict(bake,probe,dest/'probe-predictions.tsv');delta=p[128:256]-p[:128]
        gate=dict(identity=float(p[256]),max_coordinate_increase=float(max(0,delta.max())),violations_over_1e_5=int(np.sum(delta>1e-5)),finite=bool(np.isfinite(p).all()),ceiling_violations=int(np.sum(p>100)))
        write(dest/'PROBES.json',gate)
        if gate['identity']!=100 or gate['violations_over_1e_5'] or not gate['finite'] or gate['ceiling_violations']:raise ValueError('native comparator feature gate failed')
        provenance=dict(bake_sha256=sha(bake),epochs=120,pairs_per_epoch=5000,selection='fixed epoch 119',gate=gate,
                        training_objective='within-reference human ranking plus teacher ranking/MSE; same rows, distinct objective from quadratic control')
    report=dict(cell=cell,provenance=provenance,train=metrics(pred,y,train,d,maps),test=metrics(pred,d['y'],test,d,maps),train_actual=metrics(pred,d['y'],train,d,maps))
    write(dest/'REPORT.json',clean(report));complete=dict(id=cell['id'],model=cell['model'],mode=cell['mode'],report_sha256=sha(dest/'REPORT.json'))
    write(dest/'COMPLETE.json',complete);return complete


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--model',choices=['extra','neural','all'],default='all');a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('four workers maximum')
    plan,d=load(a.root)
    for path,pin in {**plan['binary_sha256'],**plan['data_sha256']}.items():
        if sha(path)!=pin:raise ValueError('frozen comparator input drift')
    fit=np.flatnonzero(~d['dev']);cells=[]
    for model in ('extra','neural'):
        for outer in range(3):
            tr=fit[d['fold'][fit]!=outer];te=fit[d['fold'][fit]==outer]
            for mode in ('true','population','conditional'):
                for seed in SEEDS:cells.append(dict(id=f'{model}-o{outer}-{mode}-s{seed}',model=model,outer=outer,mode=mode,seed=seed,train=tr.tolist(),test=te.tolist()))
    matrix=a.root/'S4-comparison.json'
    if matrix.exists() and json.loads(matrix.read_text())!=cells:raise ValueError('comparator matrix drift')
    write(matrix,cells)
    selected=[c for c in cells if a.model=='all' or c['model']==a.model]
    status=dict(stage='S4_comparators',coordinator_pid=os.getpid(),required=len(selected),completed=0,failures=[])
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(comparison_cell,str(a.root),c):c for c in selected}
        for f in as_completed(futures):
            try:f.result();status['completed']+=1
            except Exception as e:status['failures'].append(dict(id=futures[f]['id'],error=str(e)))
            status['last_check']=datetime.now(timezone.utc).isoformat();write(a.root/f'S4-{a.model}-STATUS.json',status);print(json.dumps(status),flush=True)
    if status['failures']:raise RuntimeError('comparator failures preserved')


if __name__=='__main__':main()
