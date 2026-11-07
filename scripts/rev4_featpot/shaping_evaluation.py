#!/usr/bin/env python3
"""Native inner-fold evaluation; outer results are refused by this entry point."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TMPDIR']=str(Path.home()/'tmp')
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shaping_completion import REPO, encoded, sha
from shaping_execution import load_registration


DRIVER_SHA256=sha(__file__)

class NativeFeatureGateError(ValueError):
    """Measured feature-contract failure; never a qualified candidate."""


def prepare_validation(root,partition,membership):
    from shaped_linear_study import PREP
    from v2_teacher import row_keys_sha,selection_sha
    dest=root/'evaluation_views'/partition
    if dest.exists():
        receipt=json.loads((dest/'RECEIPT.json').read_text())
        if receipt['indices']!=membership['test']:raise ValueError('validation membership drift')
        for g in receipt['groups']:
            for path,pin in g['file_sha256'].items():
                if sha(path)!=pin:raise ValueError('validation view drift')
        return receipt
    if not partition.startswith('inner-'):raise ValueError('outer evaluation is not admitted here')
    dest.mkdir(parents=True)
    selected=set(membership['test']);offset=0;groups=[]
    for g in json.loads(PREP.read_text())['groups']:
        ids=np.arange(offset,offset+g['rows']);offset+=g['rows']
        local=np.array([i for i,v in enumerate(ids) if int(v) in selected],dtype=np.int64)
        if not len(local):continue
        if g['split']!='fit':raise ValueError('development entered inner validation')
        source=Path(g['path']);keys=source.with_suffix('.keys.parquet');manifest=Path(str(source)+'.manifest.json')
        for path,key in ((source,'sha256'),(keys,'keys_sha256'),(manifest,'manifest_sha256')):
            if sha(path)!=g[key]:raise ValueError('original view drift')
        out=dest/source.name;keyout=out.with_suffix('.keys.parquet')
        pq.write_table(pq.read_table(source).take(local),out,compression='zstd')
        selected_keys=pq.read_table(keys).take(local)
        pq.write_table(selected_keys,keyout,compression='zstd')
        meta=json.loads(manifest.read_text())
        meta.update(table_sha256=sha(out),keys_sha256=sha(keyout),rows_in=g['rows'],rows_kept=len(local),
                    source_table_sha256=g['sha256'],source_manifest_sha256=g['manifest_sha256'],
                    row_keys_sha256=row_keys_sha(selected_keys),row_selection_sha256=selection_sha(local),
                    shaping_evaluation=dict(partition=partition,role='inner_validation',indices=ids[local].tolist(),
                                            original_table_sha256=g['sha256'],plan_sha256=sha(root/'PLAN.json')))
        mout=Path(str(out)+'.manifest.json');mout.write_bytes(encoded(meta))
        groups.append(dict(name=g['name'],path=str(out),indices=ids[local].tolist(),
                           file_sha256={str(p):sha(p) for p in (out,keyout,mout)}))
    if {i for g in groups for i in g['indices']}!=selected:raise ValueError('validation coverage differs')
    receipt=dict(partition=partition,indices=membership['test'],groups=groups)
    (dest/'RECEIPT.json').write_bytes(encoded(receipt));return receipt


def probe_values(values,raw_ids,declaration):
    position={int(v):i for i,v in enumerate(raw_ids)}
    damage=np.array([position[i] for i in declaration['direct']])
    rng=np.random.default_rng(20261107)
    low=values[rng.integers(len(values),size=128)].copy();high=low.copy()
    median=np.median(values,axis=0).astype(np.float32)
    changed=rng.choice(damage,size=128)
    high[np.arange(128),changed]+=np.maximum(median[changed],np.float32(1e-6))*np.float32(.25)
    ray=np.repeat(median[None,:],9,axis=0)
    factors=np.array([0,1e-8,1e-6,1e-4,.01,.1,1,10,1e4],np.float32)
    ray[:,damage]*=factors[:,None]
    data=np.concatenate((low,high,ray))
    return data


def make_probes(root,partition,train,plan):
    from shaped_linear_study import OLD
    from tree_study import RAW_IDS
    raw_path=OLD/'data/raw.npy'
    if sha(raw_path)!=plan['input_pins'][str(raw_path)]:raise ValueError('probe input drift')
    raw=np.load(raw_path,mmap_mode='r')
    values=np.array(raw[np.asarray(train)],dtype=np.float32)
    declaration=json.loads((REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json').read_text())
    data=probe_values(values,RAW_IDS,declaration)
    position={int(v):i for i,v in enumerate(RAW_IDS)}
    path=root/'evaluation_views'/partition/'probes.parquet'
    fields={f'f{i}':pa.array(data[:,position[i]] if i in position else np.zeros(len(data),np.float32)) for i in range(1853)}
    fields.update(human_score=pa.array(np.zeros(len(data))),ref_basename=pa.array(['probe']*len(data)))
    if path.exists():raise ValueError('probe view already exists without a receipt')
    pq.write_table(pa.table(fields),path,compression='zstd')
    Path(str(path)+'.manifest.json').write_bytes(encoded(dict(formula_revision=7,scope='training-input feature probes only',rows=len(data))))
    receipt=dict(path=str(path),sha256=sha(path),training_indices_sha256=__import__('hashlib').sha256(encoded(train)).hexdigest(),
                 pairs=128,identity_row=256,ray_rows=9,changed_fields='damage only; reference fields held fixed')
    (path.parent/'PROBES.json').write_bytes(encoded(receipt));return receipt


def admitted_epoch(run,result,cells):
    cell=run['cell']
    if cell not in cells or cell['stage'] not in ('inner','fresh_control_inner'):raise ValueError('only registered inner cells are admitted')
    if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('training is not complete')
    epochs=120 if 'pretrained' in cell['family'] or cell['stage']=='fresh_control_inner' else 240
    if run['epochs']!=epochs:raise ValueError('incorrect scheduled checkpoint budget')
    return epochs


def evaluation_epoch(epochs,checkpoint):
    if checkpoint is None:return epochs-1
    allowed=(49,99,119,149,199,219,239)
    if checkpoint not in allowed or checkpoint>=epochs:
        raise ValueError('unregistered diagnostic checkpoint')
    return checkpoint


def reexport_bake(receipt_path,original,run,build_path):
    receipt=json.loads(receipt_path.read_text())
    if receipt['representation']!='research-relu-max-pairs-v1':raise ValueError('unsupported re-export')
    if receipt['original_sha256']!=sha(original):raise ValueError('re-export checkpoint drift')
    if receipt['training_build_sha256']!=run['build_sha256']:raise ValueError('re-export training build drift')
    if receipt['evaluation_build_sha256']!=sha(build_path):raise ValueError('re-export evaluator drift')
    for path,pin in receipt['file_sha256'].items():
        if sha(path)!=pin:raise ValueError('re-export artifact drift')
    bake=Path(receipt['bake'])
    if str(bake) not in receipt['file_sha256']:raise ValueError('unpinned re-export bake')
    return bake


def evaluate(args):
    plan,cells,partitions=load_registration(args.study)
    run=json.loads((args.run/'RUN.json').read_text());result=json.loads((args.run/'RESULT.json').read_text())
    if sha(args.study/'PLAN.json')!=run['plan_sha256']:raise ValueError('run registration drift')
    cell=run['cell']
    epochs=admitted_epoch(run,result,cells)
    final=args.run/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin'
    if sha(final)!=result['bake_sha256']:raise ValueError('final checkpoint drift')
    checkpoint=getattr(args,'checkpoint',None)
    epoch=evaluation_epoch(epochs,checkpoint)
    bake=args.run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin'
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    reexport=getattr(args,'reexport_receipt',None)
    if reexport is None:
        if sha(args.binary_dir/'BUILD.json')!=run['build_sha256']:raise ValueError('evaluator binary set differs')
    else:
        bake=reexport_bake(reexport,bake,run,args.binary_dir/'BUILD.json')
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('binary drift')
    os.environ['REV4_V2_BIN_DIR']=str(args.binary_dir);os.environ['ZEN_PANEL_BIN']=str(args.binary_dir/'panel')
    import v2_common
    import v2_lodo_mlp as owner
    from shaped_linear_study import OLD,PREP,metrics,weights_and_targets
    owner.FITBIN=v2_common.FITBIN=args.binary_dir/'bake_dial_refit'
    membership=partitions[cell['partition']]
    validation=prepare_validation(args.study,cell['partition'],membership)
    training=json.loads((args.study/'comparators'/cell['partition']/'GROUPS.json').read_text())
    dest=args.run/'evaluation' if checkpoint is None else args.run/'curve_evaluation'/f'epoch-{epoch:03}'
    if reexport is not None:
        dest=args.study/'reexport_max_pairs'/cell['id']/args.run.name/('final' if checkpoint is None else f'epoch-{epoch:03}')
    if dest.exists():raise ValueError('evaluation exists; preserve and inspect it')
    dest.mkdir(parents=True)
    wanted=np.array(sorted(set(membership['train'])|set(membership['test'])),dtype=np.int64)
    positions={int(v):i for i,v in enumerate(wanted)}
    pred=np.full(len(wanted),np.nan)
    # Rows in each source-subset view retain their original population order.
    offset=0;train_groups={g[0]:g[1] for g in training['groups']}
    selected=set(membership['train'])
    for g in json.loads(PREP.read_text())['groups']:
        ids=[v for v in range(offset,offset+g['rows']) if v in selected];offset+=g['rows']
        if not ids:continue
        values=owner.predict(bake,Path(train_groups[g['name']]),dest/(g['name']+'-train.tsv'))
        if len(values)!=len(ids):raise ValueError('training prediction coverage differs')
        pred[[positions[v] for v in ids]]=values
    for g in validation['groups']:
        values=owner.predict(bake,Path(g['path']),dest/(g['name']+'-validation.tsv'))
        if len(values)!=len(g['indices']):raise ValueError('validation prediction coverage differs')
        pred[[positions[v] for v in g['indices']]]=values
    if not np.isfinite(pred).all():raise ValueError('missing/nonfinite predictions')
    original=Path(plan['original_plan']).parent
    for path in [*(OLD/'data'/f'{k}.npy' for k in ('pop','y')),*(original/'data'/f'{k}.npy' for k in ('component','family'))]:
        if sha(path)!=plan['input_pins'][str(path)]:raise ValueError('evaluation array drift')
    d={k:np.load(OLD/'data'/f'{k}.npy',mmap_mode='r')[wanted] for k in ('pop','y')}
    d.update({k:np.load(original/'data'/f'{k}.npy',mmap_mode='r')[wanted] for k in ('component','family')})
    train=np.array([positions[v] for v in membership['train']]);test=np.array([positions[v] for v in membership['test']])
    _,_,maps=weights_and_targets(d['y'][train],d['pop'][train],d['component'][train],'range',1.)
    probe_receipt=args.study/'evaluation_views'/cell['partition']/'PROBES.json'
    probes=json.loads(probe_receipt.read_text()) if probe_receipt.exists() else make_probes(args.study,cell['partition'],membership['train'],plan)
    if sha(probes['path'])!=probes['sha256']:raise ValueError('probe drift')
    p=owner.predict(bake,Path(probes['path']),dest/'probes.tsv')
    delta=p[128:256]-p[:128]
    gate=dict(finite=bool(np.isfinite(p).all()),identity=float(p[256]),
              damage_increase_violations=int(np.sum(delta>1e-5)),maximum_damage_increase=float(max(0,delta.max())),
              identity_and_order_claim=cell['family']!='silu_diagnostic')
    if not gate['finite'] or (gate['identity_and_order_claim'] and (gate['identity']!=100. or gate['damage_increase_violations'])):
        (dest/'FAILED_GATE.json').write_bytes(encoded(gate));raise NativeFeatureGateError('native feature qualification failed')
    structure=json.loads(subprocess.check_output([str(args.binary_dir/'bake_dial_refit'),'inspect','--in',str(bake)],text=True))
    output=structure['layers'][-1]
    active=output['stored_weights']-output['zero_weights']
    report=dict(cell=cell,checkpoint_epoch=epoch,checkpoint_sha256=sha(bake),maps=maps,feature_gate=gate,
                structure=structure,active_output_coefficients=active,one_output_coefficient=active==1,
                train=metrics(pred,d['y'],train,d,maps),inner_validation=metrics(pred,d['y'],test,d,maps),
                selection='registered final epoch only' if checkpoint is None else 'diagnostic checkpoint; not a selection candidate',outer_results_read=False,
                driver_sha256=DRIVER_SHA256,plan_sha256=sha(args.study/'PLAN.json'))
    if reexport is not None:
        report.update(reexport_receipt=str(reexport),reexport_receipt_sha256=sha(reexport),
                      evaluation_build_sha256=sha(args.binary_dir/'BUILD.json'),
                      selection='supplemental re-export; requires uniform representation qualification before selection')
    np.savez_compressed(dest/'predictions.npz',original_indices=wanted,native=pred)
    (dest/'REPORT.json').write_bytes(encoded(report))
    (dest/('COMPLETE.json' if checkpoint is None else 'DIAGNOSTIC_COMPLETE.json')).write_bytes(encoded(dict(cell=cell['id'],report_sha256=sha(dest/'REPORT.json'))))
    print(json.dumps({'cell':cell['id'],'inner_selection_score':report['inner_validation']['selection_score']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int,help='registered diagnostic epoch; requires completed training')
    p.add_argument('--reexport-receipt',type=Path,help='pinned supplemental representation; preserves original reports')
    evaluate(p.parse_args())


if __name__=='__main__':main()
