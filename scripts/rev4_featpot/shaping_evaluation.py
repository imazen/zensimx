#!/usr/bin/env python3
"""Native evaluation with explicit, revalidated outer admission."""
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
VARIANT_STAGES=('source_curve','row_curve','shuffle','human_only')

class NativeFeatureGateError(ValueError):
    """Measured feature-contract failure; never a qualified candidate."""


def validation_admission(root,partition):
    if partition.startswith('inner-'):return None
    if partition not in ('outer-0','outer-1','outer-2'):
        raise ValueError('unregistered evaluation partition')
    from shaping_outer_admission import resolve
    _,path=resolve(root,int(partition.split('-')[1]))
    return sha(path)


def prepare_validation(root,partition,membership):
    # Gate before reading even an existing cached outer view.
    admission=validation_admission(root,partition)
    from shaped_linear_study import PREP
    from v2_teacher import row_keys_sha,selection_sha
    dest=root/'evaluation_views'/partition
    if dest.exists():
        receipt=json.loads((dest/'RECEIPT.json').read_text())
        if receipt.get('admission_sha256')!=admission:raise ValueError('validation admission drift')
        if receipt['indices']!=membership['test']:raise ValueError('validation membership drift')
        for g in receipt['groups']:
            for path,pin in g['file_sha256'].items():
                if sha(path)!=pin:raise ValueError('validation view drift')
        return receipt
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
                    shaping_evaluation=dict(partition=partition,role='outer_test' if admission else 'inner_validation',indices=ids[local].tolist(),
                                            original_table_sha256=g['sha256'],plan_sha256=sha(root/'PLAN.json')))
        mout=Path(str(out)+'.manifest.json');mout.write_bytes(encoded(meta))
        groups.append(dict(name=g['name'],path=str(out),indices=ids[local].tolist(),
                           file_sha256={str(p):sha(p) for p in (out,keyout,mout)}))
    if {i for g in groups for i in g['indices']}!=selected:raise ValueError('validation coverage differs')
    receipt=dict(partition=partition,indices=membership['test'],groups=groups)
    if admission:receipt['admission_sha256']=admission
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
    path.parent.mkdir(parents=True,exist_ok=True)
    fields={f'f{i}':pa.array(data[:,position[i]] if i in position else np.zeros(len(data),np.float32)) for i in range(1853)}
    fields.update(human_score=pa.array(np.zeros(len(data))),ref_basename=pa.array(['probe']*len(data)))
    if path.exists():raise ValueError('probe view already exists without a receipt')
    pq.write_table(pa.table(fields),path,compression='zstd')
    Path(str(path)+'.manifest.json').write_bytes(encoded(dict(formula_revision=7,scope='training-input feature probes only',rows=len(data))))
    receipt=dict(path=str(path),sha256=sha(path),training_indices_sha256=__import__('hashlib').sha256(encoded(train)).hexdigest(),
                 pairs=128,identity_row=256,ray_rows=9,changed_fields='damage only; reference fields held fixed')
    (path.parent/'PROBES.json').write_bytes(encoded(receipt));return receipt


def outer_context(root,run,cells,checkpoint=None):
    if run['cell']['stage']=='full_fit_selection':
        if checkpoint is not None:raise ValueError('full-FIT validation cannot diagnose checkpoints')
        from shaping_full_fit_execution import search_cell,prepared
        _,_,_,admission,path,cell,family,schedule=search_cell(root,run['cell']['id'])
        if (cell not in cells or run['cell']!=cell or run['resolved_family']!=family
            or run['schedule']!=schedule or run['admission_sha256']!=sha(path)
            or run['selection_scope']!='full_fit_only' or run['development_results_read']):
            raise ValueError('full-FIT run scope or admission drift')
        prepared(root,cell['id'])
        for name,pin in run['file_sha256'].items():
            if sha(name)!=pin:raise ValueError('full-FIT run input drift')
        return dict(admission=admission,schedule=schedule,admission_sha256=sha(path),
                    full_fit=True,resolved_family=family)
    if run['cell']['stage'] not in ('outer','fresh_control_outer','fresh_control_outer_matched',*VARIANT_STAGES):return None
    if checkpoint is not None:raise ValueError('outer responses cannot diagnose checkpoints')
    from shaping_outer_admission import resolve
    from shaping_outer_execution import resolve_cell,prepared
    partition=run['cell']['partition']
    if partition not in ('outer-0','outer-1','outer-2'):raise ValueError('unregistered outer partition')
    outer=int(partition.split('-')[1]);admission,path=resolve(root,outer)
    if run['cell'] not in [*cells,*admission['additional_cells']]:raise ValueError('unregistered outer cell')
    if run['cell']['stage'] in VARIANT_STAGES:
        from shaping_variant_execution import resolve as resolve_variant
        from shaping_variant_inputs import load_variant
        resolution=resolve_variant(root,run['cell']['id'])
        if run['resolution']!=resolution:raise ValueError('variant fit resolution drift')
        for name,pin in run['qualification_sha256'].items():
            if sha(name)!=pin:raise ValueError('variant native qualification drift')
        _,_,_,membership=load_variant(root,run['cell']['id'])
        return dict(admission=admission,schedule=resolution['schedule'],admission_sha256=sha(path),
                    variant=resolution,membership=membership)
    schedule=resolve_cell(run['cell'],admission)
    if run['admission_sha256']!=sha(path) or run['schedule']!=schedule:raise ValueError('outer run admission drift')
    prepared(root,outer,path)
    return dict(admission=admission,schedule=schedule,admission_sha256=sha(path))


def admitted_epoch(run,result,cells,outer=None):
    cell=run['cell']
    if outer is not None:
        if cell['stage'] not in ('outer','fresh_control_outer','fresh_control_outer_matched','full_fit_selection',*VARIANT_STAGES):
            raise ValueError('outer context on an inner run')
        epochs=outer['schedule']['phase_epochs'][-1]
        if result['stage']!='trained' or result['exit_code']!=0 or run['epochs']!=epochs:
            raise ValueError('outer training is not complete at its scheduled epoch')
        return epochs
    if cell not in cells or cell['stage'] not in ('inner','fresh_control_inner','rescue','exposure_extension'):raise ValueError('only registered inner cells are admitted')
    if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('training is not complete')
    epochs=120 if 'pretrained' in cell['family'] or cell['stage']=='fresh_control_inner' else 240
    if cell['stage']=='exposure_extension':epochs=cell['phase_epochs'][-1]
    if run['epochs']!=epochs:raise ValueError('incorrect scheduled checkpoint budget')
    return epochs


def training_targets(labels,positions,ids,table_path):
    """Use exactly the audited trainer targets; leave every outer target intact."""
    values=pq.read_table(table_path,columns=['human_score'])['human_score'].to_numpy()
    if len(values)!=len(ids) or not np.isfinite(values).all():raise ValueError('variant training target coverage differs')
    labels[[positions[v] for v in ids]]=values


def diagnostic_epochs(epochs):
    if epochs not in (120,240,480):raise ValueError('unregistered phase duration')
    return tuple(e for e in (49,99,119,149,199,219,239,299,349,399,429,449,459,479) if e<epochs)


def evaluation_epoch(epochs,checkpoint):
    if checkpoint is None:return epochs-1
    allowed=diagnostic_epochs(epochs)
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


def apply_report_scope(report,context,family):
    if context is None:return
    full_fit=context.get('full_fit',False)
    role='full_fit_validation' if full_fit else 'outer_test'
    report[role]=report.pop('inner_validation')
    report.update(outer_results_read=True,admission_sha256=context['admission_sha256'],
                  selection='full-FIT tuning only; excluded from nested-study selection' if full_fit else 'outer test; excluded from all selection')
    if full_fit:
        report.update(selection_scope='full_fit_only',resolved_family=family,development_results_read=False)


def evaluate(args):
    plan,cells,partitions=load_registration(args.study)
    run=json.loads((args.run/'RUN.json').read_text());result=json.loads((args.run/'RESULT.json').read_text())
    if run['cell']['stage']=='exposure_extension':
        from shaping_extension import resolve as resolve_extension
        resolution=resolve_extension(args.study,run['cell']['id'])
        if run['resolution']!=resolution:raise ValueError('extension registration/selection drift')
        cells=[*cells,resolution['cell']]
    if sha(args.study/'PLAN.json')!=run['plan_sha256']:raise ValueError('run registration drift')
    cell=run['cell']
    checkpoint=getattr(args,'checkpoint',None)
    outer=outer_context(args.study,run,cells,checkpoint)
    variant=outer.get('variant') if outer is not None else None
    family=(outer['resolved_family'] if outer and outer.get('full_fit') else variant['resolved_family'] if variant else cell['family'])
    epochs=admitted_epoch(run,result,cells,outer)
    if cell['stage'] in ('rescue','exposure_extension') or outer is not None:
        if cell['stage']=='rescue':
            from shaping_rescue import resolve
            if run['resolution']!=resolve(args.study,cell['id']):raise ValueError('rescue selection/trigger drift')
        exposure=cell['total_pair_draws'] if cell['stage']=='exposure_extension' else 12000000
        if outer is not None:exposure=outer['schedule']['total_pair_draws']
        if run['training_exposure_pair_draws']!=exposure or run['fit_directory']!='fit':raise ValueError('rescue/extension exposure/layout differs')
        phases=result['phases']
        expected=[120,120] if '_pretrained_' in cell['family'] else [240]
        if cell['stage']=='exposure_extension':expected=cell['phase_epochs']
        if outer is not None:expected=outer['schedule']['phase_epochs']
        if [p['epochs'] for p in phases]!=expected or any(p['exit_code']!=0 or p['stage']!='trained' for p in phases):
            raise ValueError('incomplete registered rescue phases')
        phase_names=['pretraining','fit'] if len(phases)==2 else ['fit']
        for name,phase in zip(phase_names,phases):
            directory=args.run/name
            if json.loads((directory/'RESULT.json').read_text())!=phase:raise ValueError('rescue phase receipt drift')
            if phase['pairs_per_epoch']!=50000:raise ValueError('rescue phase exposure differs')
            if sha(directory/'checkpoints'/f"ckpt_epoch{phase['epochs']-1:03}.bin")!=phase['bake_sha256']:
                raise ValueError('rescue phase model drift')
            for path,pin in phase['manifest_sha256'].items():
                if sha(path)!=pin:raise ValueError('rescue phase manifest drift')
    fit_root=args.run/run.get('fit_directory','.')
    final=fit_root/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin'
    if sha(final)!=result['bake_sha256']:raise ValueError('final checkpoint drift')
    epoch=evaluation_epoch(epochs,checkpoint)
    bake=fit_root/'checkpoints'/f'ckpt_epoch{epoch:03}.bin'
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    reexport=getattr(args,'reexport_receipt',None)
    if outer is not None:
        identity=outer['admission']['evaluation_representations'][family]
        representation='research-relu-max-pairs-v1' if reexport is not None else 'original'
        if representation!=identity['representation']:raise ValueError('outer evaluation representation differs from inner selection')
        if representation!='original' and sha(args.binary_dir/'BUILD.json')!=identity['evaluation_build_sha256']:
            raise ValueError('outer repaired runtime differs from inner selection')
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
    membership=outer['membership'] if variant else partitions[cell['partition']]
    validation=prepare_validation(args.study,cell['partition'],membership)
    training_path=Path(variant['groups_path']) if variant else args.study/'comparators'/cell['partition']/'GROUPS.json'
    training=json.loads(training_path.read_text())
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
    if variant:
        offset=0
        for g in json.loads(PREP.read_text())['groups']:
            ids=[v for v in range(offset,offset+g['rows']) if v in selected];offset+=g['rows']
            if ids:training_targets(d['y'],positions,ids,train_groups[g['name']])
    train=np.array([positions[v] for v in membership['train']]);test=np.array([positions[v] for v in membership['test']])
    _,_,maps=weights_and_targets(d['y'][train],d['pop'][train],d['component'][train],'range',1.)
    probe_partition='variant-'+variant['view_identifier'][:20] if variant else cell['partition']
    probe_receipt=args.study/'evaluation_views'/probe_partition/'PROBES.json'
    probes=json.loads(probe_receipt.read_text()) if probe_receipt.exists() else make_probes(args.study,probe_partition,membership['train'],plan)
    if probes['training_indices_sha256']!=__import__('hashlib').sha256(encoded(membership['train'])).hexdigest():
        raise ValueError('feature probe training membership drift')
    if sha(probes['path'])!=probes['sha256']:raise ValueError('probe drift')
    p=owner.predict(bake,Path(probes['path']),dest/'probes.tsv')
    delta=p[128:256]-p[:128]
    gate=dict(finite=bool(np.isfinite(p).all()),identity=float(p[256]),
              damage_increase_violations=int(np.sum(delta>1e-5)),maximum_damage_increase=float(max(0,delta.max())),
              identity_and_order_claim=family!='silu_diagnostic')
    if not gate['finite'] or (gate['identity_and_order_claim'] and (gate['identity']!=100. or gate['damage_increase_violations'])):
        (dest/'FAILED_GATE.json').write_bytes(encoded(gate));raise NativeFeatureGateError('native feature qualification failed')
    structure=json.loads(subprocess.check_output([str(args.binary_dir/'bake_dial_refit'),'inspect','--in',str(bake)],text=True))
    output=structure['layers'][-1]
    active=output['stored_weights']-output['zero_weights']
    report=dict(cell=cell,checkpoint_epoch=epoch,checkpoint_sha256=sha(bake),maps=maps,feature_gate=gate,
                structure=structure,active_output_coefficients=active,one_output_coefficient=active==1,
                train=metrics(pred,d['y'],train,d,maps,allow_absent_maps=bool(variant)),
                inner_validation=metrics(pred,d['y'],test,d,maps,allow_absent_maps=bool(variant)),
                selection='registered final epoch only' if checkpoint is None else 'diagnostic checkpoint; not a selection candidate',outer_results_read=False,
                driver_sha256=DRIVER_SHA256,plan_sha256=sha(args.study/'PLAN.json'))
    apply_report_scope(report,outer,family)
    if variant:
        report.update(resolved_family=family,view_identifier=variant['view_identifier'],
                      training_target_policy='registered permutation' if cell['stage']=='shuffle' else 'actual targets',
                      outer_target_policy='actual targets',training_groups_sha256=sha(training_path))
    if reexport is not None:
        report.update(reexport_receipt=str(reexport),reexport_receipt_sha256=sha(reexport),
                      evaluation_build_sha256=sha(args.binary_dir/'BUILD.json'),
                      representation='research-relu-max-pairs-v1',
                      selection=report['selection'] if outer is not None else 'supplemental re-export; requires uniform representation qualification before selection')
    np.savez_compressed(dest/'predictions.npz',original_indices=wanted,native=pred)
    (dest/'REPORT.json').write_bytes(encoded(report))
    (dest/('COMPLETE.json' if checkpoint is None else 'DIAGNOSTIC_COMPLETE.json')).write_bytes(encoded(dict(cell=cell['id'],report_sha256=sha(dest/'REPORT.json'))))
    role='full_fit_validation' if outer and outer.get('full_fit') else 'outer_test' if outer is not None else 'inner_validation'
    score_key='full_fit_selection_score' if role=='full_fit_validation' else 'outer_test_score' if outer is not None else 'inner_selection_score'
    print(json.dumps({'cell':cell['id'],score_key:report[role]['selection_score']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int,help='registered diagnostic epoch; requires completed training')
    p.add_argument('--reexport-receipt',type=Path,help='pinned supplemental representation; preserves original reports')
    evaluate(p.parse_args())


if __name__=='__main__':main()
