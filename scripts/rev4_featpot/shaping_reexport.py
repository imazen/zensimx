"""Preserve and evaluate a separately pinned native max-pair representation."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from shaping_completion import MAX_PAIR_FAMILIES,max_pair_families,encoded,sha
from shaping_evaluation import admitted_epoch,evaluation_epoch,evaluate,outer_context
from shaping_execution import load_registration
from shaping_probe_trace import unique_signed_pairs


def admit_family(family,analytic_diagnostic=False,study=None):
    if family=='relu':return
    if family=='softplus' and analytic_diagnostic:return
    if study is not None and family not in MAX_PAIR_FAMILIES and family in max_pair_families(study):return
    raise ValueError('non-ReLU re-export requires explicit Softplus diagnostic admission')


def admit_representation(family,outer=None,analytic_diagnostic=False,study=None):
    if outer is None:
        admit_family(family,analytic_diagnostic,study)
    elif family not in (MAX_PAIR_FAMILIES if study is None else max_pair_families(study)) or outer['admission']['evaluation_representations'][family]['representation']!='research-relu-max-pairs-v1':
        raise ValueError('later max-pair representation was not admitted by inner selection')


def reexport(args):
    from shaping_reduced_scope import admit_compute
    admit_compute(args.study)
    _,cells,_=load_registration(args.study)
    run=json.loads((args.run/'RUN.json').read_text())
    if run['cell']['stage']=='exposure_extension':
        from shaping_extension import resolve
        cells=[*cells,resolve(args.study,run['cell']['id'])['cell']]
    result=json.loads((args.run/'RESULT.json').read_text())
    outer=outer_context(args.study,run,cells,args.checkpoint)
    epochs=admitted_epoch(run,result,cells,outer)
    epoch=evaluation_epoch(epochs,args.checkpoint)
    family=run['cell']['family'] if outer is None else outer.get('resolved_family',outer.get('variant',{}).get('resolved_family',run['cell']['family']))
    admit_representation(family,outer,getattr(args,'analytic_diagnostic',False),args.study)
    if sha(args.study/'PLAN.json')!=run['plan_sha256']:raise ValueError('registration drift')
    build_path=args.binary_dir/'BUILD.json'
    for name,pin in json.loads(build_path.read_text())['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('binary drift')
    original=args.run/run.get('fit_directory','.')/'checkpoints'/f'ckpt_epoch{epoch:03}.bin'
    dest=args.study/'reexports'/run['cell']['id']/args.run.name/('final' if args.checkpoint is None else f'epoch-{epoch:03}')
    if dest.exists():raise ValueError('preserve existing re-export')
    dest.mkdir(parents=True)
    inspection=subprocess.run([str(args.inspector),str(original),'--weights'],check=True,capture_output=True,text=True)
    model=dest/'original-inspection.json';model.write_text(inspection.stdout)
    (dest/'inspection.log').write_text(inspection.stderr)
    pairs=unique_signed_pairs(json.loads(inspection.stdout))
    declaration=dest/'pairs.txt'
    declaration.write_text('relu-max-pairs-v1\n'+''.join(f'{a} {b}\n' for a,b in pairs))
    bake=dest/'repaired.bin'
    native=subprocess.run([str(args.binary_dir/'bake_dial_refit'),'repair-research-max-pairs',
        '--in',str(original),'--out',str(bake),'--pairs',str(declaration)],capture_output=True,text=True)
    (dest/'repair.log').write_text(native.stdout+native.stderr)
    native.check_returncode()
    receipt=dict(representation='research-relu-max-pairs-v1',original_sha256=sha(original),
        training_build_sha256=run['build_sha256'],evaluation_build_sha256=sha(build_path),
        inspector_sha256=sha(args.inspector),bake=str(bake),pairs=pairs,
        driver_sha256=sha(__file__),file_sha256={str(x):sha(x) for x in (model,declaration,bake,dest/'repair.log')})
    args.reexport_receipt=dest/'RECEIPT.json';args.reexport_receipt.write_bytes(encoded(receipt))
    evaluate(args)


def prediction_values(path):
    import numpy as np
    import pandas as pd
    frame=pd.read_csv(path,sep='\t')
    if frame.columns.tolist()!=['row_idx','pred'] or not np.array_equal(frame.row_idx.to_numpy(),np.arange(len(frame))):
        raise ValueError('comparison prediction row order differs')
    values=frame.pred.to_numpy(dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():raise ValueError('comparison predictions empty or nonfinite')
    return values


def compare_runtime(args,output):
    """Compare both runtimes and representations without changing study reports."""
    import numpy as np
    import v2_common
    import v2_lodo_mlp as owner
    from shaping_evaluation import prepare_validation,reexport_bake
    if output.exists():raise ValueError('preserve existing runtime comparison')
    plan,cells,partitions=load_registration(args.study)
    run=json.loads((args.run/'RUN.json').read_text());result=json.loads((args.run/'RESULT.json').read_text());cell=run['cell']
    if cell['stage']!='inner' or cell not in cells:raise ValueError('registered inner comparison required')
    if run['plan_sha256']!=sha(args.study/'PLAN.json'):raise ValueError('comparison plan drift')
    epochs=admitted_epoch(run,result,cells,None);epoch=evaluation_epoch(epochs,args.checkpoint)
    original=args.run/run.get('fit_directory','.')/'checkpoints'/f'ckpt_epoch{epoch:03}.bin'
    name='final' if args.checkpoint is None else f'epoch-{epoch:03}'
    receipt=(getattr(args,'diagnostic_receipt',None) or
             args.study/'reexports'/cell['id']/args.run.name/name/'RECEIPT.json')
    build=args.binary_dir/'BUILD.json'
    for key,pin in json.loads(build.read_text())['binary_sha256'].items():
        if sha(args.binary_dir/key)!=pin:raise ValueError('comparison binary drift')
    repaired=reexport_bake(receipt,original,run,build)
    original_build=args.original_binary_dir/'BUILD.json'
    if sha(original_build)!=run['build_sha256']:raise ValueError('original runtime differs from training')
    for key,pin in json.loads(original_build.read_text())['binary_sha256'].items():
        if sha(args.original_binary_dir/key)!=pin:raise ValueError('original runtime binary drift')
    training_path=args.study/'comparators'/cell['partition']/'GROUPS.json'
    training=json.loads(training_path.read_text())
    validation=prepare_validation(args.study,cell['partition'],partitions[cell['partition']])
    tables=[(g[0],'train',Path(g[1])) for g in training['groups']]+[(g['name'],'validation',Path(g['path'])) for g in validation['groups']]
    baseline=args.run/('evaluation' if args.checkpoint is None else 'curve_evaluation/'+name)
    pins={str(p):sha(p) for p in [args.run/'RUN.json',args.run/'RESULT.json',receipt,build,original_build,training_path,original,repaired]}
    # Refuse missing original evidence before any scoring.
    for group,split,table in tables:
        path=baseline/(group+'-'+split+'.tsv');prediction_values(path)
        pins[str(path)]=sha(path);pins[str(table)]=sha(table)
    output.mkdir(parents=True);owner.FITBIN=v2_common.FITBIN=args.binary_dir/'bake_dial_refit';rows=[]
    for group,split,table in tables:
        directory=output/(group+'-'+split);directory.mkdir()
        stored=prediction_values(baseline/(group+'-'+split+'.tsv'))
        owner.FITBIN=v2_common.FITBIN=args.original_binary_dir/'bake_dial_refit'
        old=owner.predict(original,table,directory/'original-runtime.tsv')
        if not np.array_equal(stored.astype(np.float32).view(np.uint32),old.astype(np.float32).view(np.uint32)):
            raise ValueError('original runtime no longer reproduces stored predictions')
        owner.FITBIN=v2_common.FITBIN=args.binary_dir/'bake_dial_refit'
        unmarked=owner.predict(original,table,directory/'unmarked.tsv')
        changed=owner.predict(repaired,table,directory/'repaired.tsv')
        if len(old)!=len(unmarked) or len(old)!=len(changed):raise ValueError('comparison row coverage differs')
        exact=np.array_equal(old.astype(np.float32).view(np.uint32),unmarked.astype(np.float32).view(np.uint32))
        delta=np.abs(changed-old)
        rows.append(dict(group=group,split=split,rows=len(old),unmarked_exact=bool(exact),max_abs_delta=float(delta.max()),mean_abs_delta=float(delta.mean())))
    for path,pin in pins.items():
        if sha(path)!=pin:raise ValueError('comparison evidence changed during scoring')
    record=dict(cell=cell,checkpoint_epoch=epoch,comparisons=rows,all_unmarked_exact=all(r['unmarked_exact'] for r in rows),evidence_sha256=pins,driver_sha256=sha(__file__),outer_results_read=False,selection_candidate=False)
    (output/'RESULT.json').write_bytes(encoded(record))
    if not record['all_unmarked_exact']:raise ValueError('unmarked runtime predictions changed')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','run','binary-dir','inspector'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=int)
    p.add_argument('--analytic-diagnostic',action='store_true',help='allow supplemental Softplus panels; selection remains unsupported')
    p.add_argument('--compare-runtime',type=Path)
    p.add_argument('--original-binary-dir',type=Path)
    p.add_argument('--diagnostic-receipt',type=Path,help='compare a separately pinned diagnostic conversion without admitting it to selection')
    args=p.parse_args()
    if args.diagnostic_receipt is not None and args.compare_runtime is None:
        p.error('diagnostic-receipt is valid only with compare-runtime')
    if args.compare_runtime:
        if args.original_binary_dir is None:p.error('runtime comparison requires original-binary-dir')
        compare_runtime(args,args.compare_runtime)
    else:reexport(args)
