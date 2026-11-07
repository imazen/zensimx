"""Preserve and evaluate a separately pinned native max-pair representation."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_evaluation import admitted_epoch,evaluation_epoch,evaluate,outer_context
from shaping_execution import load_registration
from shaping_probe_trace import unique_signed_pairs


def reexport(args):
    _,cells,_=load_registration(args.study)
    run=json.loads((args.run/'RUN.json').read_text())
    if run['cell']['stage']=='exposure_extension':
        from shaping_extension import resolve
        cells=[*cells,resolve(args.study,run['cell']['id'])['cell']]
    result=json.loads((args.run/'RESULT.json').read_text())
    outer=outer_context(args.study,run,cells,args.checkpoint)
    epochs=admitted_epoch(run,result,cells,outer)
    epoch=evaluation_epoch(epochs,args.checkpoint)
    if run['cell']['family']!='relu':raise ValueError('this exporter qualifies first-layer ReLU pairs only')
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','run','binary-dir','inspector'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=int)
    reexport(p.parse_args())
