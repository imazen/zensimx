"""Content-bound cell placement; never an exclusive compute lock."""
from __future__ import annotations
import argparse,json,socket
from pathlib import Path
from shaping_completion import encoded,sha


def check(root,cells,build_pin,host=None):
    host=host or socket.gethostname();selected={c['id'] for c in cells};seen=set()
    for path in (root/'worker_assignments').glob('*.json'):
        record=json.loads(path.read_text());ids=set(record['cells'])
        if seen&ids:raise ValueError('overlapping worker assignments')
        seen|=ids
        if not selected&ids:continue
        if record['plan_sha256']!=sha(root/'PLAN.json') or record['build_sha256']!=build_pin:
            raise ValueError('worker assignment scientific identity differs')
        if record['host']!=host:raise ValueError('cell assigned to another worker')


def main():
    from shaping_execution import load_registration
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--partition',required=True);p.add_argument('--host',required=True)
    p.add_argument('--family',choices=('joint_shapes','analytic','relu','pretrained_shapes'),default='joint_shapes')
    p.add_argument('--cell',action='append',help='explicit registered subset; never changes study requirements')
    p.add_argument('--stage',choices=('inner','fresh_control_inner'),default='inner')
    args=p.parse_args();_,cells,_=load_registration(args.study)
    if args.stage=='fresh_control_inner' and args.family!='relu':raise ValueError('fresh control must be ReLU')
    families={'joint_shapes':('product_joint_identity','separate_joint_identity'),
              'analytic':('softplus','silu_monotone','silu_diagnostic'),'relu':('relu',),
              'pretrained_shapes':('product_pretrained_frozen','product_pretrained_joint','separate_pretrained_frozen','separate_pretrained_joint')}[args.family]
    selected=[c for c in cells if c['stage']==args.stage and c['partition']==args.partition
              and c['family'] in families]
    if len(selected)!=12*len(families):raise ValueError('complete registered family wave required')
    if args.cell:
        from shaping_execution import registered_subset
        selected=registered_subset(selected,args.cell)
    if any((args.study/'runs'/c['id']).exists() for c in selected):raise ValueError('existing attempts cannot be reassigned')
    build_pin=sha(args.binary_dir/'BUILD.json');check(args.study,selected,build_pin,args.host)
    directory=args.study/'worker_assignments';directory.mkdir(exist_ok=True)
    suffix='-fresh' if args.stage=='fresh_control_inner' else ''
    if args.cell:suffix+='-'+__import__('hashlib').sha256(encoded(sorted(args.cell))).hexdigest()[:16]
    out=directory/(args.partition+'-'+args.family.replace('_','-')+suffix+'.json')
    record=dict(host=args.host,cells=[c['id'] for c in selected],plan_sha256=sha(args.study/'PLAN.json'),
                build_sha256=build_pin,partition=args.partition,scientific_inputs_unchanged=True)
    if args.family=='pretrained_shapes':record['parent_inputs']=parent_inputs(args.study,cells,selected,build_pin)
    with out.open('xb') as stream:stream.write(encoded(record))
    print(json.dumps(record),flush=True)


def parent_inputs(study,cells,selected,build_pin):
    from shaping_pretraining import completed_parent
    files={}
    for child in selected:
        parent,run,result=completed_parent(study,child,cells)
        identity=json.loads((run/'RUN.json').read_text())
        if (identity['cell']!=parent or identity['plan_sha256']!=sha(study/'PLAN.json')
                or identity['build_sha256']!=build_pin or identity['epochs']!=240 or identity['pairs_per_epoch']!=50000):
            raise ValueError('parent scientific identity differs')
        if sha(run/'checkpoints/ckpt_epoch239.bin')!=result['bake_sha256']:raise ValueError('parent final checkpoint drift')
        for name in ('RUN.json','RESULT.json','SHAPING.json','checkpoints/ckpt_epoch119.bin','checkpoints/ckpt_epoch239.bin'):
            p=run/name
            if p.is_symlink() or p.resolve()!=p:raise ValueError('parent input symlink refused')
            files[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    return [files[k] for k in sorted(files)]


if __name__=='__main__':main()
