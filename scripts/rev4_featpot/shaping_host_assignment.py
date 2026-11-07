"""Content-bound cell placement; never an exclusive compute lock."""
from __future__ import annotations
import argparse,json,socket
from datetime import datetime,timezone
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


def relocate_joint_subset(root,assignment,requested,host,queue_status,receipt,*,proc=Path('/proc')):
    """Split placement only after the old queue has stopped new admissions."""
    from shaping_execution import load_registration
    from shaping_rescue_dispatch import worker_claims
    _,cells,_=load_registration(root)
    if assignment.parent!=root/'worker_assignments':raise ValueError('assignment outside study')
    if receipt.exists():raise ValueError('preserve existing relocation receipt')
    with worker_claims(root) as claims:
        old=json.loads(assignment.read_text());old_pin=sha(assignment);selected=set(requested)
        if (not selected or len(selected)!=len(requested) or not selected<set(old['cells'])
                or host==old['host'] or old['host']!=socket.gethostname()):
            raise ValueError('distinct unstarted subset on the current host required')
        registered=[c for c in cells if c['id'] in old['cells']]
        if (len(registered)!=len(old['cells']) or any(c['stage']!='inner' or
                c['partition']!=old['partition'] or c['family'] not in
                ('product_joint_identity','separate_joint_identity') for c in registered)):
            raise ValueError('registered joint-shaping assignment required')
        q=json.loads(queue_status.read_text())
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(q['checked_utc'])).total_seconds()
        command=(proc/str(q['pid'])/'cmdline').read_bytes().decode()
        if (not 0<=age<=30 or not q.get('draining') or not Path(q['drain_file']).exists()
                or str(root) not in command or 'shaping_' not in command
                or set(q['required_cells'])!=set(old['cells']) or selected&set(q['active'])
                or q['plan_sha256']!=old['plan_sha256'] or q['build_sha256']!=old['build_sha256']
                or old['plan_sha256']!=sha(root/'PLAN.json')):
            raise ValueError('fresh live drained queue with matching placement required')
        if selected&set(claims) or any((root/'runs'/cell).exists() for cell in selected):
            raise ValueError('started or reserved cells cannot be relocated')
        check(root,registered,old['build_sha256'])
        backup=assignment.with_suffix('.json.bak')
        kept_path=assignment.with_name(assignment.stem+'-remaining.json')
        moved_path=assignment.with_name(assignment.stem+'-relocated-'+host+'.json')
        if any(p.exists() for p in (backup,kept_path,moved_path)):
            raise ValueError('preserve existing relocation artifacts')
        kept={**old,'cells':[c for c in old['cells'] if c not in selected]}
        moved={**old,'host':host,'cells':[c for c in old['cells'] if c in selected]}
        record=dict(stage='planned',checked_utc=datetime.now(timezone.utc).isoformat(),
            original_assignment=str(assignment),original_sha256=old_pin,backup=str(backup),
            kept_assignment=str(kept_path),moved_assignment=str(moved_path),
            kept=kept,moved=moved,queue_status_sha256=sha(queue_status),
            active_children_preserved=q['active'],scientific_inputs_unchanged=True,
            training_replayed=False,driver_sha256=sha(__file__))
        with receipt.open('xb') as stream:stream.write(encoded(record))
        if sha(assignment)!=old_pin:raise ValueError('assignment changed before relocation')
        assignment.rename(backup)
        for path,data in [(kept_path,kept),(moved_path,moved)]:
            with path.open('xb') as stream:stream.write(encoded(data))
        check(root,[c for c in registered if c['id'] not in selected],old['build_sha256'])
        check(root,[c for c in registered if c['id'] in selected],old['build_sha256'],host)
        if sha(backup)!=old_pin:raise ValueError('original placement preservation differs')
        record.update(stage='placement_split',kept_sha256=sha(kept_path),moved_sha256=sha(moved_path))
        receipt.write_bytes(encoded(record))
        return record


def main():
    from shaping_execution import load_registration
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--partition',required=True);p.add_argument('--host',required=True)
    p.add_argument('--family',choices=('joint_shapes','analytic','relu','pretrained_shapes'),default='joint_shapes')
    p.add_argument('--cell',action='append',help='explicit registered subset; never changes study requirements')
    p.add_argument('--stage',choices=('inner','fresh_control_inner'),default='inner')
    p.add_argument('--relocate-from',type=Path)
    p.add_argument('--queue-status',type=Path);p.add_argument('--relocation-receipt',type=Path)
    args=p.parse_args();_,cells,_=load_registration(args.study)
    if args.relocate_from:
        if not (args.cell and args.queue_status and args.relocation_receipt):
            p.error('relocation requires cell, queue-status and relocation-receipt')
        old=json.loads(args.relocate_from.read_text())
        if (args.family!='joint_shapes' or args.stage!='inner' or old['partition']!=args.partition
                or old['build_sha256']!=sha(args.binary_dir/'BUILD.json')):
            raise ValueError('relocation must preserve the original joint partition and binary')
        record=relocate_joint_subset(args.study,args.relocate_from,args.cell,args.host,
                                    args.queue_status,args.relocation_receipt)
        print(json.dumps(record),flush=True);return
    if args.queue_status or args.relocation_receipt:p.error('relocation evidence requires relocate-from')
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
