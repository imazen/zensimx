"""Content-bound cell placement; never an exclusive compute lock."""
from __future__ import annotations
import argparse,json,socket,fcntl,os,signal,time,ast,inspect,subprocess,shlex
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha


@contextmanager
def placement_claims(root):
    # Read the shared admission owner's claims under its transaction lock.
    # This also works beside older immutable worker source exports.
    directory=root/'cpu-admission';directory.mkdir(exist_ok=True)
    with (directory/'transaction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=directory/'CLAIMS.json'
        yield json.loads(path.read_text()) if path.exists() else {}


def process_identity(proc,pid):
    fields=(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return fields[0],fields[19]


def verify_legacy_guard(driver,guard,driver_pin,guard_pin):
    if sha(driver)!=driver_pin or sha(guard)!=guard_pin:raise ValueError('legacy source checksum differs')
    nodes={n.name:n for n in ast.parse(driver.read_text()).body if isinstance(n,ast.FunctionDef)}
    expected=ast.parse("def f():\n from shaping_host_assignment import check\n check(root,[cell],sha(binary_dir/'BUILD.json'))").body[0].body
    if ast.dump(ast.Module(body=nodes['_run_native'].body[:2],type_ignores=[]))!=ast.dump(ast.Module(body=expected,type_ignores=[])):
        raise ValueError('legacy host guard must precede attempt creation')
    nodes={n.name:n for n in ast.parse(guard.read_text()).body if isinstance(n,ast.FunctionDef)}
    current=ast.parse(inspect.getsource(check)).body[0]
    if ast.dump(nodes['check'])!=ast.dump(current):raise ValueError('legacy host-check behavior differs')


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


def assigned_placement(root,cell,build_pin):
    matches=[]
    for path in (root/'worker_assignments').glob('*.json'):
        record=json.loads(path.read_text())
        if cell['id'] in record['cells']:matches.append((path,record))
    if len(matches)>1:raise ValueError('ambiguous cell placement')
    if not matches:return None
    path,record=matches[0];check(root,[cell],build_pin,record['host'])
    return dict(host=record['host'],assignment=str(path),assignment_sha256=sha(path))


def relocate_joint_subset(root,assignment,requested,host,queue_status,receipt,*,proc=Path('/proc')):
    """Split placement only after the old queue has stopped new admissions."""
    from shaping_execution import load_registration
    _,cells,_=load_registration(root)
    if assignment.parent!=root/'worker_assignments':raise ValueError('assignment outside study')
    if receipt.exists():raise ValueError('preserve existing relocation receipt')
    with placement_claims(root) as claims:
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
        if q.get('control_mechanism')=='verified_legacy_sigstop':
            if process_identity(proc,q['pid'])[0]!='T':raise ValueError('legacy controller is not admission-stopped')
            verify_legacy_guard(Path(q['legacy_driver']),Path(q['legacy_guard']),q['legacy_driver_sha256'],q['legacy_guard_sha256'])
        elif q.get('control_mechanism') is not None:raise ValueError('unknown admission-stop mechanism')
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


def relocate_legacy_joint(root,assignment,requested,host,legacy_status,driver,guard,
                          driver_pin,guard_pin,receipt,control_output,*,proc=Path('/proc')):
    """Pause only a verified legacy queue writer during a placement transaction."""
    verify_legacy_guard(driver,guard,driver_pin,guard_pin)
    old=json.loads(assignment.read_text());status=json.loads(legacy_status.read_text());pid=status['controller_pid']
    command=(proc/str(pid)/'cmdline').read_bytes().decode()
    argv=command.split('\0') if '\0' in command else command.split()
    if '--workers' not in argv:raise ValueError('legacy worker budget must be explicit')
    workers=int(argv[argv.index('--workers')+1])
    if (str(root) not in command or 'shaping_execution.py' not in command
            or set(status['cells'])!=set(old['cells']) or status['stage']!='running'):
        raise ValueError('matching live legacy controller required')
    state,start=process_identity(proc,pid)
    if state in ('T','Z','X'):raise ValueError('legacy controller already stopped or terminal')
    control_output.mkdir(parents=True,exist_ok=False)
    (control_output/'ORIGINAL_STATUS.json').write_bytes(encoded(status))
    paused=False
    try:
        os.kill(pid,signal.SIGSTOP);paused=True
        deadline=time.monotonic()+5
        while process_identity(proc,pid)[0]!='T':
            if time.monotonic()>=deadline:raise ValueError('legacy controller did not stop admissions')
            time.sleep(.02)
        active={};pins={}
        for cell in old['cells']:
            for path in (root/'runs'/cell).glob('attempt-*/RUN.json'):
                run=json.loads(path.read_text())
                if (run['cell']['id']!=cell or run['driver_sha256']!=driver_pin or run['plan_sha256']!=old['plan_sha256']
                        or run['build_sha256']!=old['build_sha256'] or run['epochs']!=240
                        or run['pairs_per_epoch']!=50000):raise ValueError('legacy run identity differs')
                pins[str(path)]=sha(path)
                if path.with_name('RESULT.json').exists():continue
                child=(proc/str(run['pid'])/'cmdline').read_bytes().decode()
                if str(path.parent) not in child or 'run-heavy' not in child:
                    raise ValueError('legacy active child is not verified live')
                child_fields=(proc/str(run['pid'])/'stat').read_text().rsplit(')',1)[1].split()
                if int(child_fields[1])!=pid:raise ValueError('legacy active child has another controller')
                active[cell]=run['pid']
        # Every executor slot must be blocked on its own existing live child.
        # Otherwise an unrecorded task could already have passed the old guard.
        if len(active)!=workers:raise ValueError('all legacy worker slots must have verified live children')
        intent=control_output/'ADMISSION_STOP.json'
        intent.write_bytes(encoded(dict(pid=pid,start_ticks=start,mechanism='SIGSTOP of queue writer only',active_children_preserved=active)))
        snapshot=dict(pid=pid,checked_utc=datetime.now(timezone.utc).isoformat(),draining=True,
            drain_file=str(intent),active=active,required_cells=old['cells'],plan_sha256=old['plan_sha256'],
            build_sha256=old['build_sha256'],control_mechanism='verified_legacy_sigstop',
            legacy_driver=str(driver),legacy_guard=str(guard),legacy_driver_sha256=driver_pin,
            legacy_guard_sha256=guard_pin,run_identity_sha256=pins)
        path=control_output/'CONTROL_STATUS.json';path.write_bytes(encoded(snapshot))
        result=relocate_joint_subset(root,assignment,requested,host,path,receipt,proc=proc)
        result.update(legacy_controller_pid=pid,legacy_start_ticks=start,
            control_status=str(path),old_queue_keeps_original_requirement=True,
            expected_old_queue_refusal='reassigned cells must fail the frozen host guard before attempt creation')
        receipt.write_bytes(encoded(result))
        return result
    finally:
        if paused:
            if process_identity(proc,pid)[1]!=start:raise ValueError('legacy PID identity changed; refusing signal')
            os.kill(pid,signal.SIGCONT)
            (control_output/'RESUMED.json').write_bytes(encoded(dict(pid=pid,start_ticks=start,resumed_utc=datetime.now(timezone.utc).isoformat())))


def mirror_placement(root,assignment,source_record,files):
    """Verify every incoming placement before changing the coordinator registry."""
    from shaping_execution import load_registration
    _,cells,_=load_registration(root)
    old=json.loads(assignment.read_text());record=source_record
    if record['original_assignment']!=str(assignment) or record['original_sha256']!=sha(assignment):
        raise ValueError('coordinator/source original placement differs')
    backup=Path(record['backup']);kept=Path(record['kept_assignment']);moved=Path(record['moved_assignment'])
    if any(p.parent!=assignment.parent or p.exists() for p in (backup,kept,moved)):
        raise ValueError('placement mirror destinations must be absent in the assignment directory')
    expected={backup:record['original_sha256'],kept:record['kept_sha256'],moved:record['moved_sha256']}
    for destination,pin in expected.items():
        if sha(files/destination.name)!=pin:raise ValueError('incoming placement checksum differs')
    a=json.loads((files/kept.name).read_text());b=json.loads((files/moved.name).read_text())
    if (a!=record['kept'] or b!=record['moved'] or set(a['cells'])&set(b['cells'])
            or set(a['cells'])|set(b['cells'])!=set(old['cells'])
            or any(row[k]!=old[k] for row in (a,b) for k in ('plan_sha256','build_sha256','partition'))
            or a['host']!=old['host']):raise ValueError('incoming placement split differs')
    assignment.rename(backup)
    for path in (kept,moved):
        with path.open('xb') as stream:stream.write((files/path.name).read_bytes())
    for row in (a,b):check(root,[c for c in cells if c['id'] in row['cells']],old['build_sha256'],row['host'])
    return dict(backup=str(backup),kept=str(kept),moved=str(moved))


def coordinate_remote_legacy(root,assignment,requested,target,source_host,remote_tool,
        legacy_status,driver,guard,driver_pin,guard_pin,collector_status,output,*,proc=Path('/proc')):
    """Keep collection quiescent until source and coordinator placement agree."""
    original_pin=sha(assignment);old=json.loads(assignment.read_text())
    if old['host']!=source_host:raise ValueError('source host differs from assignment')
    status=json.loads(collector_status.read_text());pid=status['pid']
    command=(proc/str(pid)/'cmdline').read_bytes().decode()
    state,start=process_identity(proc,pid)
    children=proc/str(pid)/'task'/str(pid)/'children'
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(status['checked_utc'])).total_seconds()
    if (state in ('T','Z','X') or status['stage']!='watching' or not 0<=age<=120
            or 'shaping_worker_watch.py' not in command or source_host not in command
            or children.read_text().strip()):raise ValueError('fresh idle collector required')
    output.mkdir(parents=True,exist_ok=False);(output/'ORIGINAL_COLLECTOR.json').write_bytes(encoded(status))
    receipt=output/'SOURCE_RELOCATION.json';control=output/'source-control'
    remote_source=driver.parent
    loader='import sys,runpy;root,path=sys.argv[1:3];sys.path.insert(0,root);sys.argv=[path,*sys.argv[3:]];runpy.run_path(path,run_name="__main__")'
    command=['python3','-B','-c',loader,str(remote_source),str(remote_tool),'--study',str(root),
        '--binary-dir',str(root.parent/'study-bin'),'--partition',old['partition'],'--host',target,
        '--family','joint_shapes','--relocate-from',str(assignment),'--legacy-status',str(legacy_status),
        '--legacy-driver',str(driver),'--legacy-guard',str(guard),'--legacy-driver-pin',driver_pin,
        '--legacy-guard-pin',guard_pin,'--relocation-receipt',str(receipt),'--control-output',str(control)]
    for cell in requested:command+=['--cell',cell]
    remote_command='TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- '+shlex.join(command)
    (output/'COMMAND.json').write_bytes(encoded(dict(command=command,remote_command=remote_command,source_host=source_host,collector_pid=pid,collector_start_ticks=start)))
    paused=False;remote_started=False;safe_to_resume=False
    try:
        os.kill(pid,signal.SIGSTOP);paused=True;deadline=time.monotonic()+5
        while process_identity(proc,pid)[0]!='T':
            if time.monotonic()>=deadline:raise ValueError('collector did not stop')
            time.sleep(.02)
        if children.read_text().strip():raise ValueError('collector acquired an active child before maintenance')
        subprocess.run(['ssh',source_host,shlex.join(['mkdir',str(output)])],check=True,capture_output=True,text=True,timeout=30)
        remote_started=True
        result=subprocess.run(['ssh',source_host,remote_command],capture_output=True,text=True,timeout=60)
        (output/'REMOTE.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('source relocation refused; preserve remote log')
        record=json.loads(result.stdout)
        if record['original_sha256']!=original_pin or set(record['moved']['cells'])!=set(requested):
            raise ValueError('remote relocation identity differs')
        files=output/'SOURCE_FILES';files.mkdir()
        names=[str(receipt),record['backup'],record['kept_assignment'],record['moved_assignment'],
               str(control/'CONTROL_STATUS.json'),str(control/'RESUMED.json')]
        subprocess.run(['scp',*[source_host+':'+name for name in names],str(files)+'/'],check=True,capture_output=True,text=True,timeout=30)
        if json.loads((files/receipt.name).read_text())!=record:raise ValueError('remote receipt differs')
        mirrored=mirror_placement(root,assignment,record,files);safe_to_resume=True
        receipt.write_bytes(encoded(record))
        (output/'LOCAL_APPLIED.json').write_bytes(encoded(dict(stage='placement_mirrored',checked_utc=datetime.now(timezone.utc).isoformat(),paths=mirrored,source_receipt_sha256=sha(receipt))))
        return record
    except BaseException as error:
        (output/'FAILURE.json').write_bytes(encoded(dict(error=str(error),remote_started=remote_started,
            collector_left_stopped=paused and remote_started and not safe_to_resume,pid=pid,start_ticks=start)))
        raise
    finally:
        if paused and (safe_to_resume or not remote_started):
            if process_identity(proc,pid)[1]!=start:raise ValueError('collector PID identity changed; refusing signal')
            os.kill(pid,signal.SIGCONT)
            (output/'COLLECTOR_RESUMED.json').write_bytes(encoded(dict(pid=pid,start_ticks=start,checked_utc=datetime.now(timezone.utc).isoformat())))


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
    p.add_argument('--legacy-status',type=Path);p.add_argument('--legacy-driver',type=Path)
    p.add_argument('--legacy-guard',type=Path);p.add_argument('--legacy-driver-pin');p.add_argument('--legacy-guard-pin')
    p.add_argument('--control-output',type=Path)
    args=p.parse_args();_,cells,_=load_registration(args.study)
    if args.relocate_from:
        if not (args.cell and args.relocation_receipt and (args.queue_status or args.legacy_status)):
            p.error('relocation requires cell, relocation-receipt and queue-status or legacy-status')
        old=json.loads(args.relocate_from.read_text())
        if (args.family!='joint_shapes' or args.stage!='inner' or old['partition']!=args.partition
                or old['build_sha256']!=sha(args.binary_dir/'BUILD.json')):
            raise ValueError('relocation must preserve the original joint partition and binary')
        if args.legacy_status:
            if args.queue_status or not all((args.legacy_driver,args.legacy_guard,args.legacy_driver_pin,args.legacy_guard_pin,args.control_output)):
                p.error('legacy relocation requires source guards and control-output without queue-status')
            record=relocate_legacy_joint(args.study,args.relocate_from,args.cell,args.host,args.legacy_status,
                args.legacy_driver,args.legacy_guard,args.legacy_driver_pin,args.legacy_guard_pin,
                args.relocation_receipt,args.control_output)
        else:
            record=relocate_joint_subset(args.study,args.relocate_from,args.cell,args.host,
                                        args.queue_status,args.relocation_receipt)
        print(json.dumps(record),flush=True);return
    if any((args.queue_status,args.relocation_receipt,args.legacy_status,args.legacy_driver,
            args.legacy_guard,args.legacy_driver_pin,args.legacy_guard_pin,args.control_output)):
        p.error('relocation evidence requires relocate-from')
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
    with placement_claims(args.study) as claims:
        if set(c['id'] for c in selected)&set(claims):raise ValueError('reserved cells cannot be assigned')
        if any((args.study/'runs'/c['id']).exists() for c in selected):raise ValueError('existing attempts cannot be reassigned')
        build_pin=sha(args.binary_dir/'BUILD.json');check(args.study,selected,build_pin,args.host)
        if any(assigned_placement(args.study,c,build_pin) for c in selected):
            raise ValueError('reuse or split existing placement; never create overlapping assignments')
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
