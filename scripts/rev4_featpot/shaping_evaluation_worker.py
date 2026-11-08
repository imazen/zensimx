"""Evaluate collected native fits through their stage's original evidence owner."""
from __future__ import annotations
import argparse,fcntl,json,os,shlex,socket,subprocess,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from shaping_completion import encoded,sha

HOSTS=('r5600g','i134','r3500','r3800x','r7900x','i270')
DERIVED=('evaluation','curve_evaluation','CONVERGENCE.json','CONVERGENCE_FAILURE.json','EVALUATION_FAILURES.json','LATER_EVALUATION_FAILURE.json')
LATER_STAGES=('outer','fresh_control_outer','fresh_control_outer_matched','source_curve','row_curve','shuffle','human_only')


def later_scope(study,identity,cells):
    from shaping_evaluation import outer_context
    context=outer_context(study,identity,cells)
    if context is None:raise ValueError('later evaluation requires original outer admission')
    family=context.get('variant',{}).get('resolved_family',identity['cell']['family'])
    return dict(admission_sha256=context['admission_sha256'],schedule=context['schedule'],resolved_family=family,
                representation=context['admission']['evaluation_representations'][family])


def inventory(root):
    from shaping_worker_collect import inventory as original
    return original(root)


def components(cell,attempt):
    base=Path('runs')/cell/attempt
    return [*(base/name for name in DERIVED),
            Path('reexports')/cell/attempt,Path('reexport_max_pairs')/cell/attempt]


def request(study,run,*,target_host=None):
    """Metadata-only admission; never replay previously evaluated fits."""
    from shaping_execution import load_registration
    _,cells,_=load_registration(study)
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    cell=identity['cell'];collection=run/'WORKER_COLLECTION.json'
    later=cell['stage'] in LATER_STAGES
    if ((cell not in cells and not later) or cell['stage'] not in ('inner','fresh_control_inner','rescue',*LATER_STAGES)
        or identity.get('execution_host') not in HOSTS or not collection.exists()):
        raise ValueError('collected primary fit on an approved worker required')
    if result['cell']!=cell or result['stage']!='trained' or result['exit_code']!=0:
        raise ValueError('terminal successful worker fit required')
    if any((study/p).exists() for p in components(cell['id'],run.name)):
        raise ValueError('existing evaluation requires explicit reconciliation; never replay')
    epochs=result['epochs']
    from shaping_evaluation import diagnostic_epochs
    model_root=run/identity.get('fit_directory','.')/'checkpoints'
    scope=later_scope(study,identity,cells) if later else None
    epochs_to_pin={epochs-1} if later else {epochs-1,*diagnostic_epochs(epochs)}
    pins={str(p.relative_to(study)):sha(p) for p in
          [run/'RUN.json',run/'RESULT.json',run/identity.get('fit_directory','.')/'train.log',
           *(model_root/f'ckpt_epoch{e:03}.bin' for e in sorted(epochs_to_pin))]}
    if sha(model_root/f'ckpt_epoch{epochs-1:03}.bin')!=result['bake_sha256']:raise ValueError('final model drift')
    host=target_host or identity['execution_host']
    if host not in HOSTS:raise ValueError('approved evaluation host required')
    moving=host!=identity['execution_host']
    if moving and (cell['family']!='relu' or cell['stage'] not in ('inner','fresh_control_inner')):
        raise ValueError('cross-host transport currently supports primary ReLU controls only')
    if moving:
        for name in ('keep.txt','COLLECTION_DERIVED_INPUTS.json','WORKER_COLLECTION.json'):
            path=run/name
            if path.exists():pins[str(path.relative_to(study))]=sha(path)
    value=dict(cell=cell,attempt=run.name,host=host,
                plan_sha256=sha(study/'PLAN.json'),source_sha256=pins,
                reduced_scope_sha256=sha(study/'REDUCED_SCOPE.json'),
                collection_receipt_sha256=sha(collection),outer_results_read=False)
    if later:value['later_scope']=scope
    if moving:value.update(fit_execution_host=identity['execution_host'],
                           input_sizes={name:(study/name).stat().st_size for name in pins})
    return value


def verify_request(study,record,*,allow_missing=False):
    from shaping_reduced_scope import admit_compute
    admit_compute(study)
    if sha(study/'REDUCED_SCOPE.json')!=record['reduced_scope_sha256']:
        raise ValueError('worker reduced scope differs')
    if socket.gethostname()!=record['host'] or record['host'] not in HOSTS:
        raise ValueError('worker host differs')
    if record['outer_results_read'] or record['plan_sha256']!=sha(study/'PLAN.json'):
        raise ValueError('worker evaluation scope differs')
    if (record['cell']['stage'] in LATER_STAGES)!=('later_scope' in record):
        raise ValueError('worker evaluation stage scope differs')
    if 'fit_execution_host' in record:
        if (record['fit_execution_host'] not in HOSTS or record['fit_execution_host']==record['host']
            or record['cell']['family']!='relu' or record['cell']['stage'] not in ('inner','fresh_control_inner')
            or set(record['input_sizes'])!=set(record['source_sha256'])
            or len(record['input_sizes'])>1000
            or any(type(n) is not int or n<0 for n in record['input_sizes'].values())
            or sum(record['input_sizes'].values())>128*1024**2):raise ValueError('cross-host input scope differs')
    from pathlib import PurePosixPath
    for name,pin in record['source_sha256'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.parts[:3]!=('runs',record['cell']['id'],record['attempt']):
            raise ValueError('worker evaluation input path differs')
        source=study/name
        if any(p.is_symlink() for p in [source,*source.parents]):raise ValueError('symlink in evaluation input')
        if allow_missing and not source.exists():continue
        if sha(source)!=pin:raise ValueError('worker evaluation input drift')
        if 'input_sizes' in record and source.stat().st_size!=record['input_sizes'][name]:raise ValueError('evaluation input size differs')
    original=study/'runs'/record['cell']['id']/record['attempt']/'RUN.json'
    if 'fit_execution_host' in record and original.exists():
        if json.loads(original.read_text())['execution_host']!=record['fit_execution_host']:
            raise ValueError('training origin differs')
    if 'later_scope' in record:
        from shaping_execution import load_registration
        _,cells,_=load_registration(study)
        identity=json.loads(original.read_text())
        if identity['execution_host']!=record['host'] or later_scope(study,identity,cells)!=record['later_scope']:
            raise ValueError('later evaluation original admission differs')


def transfer_inputs(args,record,request_path,remote_request):
    """Preserve existing files; pin only the small completed ReLU fit closure."""
    if 'fit_execution_host' not in record:raise ValueError('explicit cross-host origin required')
    host=record['host'];script=Path(args.remote_source)/'scripts/rev4_featpot/shaping_evaluation_worker.py'
    base=['python3','-B',str(script),'--study',str(args.study),'--request',str(remote_request)]
    subprocess.run(['ssh',host,shlex.join([*base,'--verify-inputs'])],check=True)
    paths=request_path.with_name('INPUT_PATHS.txt')
    paths.write_bytes(b''.join(str(args.study/name).lstrip('/').encode()+b'\0' for name in record['source_sha256']))
    receiver=['env','TMPDIR=/home/lilith/tmp','/home/lilith/work/claudehints/scripts/run-heavy',
              '--mem','512M','--jobs','1','--',*base,'--receive','--']
    subprocess.run(['rsync','-rlptR','--omit-dir-times','--no-implied-dirs','--ignore-existing','--from0',
        '--files-from='+str(paths),'--rsync-path='+shlex.join(receiver),'/',host+':/'],check=True)


def receive(args):
    record=json.loads(args.request.read_text());verify_request(args.study,record,allow_missing=True)
    if 'fit_execution_host' not in record:raise ValueError('explicit cross-host origin required')
    arguments=args.rsync_arguments
    if arguments[:1]==['--']:arguments=arguments[1:]
    if arguments[:1]!=['--server']:raise ValueError('rsync server arguments required')
    from shaping_storage import reservation
    minimum=json.loads((args.study/'PLAN.json').read_text())['minimum_ssd_free_bytes']
    with reservation(args.study,args.request.parent/'input-admission',minimum,bytes_required=128*1024**2):
        subprocess.run(['rsync',*arguments],check=True);verify_request(args.study,record)


def capacity(study=None):
    busy=0
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            if path.read_text().strip() in ('zensim_mlp_trai','rustc'):busy+=1
        except (FileNotFoundError,ProcessLookupError):pass
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    from shaping_training_budget import training_limit
    return busy<training_limit(study) and available>=7*1024**3


def require_terminal(record,files):
    if 'later_scope' in record:
        base=Path('runs')/record['cell']['id']/record['attempt']
        repaired=record['later_scope']['representation']['representation']=='research-relu-max-pairs-v1'
        parent=Path('reexport_max_pairs')/record['cell']['id']/record['attempt'] if repaired else base
        final=parent/'final' if repaired else parent/'evaluation'
        complete=all(str(final/name) in files for name in ('REPORT.json','COMPLETE.json'))
        failed=str(parent/'LATER_EVALUATION_FAILURE.json') in files and str(final/'FAILED_GATE.json') in files
        if not complete and not failed:raise ValueError('complete later-stage report or explicit failed gate required')
        return
    roots=[Path('runs')/record['cell']['id']/record['attempt']]
    if record['cell']['family'] in ('relu','softplus'):
        roots.append(Path('reexport_max_pairs')/record['cell']['id']/record['attempt'])
    for root in roots:
        if not any(str(root/name) in files for name in ('CONVERGENCE.json','CONVERGENCE_FAILURE.json','EVALUATION_FAILURES.json')):
            raise ValueError('complete convergence or explicit failed evaluation required')


def work(args):
    record=json.loads(args.request.read_text());verify_request(args.study,record)
    from shaping_storage import reservation
    minimum=json.loads((args.study/'PLAN.json').read_text())['minimum_ssd_free_bytes']
    if args.bundle.parent.stat().st_dev!=args.study.stat().st_dev:
        raise ValueError('evaluation export must use reserved study filesystem')
    with reservation(args.study,args.bundle,minimum,bytes_required=1024**3):
        return _work(args)


def _work(args):
    record=json.loads(args.request.read_text());verify_request(args.study,record)
    if not capacity(args.study):return dict(stage='waiting_for_capacity')
    if args.bundle.exists():raise ValueError('preserve existing evaluation export')
    from shaping_supervision import sweep,later_sweep
    from shaping_reexport_supervision import sweep as supplemental
    values=SimpleNamespace(study=args.study,binary_dir=args.binary_dir,rescue_binary_dir=getattr(args,'rescue_binary_dir',None),
                           cells=(record['cell']['id'],))
    if 'later_scope' in record:
        values.repair_binary_dir=getattr(args,'repair_binary_dir',None)
        values.repair_inspector=getattr(args,'inspector',None)
        values.candidate_binary_dir=getattr(args,'candidate_binary_dir',[])
        later_sweep(values)
    else:sweep(values)
    if 'later_scope' not in record and record['cell']['family'] in ('relu','softplus'):
        supplemental(SimpleNamespace(study=args.study,binary_dir=args.repair_binary_dir,inspector=args.inspector,
                                     cells=values.cells,family=record['cell']['family']))
    verify_request(args.study,record)
    files={}
    for relative in components(record['cell']['id'],record['attempt']):
        path=args.study/relative
        if path.is_file():files[str(relative)]=dict(bytes=path.stat().st_size,sha256=sha(path))
        elif path.is_dir():files.update({str(relative/name):row for name,row in inventory(path).items()})
    if not files:raise ValueError('no evaluation evidence produced')
    require_terminal(record,files)
    if len(files)>=1000 or sum(row['bytes'] for row in files.values())>1024**3-16*1024**2:
        raise ValueError('evaluation export exceeds bounded transport allowance')
    manifest=dict(files=files,request_sha256=sha(args.request),host=socket.gethostname(),driver_sha256=sha(__file__))
    args.bundle.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.bundle,'x') as archive:
        import io
        payload=encoded(manifest);header=tarfile.TarInfo('COLLECTION.json');header.size=len(payload)
        archive.addfile(header,io.BytesIO(payload))
        for name in files:archive.add(args.study/name,arcname='payload/'+name,recursive=False)
    result=dict(stage='complete',bundle_sha256=sha(args.bundle),files=len(files),request_sha256=sha(args.request))
    Path(str(args.bundle)+'.json').write_bytes(encoded(result));return result


def publish(study,request_path,bundle,pin,staging):
    from shaping_worker_collect import unpack
    record=json.loads(request_path.read_text());manifest=unpack(bundle,staging,pin)
    if manifest['request_sha256']!=sha(request_path) or manifest['host']!=record['host']:
        raise ValueError('evaluation export request differs')
    for name,expected in record['source_sha256'].items():
        if sha(study/name)!=expected:raise ValueError('coordinator fit changed during evaluation')
    allowed=components(record['cell']['id'],record['attempt'])
    if any(not any(Path(name)==prefix or prefix in Path(name).parents for prefix in allowed) for name in manifest['files']):
        raise ValueError('unexpected evaluation export component')
    require_terminal(record,manifest['files'])
    for relative in allowed:
        source=staging/relative;target=study/relative
        if not source.exists():continue
        if target.exists():raise ValueError('preserve existing coordinator evaluation')
    # Publish entire report trees; completion markers never precede their files.
    for relative in allowed:
        source=staging/relative
        if source.exists():
            target=study/relative;target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
    receipt=dict(request_sha256=sha(request_path),bundle_sha256=pin,source_host=record['host'],
                 files=manifest['files'],imported_utc=datetime.now(timezone.utc).isoformat())
    if 'fit_execution_host' in record:receipt['fit_execution_host']=record['fit_execution_host']
    target=study/'runs'/record['cell']['id']/record['attempt']/'EVALUATION_WORKER_IMPORT.json'
    target.write_bytes(encoded(receipt));return receipt


def watch(args):
    placements={}
    if getattr(args,'placement_file',None):
        placement=json.loads(args.placement_file.read_text())
        if placement['plan_sha256']!=sha(args.study/'PLAN.json') or placement['outer_results_read']:
            raise ValueError('evaluation placement plan differs')
        placements=placement['placements']
        if any(host not in HOSTS for host in placements.values()):raise ValueError('unapproved evaluation placement')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.owner').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            pending=[];blocked=[];imported=[]
            for path in sorted((args.study/'runs').glob('*/attempt-*/RESULT.json')):
                value=json.loads(path.read_text());run=path.parent
                stages=LATER_STAGES if getattr(args,'later_only',False) else ('inner','fresh_control_inner','rescue')
                if value.get('stage')!='trained' or value['cell']['stage'] not in stages or not (run/'WORKER_COLLECTION.json').exists():continue
                if any((args.study/p).exists() for p in components(value['cell']['id'],run.name)):continue
                record=request(args.study,run,target_host=placements.get(value['cell']['id']));host=record['host'];directory=args.output/value['cell']['id']
                request_path=directory/'REQUEST.json'
                if directory.exists():
                    if (directory/'FAILURE.json').exists() or not (directory/'WAITING.json').exists():blocked.append(value['cell']['id']);continue
                    if json.loads(request_path.read_text())!=record:raise ValueError('waiting evaluation request drift')
                else:
                    directory.mkdir();request_path.write_bytes(encoded(record))
                remote=Path(args.remote_source)/'scripts/rev4_featpot/shaping_evaluation_worker.py'
                remote_request=args.study.parent/'evaluation-worker-jobs'/value['cell']['id']/'REQUEST.json'
                remote_bundle=remote_request.parent/'evaluation.tar'
                subprocess.run(['ssh',host,'mkdir -p '+shlex.quote(str(remote_request.parent))],check=True)
                subprocess.run(['scp',str(request_path),host+':'+str(remote_request)],check=True)
                if 'fit_execution_host' in record:transfer_inputs(args,record,request_path,remote_request)
                command=['env','TMPDIR=/home/lilith/tmp','PYTHONPATH='+str(args.study.parent/'python-runtime'),
                    '/home/lilith/work/claudehints/scripts/run-heavy','--mem','3G','--jobs','1','--',
                    'python3','-B',str(remote),'--work','--study',str(args.study),'--request',str(remote_request),
                    '--bundle',str(remote_bundle),'--binary-dir',str(args.binary_dir),
                    '--repair-binary-dir',str(args.repair_binary_dir),'--inspector',str(args.inspector)]
                if getattr(args,'rescue_binary_dir',None):command.extend(['--rescue-binary-dir',str(args.rescue_binary_dir)])
                for directory_path in getattr(args,'candidate_binary_dir',[]):command.extend(['--candidate-binary-dir',str(directory_path)])
                log_path=directory/f"worker-{len(list(directory.glob('worker-*.log'))):04}.log"
                with log_path.open('x') as log:
                    result=subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    (directory/'FAILURE.json').write_bytes(encoded(dict(exit_code=result.returncode,host=host,remote_completion_unresolved=True)))
                    blocked.append(record['cell']['id']);continue
                # A capacity wait creates no export and must not replay evaluation.
                probe=subprocess.run(['ssh',host,'test -f '+shlex.quote(str(remote_bundle)+'.json')])
                if probe.returncode:
                    (directory/'WAITING.json').write_bytes(encoded(dict(stage='capacity_wait',host=host)))
                    pending.append(record['cell']['id']);continue
                bundle=directory/'evaluation.tar'
                from shaping_storage import reservation
                minimum=json.loads((args.study/'PLAN.json').read_text())['minimum_ssd_free_bytes']
                with reservation(args.study,bundle,minimum):
                    subprocess.run(['scp',host+':'+str(remote_bundle),host+':'+str(remote_bundle)+'.json',str(directory)+'/'],check=True)
                    sidecar=json.loads((directory/(remote_bundle.name+'.json')).read_text())
                    publish(args.study,request_path,bundle,sidecar['bundle_sha256'],directory/'verified')
                imported.append(record['cell']['id'])
            status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),stage='watching',
                        imported_this_sweep=imported,waiting=pending,requires_reconciliation=blocked)
            (args.output/'STATUS.json').write_bytes(encoded(status));print(json.dumps(status),flush=True)
            if not args.watch:return status
            time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    for name in ('binary-dir','rescue-binary-dir','repair-binary-dir','inspector','request','bundle','output','remote-source','placement-file'):p.add_argument('--'+name,type=Path)
    p.add_argument('--work',action='store_true');p.add_argument('--watch',action='store_true')
    p.add_argument('--later-only',action='store_true')
    p.add_argument('--candidate-binary-dir',type=Path,action='append',default=[])
    p.add_argument('--verify-inputs',action='store_true');p.add_argument('--receive',action='store_true')
    p.add_argument('rsync_arguments',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.verify_inputs:print(json.dumps(verify_request(args.study,json.loads(args.request.read_text()),allow_missing=True)),flush=True)
    elif args.receive:receive(args)
    elif args.work:print(json.dumps(work(args)),flush=True)
    else:watch(args)


if __name__=='__main__':main()
