"""Move admitted rescue execution while reproducing the original evidence gate."""
from __future__ import annotations
import argparse,contextlib,json,os,shlex,socket,subprocess,sys,time
from pathlib import Path
from shaping_completion import DECLARATION_SHA256,REPO,encoded,sha

# Frozen Linux/x86 native artifacts can execute on these transport endpoints.
# The study's current owner-approved pool remains a separate admission gate.
HOSTS=('r3500','r3800x','r5600g','i134','r7900x','i270')
MAX_INPUT_BYTES=384*1024**2
DATA_ROOTS=(Path('/mnt/v/output/zensim'),Path('/srv/zen-research'))


def limits(kind):
    # Bootstrap reproduces all three outer admissions. Controls retain all 44
    # frozen array/table inputs. Ordinary neural fit closures keep their cap.
    return (2*1024**3,16000) if kind in ('bootstrap','control','cost_prepare') else (MAX_INPUT_BYTES,4000)


def memory_allowance(record):
    """Reserve the same bound as the largest native phase and its parent scope."""
    kind=record.get('execution_kind')
    if kind in ('control','extension'):
        return 4*1024**3
    if kind in ('outer','variant'):
        phases=record['resolution']['schedule']['phase_epochs']
        if not phases or any(type(e) is not int or e not in (120,240,480) for e in phases):
            raise ValueError('registered downstream phase lengths required for memory admission')
        if max(phases)==480:
            return 4*1024**3
    return 2*1024**3


def source_identity(value,source=None):
    """Source exports may move; data paths and source-content hashes may not."""
    source=REPO if source is None else Path(source);path=Path(value)
    return 'source/'+path.relative_to(source).as_posix() if source in path.parents else str(value)


@contextlib.contextmanager
def reads():
    """Single-process metadata capture; workers still rerun the actual gate."""
    paths=set();original=Path.open
    def tracked(path,mode='r',*args,**kwargs):
        stream=original(path,mode,*args,**kwargs)
        if 'r' in mode or '+' in mode:paths.add(path.absolute())
        return stream
    Path.open=tracked
    try:yield paths
    finally:Path.open=original


def category(path,source=REPO):
    if source in path.parents:return 'source',str(path.relative_to(source))
    if path.name=='color_noise_fx2_declaration.json':return 'declaration','scripts/rev4_featpot/color_noise_fx2_declaration.json'
    if any(root in path.parents for root in DATA_ROOTS):return 'data',str(path)
    raise ValueError('unexpected rescue dependency: '+str(path))


def resolve_inputs(study,cell_id,host,binaries,qualification):
    from shaping_execution import load_registration
    from shaping_rescue import resolve
    from shaping_rescue_execution import primary_source
    from shaping_reduced_scope import admit_rescue
    with reads() as paths:
        _,cells,_=load_registration(study);resolution=resolve(study,cell_id);cell=resolution['cell']
        admit_rescue(study,cell,host=host)
        from shaping_corrected_view import require_downstream_build
        require_downstream_build(study,cell['family'],sha(binaries/'BUILD.json'))
        if (study/'runs'/cell_id).exists():raise ValueError('preserve existing rescue; never replay')
        groups=study/'comparators'/cell['partition']/'GROUPS.json'
        record=json.loads(groups.read_text())
        if any(row[2]<=0 or row[3]!=0 for row in record['groups']):raise ValueError('TRAIN-only rescue inputs required')
        paths.add(groups);paths.update(map(Path,record['file_sha256']))
        metadata=Path('/mnt/v/output/zensim/tree-study-20261106/peers/metadata.parquet')
        plan=json.loads((study/'PLAN.json').read_text())
        if sha(metadata)!=plan['input_pins'][str(metadata)]:raise ValueError('family metadata drift')
        sources=[primary_source(study,cells,cell,resolution['resolved_setting'])]
        if '_pretrained_' in cell['family']:
            sources.append(primary_source(study,cells,cell,resolution['resolved_setting'],cell['family'].split('_pretrained_')[0]+'_joint_identity'))
        for source,original in sources:
            for name in ('RUN.json','RESULT.json','keep.txt','HEAD.json','SHAPING.json',
                         'COLLECTION_DERIVED_INPUTS.json','WORKER_COLLECTION.json'):
                if (source/name).exists():paths.add(source/name)
            declaration=Path(original['command'][original['command'].index('--derived-inputs')+1])
            if not (source/'COLLECTION_DERIVED_INPUTS.json').exists():
                if sha(declaration)!=DECLARATION_SHA256:raise ValueError('frozen declaration drift')
                paths.add(declaration)
        build=json.loads((binaries/'BUILD.json').read_text())
        for name,pin in build['binary_sha256'].items():
            if sha(binaries/name)!=pin:raise ValueError('qualified rescue binary drift')
        if cell['family'] in ('softplus','silu_monotone','silu_diagnostic'):
            q=json.loads((qualification/'RESULT.json').read_text())
            if not q['match'] or q['epochs']!=240 or q['build_sha256']!=sha(binaries/'BUILD.json'):
                raise ValueError('full neutral qualification required')
    return seal_inputs(paths,study,cell,host,binaries,qualification,resolution)


def seal_inputs(paths,study,cell,host,binaries,qualification,resolution,*,execution_kind=None,human_qualification=None):
    # Bind every imported project module, in addition to the documents and data
    # actually read. The exported worker source must match these exact bytes.
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).absolute()
            if REPO in path.parents and path.suffix=='.py':paths.add(path)
    paths.add(Path(__file__).absolute())
    for name in ('shaping_training_budget.py','shaping_rescue_dispatch.py'):
        paths.add(REPO/'scripts/rev4_featpot'/name)
    budget=study/'TRAINING_CPU_BUDGET.json'
    if budget.exists():
        paths.add(budget);paths.add(Path(json.loads(budget.read_text())['approval_path']))
    files={}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():raise ValueError('nonregular rescue dependency')
        kind,name=category(path);files[str(path)]=dict(kind=kind,name=name,bytes=path.stat().st_size,sha256=sha(path))
    total=sum(row['bytes'] for row in files.values())
    max_bytes,max_files=limits(execution_kind)
    if total>max_bytes or len(files)>max_files:raise ValueError('rescue evidence capsule exceeds bounded input allowance')
    record=dict(schema='shaping-rescue-worker-v1',cell=cell,host=host,study=str(study),binary_dir=str(binaries),
                neutral_qualification=str(qualification) if qualification is not None else None,resolution=resolution,files=files,input_bytes=total,
                plan_sha256=sha(study/'PLAN.json'),build_sha256=sha(binaries/'BUILD.json'),
                driver_sha256=sha(__file__),source_root=str(REPO),outer_results_read=execution_kind in ('bootstrap','cost_prepare'))
    if execution_kind is not None:
        record.update(schema='shaping-native-worker-v2',execution_kind=execution_kind,
                      human_qualification=str(human_qualification) if human_qualification else None)
    return record


def register(study,cell,host,binaries,qualification,output,*,execution_kind=None,human_qualification=None):
    if host not in HOSTS:raise ValueError('approved compute worker required')
    if execution_kind is None:record=resolve_inputs(study,cell,host,binaries,qualification)
    else:
        if execution_kind=='corrected_inner':
            from shaping_corrected_worker import resolve_inputs as corrected_inputs
            record=corrected_inputs(study,cell,host,binaries,qualification)
        else:
            from shaping_downstream_worker import resolve_inputs as later_inputs
            record=later_inputs(study,cell,host,binaries,qualification,execution_kind,human_qualification)
    output.mkdir(parents=True,exist_ok=False);(output/'REQUEST.json').write_bytes(encoded(record))
    assignment=dict(host=host,cells=[cell],partition=record['cell']['partition'],
                    plan_sha256=record['plan_sha256'],build_sha256=record['build_sha256'])
    if execution_kind is not None:
        assignment.update(worker_request_path=str(output/'REQUEST.json'),worker_request_sha256=sha(output/'REQUEST.json'))
    (output/'ASSIGNMENT.json').write_bytes(encoded(assignment));return record


def verify(record,*,allow_missing=False):
    if (record['schema'] not in ('shaping-rescue-worker-v1','shaping-native-worker-v2') or record['host']!=socket.gethostname()
        or record['host'] not in HOSTS or record['driver_sha256']!=sha(__file__)
        or record['outer_results_read'] is not (record.get('execution_kind') in ('bootstrap','cost_prepare'))):
        raise ValueError('worker request identity differs')
    if record['schema']=='shaping-native-worker-v2':
        stages={'outer':('outer','fresh_control_outer','fresh_control_outer_matched'),
                'variant':('source_curve','row_curve','shuffle','human_only'),'corrected_inner':('inner',),'bootstrap':('bootstrap',),
                'control':('outer','source_curve','shuffle','human_only'),'extension':('exposure_extension',),
                'cost_prepare':('cost_prepare',)}
        if record.get('execution_kind') not in stages or record['cell']['stage'] not in stages[record['execution_kind']]:
            raise ValueError('downstream worker stage differs')
    elif record.get('execution_kind') is not None:raise ValueError('legacy rescue request cannot change execution kind')
    max_bytes,max_files=limits(record.get('execution_kind'))
    if len(record['files'])>max_files or sum(r['bytes'] for r in record['files'].values())>max_bytes:
        raise ValueError('worker input allowance exceeded')
    missing=[]
    for original,row in record['files'].items():
        if not Path(original).is_absolute() or '..' in Path(original).parts:raise ValueError('unsafe dependency path')
        kind,name=category(Path(original),Path(record['source_root']))
        if kind!=row['kind'] or name!=row['name']:raise ValueError('dependency path scope differs')
        if kind=='declaration' and row['sha256']!=DECLARATION_SHA256:raise ValueError('frozen declaration pin differs')
        path=REPO/name if kind in ('source','declaration') else Path(name)
        if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('symlink in worker dependency')
        if allow_missing and kind=='data' and not path.exists():missing.append(str(path));continue
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('worker dependency drift: '+str(path))
    return missing


def transfer(request,remote_source):
    record=json.loads(request.read_text());host=record['host']
    if host not in HOSTS:raise ValueError('approved target required')
    remote_request=request # Research paths intentionally stay byte-identical.
    script=remote_source/'scripts/rev4_featpot/shaping_rescue_worker.py'
    subprocess.run(['ssh',host,'mkdir -p '+shlex.quote(str(request.parent))],check=True)
    subprocess.run(['scp',str(request),host+':'+str(remote_request)],check=True)
    command=['python3','-B',str(script),'verify','--request',str(remote_request),'--allow-missing']
    # Refuse existing drift before rsync; --ignore-existing cannot overwrite it.
    subprocess.run(['ssh',host,shlex.join(command)],check=True)
    manifest=request.with_name('INPUT_PATHS.txt')
    manifest.write_bytes(b''.join(Path(name).as_posix().lstrip('/').encode()+b'\0'
                                for name,row in record['files'].items() if row['kind']=='data'))
    receiver=['/home/lilith/work/claudehints/scripts/run-heavy','--mem','512M','--jobs','1','--',
              'python3','-B',str(script),'receive','--request',str(remote_request),'--']
    subprocess.run(['rsync','-rlptR','--omit-dir-times','--no-implied-dirs','--ignore-existing','--from0','--files-from='+str(manifest),
                    '--rsync-path='+shlex.join(receiver),'/',host+':/'],check=True)
    subprocess.run(['scp',str(request.with_name('ASSIGNMENT.json')),host+':'+str(request.with_name('ASSIGNMENT.json'))],check=True)
    subprocess.run(['ssh',host,shlex.join(command[:-1])],check=True)


def receive(request,arguments):
    from shaping_storage import reservation
    record=json.loads(request.read_text());verify(record,allow_missing=True)
    if not arguments or arguments[0]!='--server':raise ValueError('rsync server arguments required')
    with contextlib.ExitStack() as reservations:
        remaining=limits(record.get('execution_kind'))[0];part=0
        while remaining:
            amount=min(remaining,1024**3)
            destination=request.parent/('input-admission' if part==0 else f'input-admission-{part}')
            reservations.enter_context(reservation(Path(record['study']),destination,30*1024**3,bytes_required=amount))
            remaining-=amount;part+=1
        subprocess.run(['rsync',*arguments],check=True)
        verify(record)


def work(request):
    from shaping_rescue import resolve
    from shaping_rescue_dispatch import reserve_worker,release_worker,bind_worker
    from shaping_storage import reservation
    record=json.loads(request.read_text());verify(record)
    study=Path(record['study']);cell=record['cell'];key=cell['id']
    if record.get('execution_kind') is not None:
        if record['execution_kind']=='corrected_inner':from shaping_corrected_worker import reproduce
        elif record['execution_kind']=='bootstrap':from shaping_bootstrap_worker import reproduce
        elif record['execution_kind']=='cost_prepare':from shaping_cost_worker import reproduce
        elif record['execution_kind']=='control':
            from shaping_control_worker import reproduce
            record['_request_path']=str(request)
        else:from shaping_downstream_worker import reproduce
        resolution,command=reproduce(record)
    else:
        resolution=resolve(study,key)
        command=None
    if resolution!=record['resolution']:raise ValueError('worker cannot reproduce original evidence gate')
    if command is None:
        command=['python3','-B',str(REPO/'scripts/rev4_featpot/shaping_rescue_execution.py'),'--study',str(study),
                 '--binary-dir',record['binary_dir'],'--neutral-qualification',record['neutral_qualification'],'--cell',key]
    if (study/'runs'/key).exists():raise ValueError('preserve existing attempt; never replay')
    phase_count=None
    if record.get('execution_kind') in ('outer','variant','extension'):
        phase_count=len(record['resolution']['schedule']['phase_epochs'])
        if phase_count not in (1,2):raise ValueError('one or two registered snapshot phases required')
    admission_options={}
    if memory_allowance(record)==4*1024**3:admission_options['memory_allowance_bytes']=4*1024**3
    if record.get('execution_kind')=='corrected_inner':admission_options['warmup_until_checkpoint']=True
    if phase_count is not None:admission_options['snapshot_phases']=phase_count
    while True:
        admission=reserve_worker(study,key,**admission_options)
        if not admission['can_start']:
            time.sleep(10);continue
        if record.get('execution_kind')=='corrected_inner' or phase_count is not None:
            from shaping_snapshot_retention import capacity
            try:observation=capacity(study,key)
            except Exception:
                release_worker(study,key);raise
            if not observation['can_start']:
                release_worker(study,key)
                request.with_name('STATUS.json').write_bytes(encoded(dict(stage='waiting_for_snapshot_retirement',
                    pid=os.getpid(),request_sha256=sha(request),snapshot_capacity=observation)))
                time.sleep(10);continue
        break
    try:
        with reservation(study,study/'runs'/key/'attempt-001',(34 if phase_count is not None else 30)*1024**3):
            with request.with_name('worker.log').open('x') as log:
                environment=os.environ.copy();environment.pop('ZENSIM_PHASE_SNAPSHOTS',None)
                if phase_count is not None:environment['ZENSIM_PHASE_SNAPSHOTS']='1'
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
                bind_worker(study,key,process.pid)
                status=dict(pid=process.pid,request_sha256=sha(request),stage='running',command=command)
                request.with_name('STATUS.json').write_bytes(encoded(status));code=process.wait()
            status.update(exit_code=code,stage=('analysis_finished' if record.get('execution_kind') in ('bootstrap','cost_prepare') else 'trained') if code==0 else 'failed')
            request.with_name('STATUS.json').write_bytes(encoded(status))
            if code:raise RuntimeError('rescue worker failed; preserve attempt')
            return status
    finally:release_worker(study,key)


def wave(manifest):
    """Bound controller concurrency; individual jobs share CPU/SSD admission."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    record=json.loads(manifest.read_text());requests=[Path(p) for p in record['requests']]
    if not 1<=record['workers']<=4 or len(set(requests))!=len(requests):raise ValueError('bounded distinct worker wave required')
    for path,pin in record['request_sha256'].items():
        if sha(path)!=pin:raise ValueError('wave request drift')
    if set(record['request_sha256'])!=set(map(str,requests)):raise ValueError('complete wave pins required')
    status=dict(pid=os.getpid(),manifest_sha256=sha(manifest),stage='running',completed={},failed={})
    status_path=manifest.with_name('WAVE_STATUS.json')
    if status_path.exists():raise ValueError('preserve existing wave; no automatic replay')
    with status_path.open('x') as out:out.write(encoded(status).decode())
    with ThreadPoolExecutor(max_workers=record['workers']) as pool:
        futures={pool.submit(work,path):str(path) for path in requests}
        for future in as_completed(futures):
            path=futures[future]
            try:status['completed'][path]=future.result()
            except Exception as error:status['failed'][path]=str(error)
            pending=status_path.with_suffix('.writing');pending.write_bytes(encoded(status));pending.replace(status_path)
    status['stage']='failed_with_preserved_attempts' if status['failed'] else 'trained'
    status_path.write_bytes(encoded(status));return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','verify','transfer','receive','work','wave'))
    for name in ('study','binaries','qualification','output','request','remote-source','human-qualification'):p.add_argument('--'+name,type=Path)
    p.add_argument('--execution-kind',choices=('outer','variant','corrected_inner','extension'))
    p.add_argument('--cell');p.add_argument('--host',choices=HOSTS);p.add_argument('--allow-missing',action='store_true')
    a,remaining=p.parse_known_args()
    if a.mode=='receive':
        if remaining[:1]==['--']:remaining=remaining[1:]
        receive(a.request,remaining);return
    if remaining:p.error('unexpected arguments')
    if a.mode=='register':result=register(a.study,a.cell,a.host,a.binaries,a.qualification,a.output,
        execution_kind=a.execution_kind,human_qualification=a.human_qualification)
    elif a.mode=='transfer':result=transfer(a.request,a.remote_source)
    elif a.mode=='verify':result=verify(json.loads(a.request.read_text()),allow_missing=a.allow_missing)
    elif a.mode=='wave':result=wave(a.request)
    else:result=work(a.request)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
