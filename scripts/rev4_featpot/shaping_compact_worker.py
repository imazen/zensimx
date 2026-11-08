"""Run unchanged frozen worker admissions in short-lived, bounded processes.

The native owner, request, scientific gate and command are unchanged. Only idle
wave queues may hand over; an active fit or CPU claim refuses that handoff.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager,ExitStack
from datetime import datetime,timezone
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.writing')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def install(source):
    path=source/'scripts/rev4_featpot';sys.path.insert(0,str(path))
    import shaping_rescue_worker as frozen
    if Path(frozen.__file__).resolve()!=path/'shaping_rescue_worker.py':raise ValueError('another worker source is already loaded')
    return frozen


def reproduce(source,request):
    frozen=install(source);record=json.loads(request.read_text());frozen.verify(record)
    kind=record.get('execution_kind')
    if kind=='corrected_inner':from shaping_corrected_worker import reproduce as owner
    elif kind=='bootstrap':from shaping_bootstrap_worker import reproduce as owner
    elif kind=='control':
        from shaping_control_worker import reproduce as owner
        record['_request_path']=str(request)
    elif kind is not None:from shaping_downstream_worker import reproduce as owner
    else:
        from shaping_rescue import resolve
        resolution=resolve(Path(record['study']),record['cell']['id'])
        command=['python3','-B',str(source/'scripts/rev4_featpot/shaping_rescue_execution.py'),
                 '--study',record['study'],'--binary-dir',record['binary_dir'],
                 '--neutral-qualification',record['neutral_qualification'],'--cell',record['cell']['id']]
    if kind is not None:resolution,command=owner(record)
    if resolution!=record['resolution']:raise ValueError('worker cannot reproduce original evidence gate')
    return dict(request_sha256=sha(request),resolution=resolution,command=command,
                frozen_worker_sha256=sha(frozen.__file__),launcher_sha256=sha(__file__))


@contextmanager
def preflight_slot(study):
    # Two half-GiB scopes, independent of training. No long training lock.
    root=study/'preflight-admission';root.mkdir(exist_ok=True)
    while True:
        available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        if available>=4*1024**3+512*1024**2:
            for index in range(2):
                lock=(root/f'{index}.lock').open('a')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lock.close();continue
                try:yield
                finally:lock.close()
                return
        time.sleep(1)


def preflight(source,request,wrapper=None):
    record=json.loads(request.read_text());study=Path(record['study'])
    wrapper=wrapper or Path.home()/'work/claudehints/scripts/run-heavy'
    command=[str(wrapper),'--mem','512M','--jobs','1','--',sys.executable,'-B',str(Path(__file__).resolve()),
             'preflight','--source',str(source),'--request',str(request)]
    with preflight_slot(study):
        result=subprocess.run(command,capture_output=True,text=True,check=True)
    value=json.loads(result.stdout)
    if value['request_sha256']!=sha(request) or value['resolution']!=record['resolution']:
        raise ValueError('isolated admission response differs')
    return value,result.stderr


def lightweight():
    names=('numpy','pandas','pyarrow','scipy','sklearn')
    loaded=[name for name in names if name in sys.modules]
    if loaded:raise ValueError('long-lived controller imported data libraries: '+','.join(loaded))


def work(source,request):
    frozen=install(source);record=json.loads(request.read_text());frozen.verify(record)
    study=Path(record['study']);key=record['cell']['id'];kind=record.get('execution_kind')
    if (study/'runs'/key).exists():raise ValueError('preserve existing attempt; never replay')
    receipt,log=preflight(source,request)
    with request.with_name('COMPACT_PREFLIGHT.log').open('x') as stream:stream.write(log)
    with request.with_name('COMPACT_PREFLIGHT.json').open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
    from shaping_rescue_dispatch import reserve_worker,release_worker,bind_worker
    from shaping_storage import reservation
    lightweight()
    while True:
        if kind=='control':admission=reserve_worker(study,key,memory_allowance_bytes=4*1024**3)
        elif kind=='corrected_inner':admission=reserve_worker(study,key,warmup_until_checkpoint=True)
        else:admission=reserve_worker(study,key)
        if not admission['can_start']:time.sleep(10);continue
        if kind=='corrected_inner':
            from shaping_snapshot_retention import capacity
            try:observation=capacity(study,key)
            except Exception:release_worker(study,key);raise
            if not observation['can_start']:
                release_worker(study,key)
                save(request.with_name('STATUS.json'),dict(stage='waiting_for_snapshot_retirement',pid=os.getpid(),
                     request_sha256=sha(request),snapshot_capacity=observation));time.sleep(10);continue
        break
    try:
        # A second verifier after the capacity wait retains the immutable capsule.
        frozen.verify(record)
        if (study/'runs'/key).exists():raise ValueError('attempt appeared while waiting; preserve it')
        with reservation(study,study/'runs'/key/'attempt-001',30*1024**3):
            with request.with_name('worker.log').open('x') as stream:
                process=subprocess.Popen(receipt['command'],stdout=stream,stderr=subprocess.STDOUT)
                bind_worker(study,key,process.pid)
                status=dict(pid=process.pid,request_sha256=sha(request),stage='running',command=receipt['command'],
                            compact_launcher_sha256=sha(__file__),preflight_sha256=sha(request.with_name('COMPACT_PREFLIGHT.json')))
                save(request.with_name('STATUS.json'),status);code=process.wait()
            status.update(exit_code=code,stage=('analysis_finished' if kind=='bootstrap' else 'trained') if code==0 else 'failed')
            save(request.with_name('STATUS.json'),status)
            if code:raise RuntimeError('frozen worker failed; preserve attempt')
            return status
    finally:release_worker(study,key)


def manifest(path):
    wave=json.loads(path.read_text())
    if not 1<=wave['workers']<=4 or len(wave['requests'])!=len(set(wave['requests'])):
        raise ValueError('bounded distinct worker wave required')
    if set(wave['request_sha256'])!=set(wave['requests']):raise ValueError('complete wave request pins required')
    for name,pin in wave['request_sha256'].items():
        if sha(name)!=pin:raise ValueError('wave request drift')
    return wave


def handoff(source,path,output,freeze_only=False):
    wave=manifest(path);status=json.loads(path.with_name('WAVE_STATUS.json').read_text());pid=status['pid']
    if status['manifest_sha256']!=sha(path) or status['completed'] or status['failed']:
        raise ValueError('only an entirely unopened idle wave may hand over')
    proc=Path('/proc')/str(pid);command=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
    if (str(source/'scripts/rev4_featpot/shaping_rescue_worker.py') not in command or 'wave' not in command
        or command.count('--request')!=1 or command[command.index('--request')+1]!=str(path)):
        raise ValueError('original wave process differs')
    if output.exists():raise ValueError('preserve prior handoff')
    records=[json.loads(Path(name).read_text()) for name in wave['requests']];studies={r['study'] for r in records}
    if len(studies)!=1:raise ValueError('one study required')
    study=Path(studies.pop());frozen=install(source)
    for row in records:frozen.verify(row)
    stopped=False
    with ExitStack() as stack:
        for name in ('cpu-admission','storage-admission'):
            lock=stack.enter_context((study/name/'transaction.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            os.kill(pid,signal.SIGSTOP);stopped=True
            for _ in range(200):
                states=[p.read_text().rsplit(')',1)[1].split()[0] for p in (proc/'task').glob('*/stat')]
                if states and all(state=='T' for state in states):break
                time.sleep(.01)
            else:raise ValueError('idle wave did not stop at its admission boundary')
            if (proc/f'task/{pid}/children').read_text().strip():raise ValueError('wave has active children; preserve it')
            claims=json.loads((study/'cpu-admission/CLAIMS.json').read_text())
            if any(row['cell']['id'] in claims or (study/'runs'/row['cell']['id']).exists() for row in records):
                raise ValueError('wave contains an admitted or started fit; preserve it')
            if any(row.get('dispatcher_pid')==pid for row in claims.values()):raise ValueError('wave still owns a CPU claim')
            manifest(path);output.mkdir()
            record=dict(schema='idle-worker-wave-handoff-v1',stage='prepared',wave=str(path),wave_sha256=sha(path),
                        source=str(source),retired_pid=pid,retired_start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19],
                        previous_status=status,request_sha256=wave['request_sha256'],launcher_sha256=sha(__file__),
                        native_fits_interrupted=0,checked_utc=datetime.now(timezone.utc).isoformat())
            save(output/'HANDOFF.json',record)
            if freeze_only:
                record['stage']='frozen_idle_wave';save(output/'HANDOFF.json',record)
                stopped=False  # Keep this verified idle queue at its boundary.
                return record
            os.kill(pid,signal.SIGTERM)
        finally:
            if stopped and proc.exists():os.kill(pid,signal.SIGCONT)
    for _ in range(100):
        if not proc.exists() or (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
        time.sleep(.02)
    else:raise ValueError('old idle wave did not terminate; reconcile without replay')
    record['stage']='retired_idle_wave';save(output/'HANDOFF.json',record);return record


def run_wave(args):
    wave=manifest(args.wave);install(args.source)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'WAVE_STATUS.json').exists():raise ValueError('preserve existing compact wave; never replay')
    if not args.qualify:
        hand=json.loads((args.output/'HANDOFF.json').read_text())
        if (hand['stage']!='retired_idle_wave' or hand['wave_sha256']!=sha(args.wave)
            or hand['launcher_sha256']!=sha(__file__) or hand['source']!=str(args.source)):
            raise ValueError('explicit matching idle handoff required')
    status=dict(pid=os.getpid(),manifest_sha256=sha(args.wave),stage='qualifying' if args.qualify else 'running',
                completed={},failed={},launcher_sha256=sha(__file__),frozen_source=str(args.source))
    with (args.output/'WAVE_STATUS.json').open('x') as stream:stream.write(json.dumps(status,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=wave['workers']) as pool:
        action=(lambda request:preflight(args.source,request)[0]) if args.qualify else (lambda request:work(args.source,request))
        futures={pool.submit(action,Path(name)):name for name in wave['requests']}
        for future in as_completed(futures):
            name=futures[future]
            try:status['completed'][name]=future.result()
            except Exception as error:status['failed'][name]=str(error)
            lightweight();save(args.output/'WAVE_STATUS.json',status)
    status.update(stage='failed_preserved' if status['failed'] else 'qualified' if args.qualify else 'complete',
                  checked_utc=datetime.now(timezone.utc).isoformat(),heavy_modules_retained=[])
    save(args.output/'WAVE_STATUS.json',status);return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('preflight','handoff','wave'))
    parser.add_argument('--source',type=Path,required=True)
    for name in ('request','wave','output','study'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--qualify',action='store_true')
    parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    value=(reproduce(args.source,args.request) if args.mode=='preflight' else
           handoff(args.source,args.wave,args.output,args.freeze_only) if args.mode=='handoff' else run_wave(args))
    print(json.dumps(value,sort_keys=True),flush=True)
