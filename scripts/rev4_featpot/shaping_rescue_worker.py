"""Move admitted rescue execution while reproducing the original evidence gate."""
from __future__ import annotations
import argparse,contextlib,json,os,shlex,socket,subprocess,sys,time
from pathlib import Path
from shaping_completion import DECLARATION_SHA256,REPO,encoded,sha

HOSTS=('r3500','r3800x','r5600g','i134')
MAX_INPUT_BYTES=384*1024**2
DATA_ROOTS=(Path('/mnt/v/output/zensim'),Path('/srv/zen-research'))


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
    # Bind every imported project module, in addition to the documents and data
    # actually read. The exported worker source must match these exact bytes.
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).absolute()
            if REPO in path.parents and path.suffix=='.py':paths.add(path)
    paths.add(Path(__file__).absolute())
    files={}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():raise ValueError('nonregular rescue dependency')
        kind,name=category(path);files[str(path)]=dict(kind=kind,name=name,bytes=path.stat().st_size,sha256=sha(path))
    total=sum(row['bytes'] for row in files.values())
    if total>MAX_INPUT_BYTES or len(files)>4000:raise ValueError('rescue evidence capsule exceeds bounded input allowance')
    return dict(schema='shaping-rescue-worker-v1',cell=cell,host=host,study=str(study),binary_dir=str(binaries),
                neutral_qualification=str(qualification),resolution=resolution,files=files,input_bytes=total,
                plan_sha256=sha(study/'PLAN.json'),build_sha256=sha(binaries/'BUILD.json'),
                driver_sha256=sha(__file__),source_root=str(REPO),outer_results_read=False)


def register(study,cell,host,binaries,qualification,output):
    if host not in HOSTS:raise ValueError('approved compute worker required')
    record=resolve_inputs(study,cell,host,binaries,qualification)
    output.mkdir(parents=True,exist_ok=False);(output/'REQUEST.json').write_bytes(encoded(record))
    assignment=dict(host=host,cells=[cell],partition=record['cell']['partition'],
                    plan_sha256=record['plan_sha256'],build_sha256=record['build_sha256'])
    (output/'ASSIGNMENT.json').write_bytes(encoded(assignment));return record


def verify(record,*,allow_missing=False):
    if (record['schema']!='shaping-rescue-worker-v1' or record['host']!=socket.gethostname()
        or record['host'] not in HOSTS or record['driver_sha256']!=sha(__file__) or record['outer_results_read']):
        raise ValueError('worker request identity differs')
    if len(record['files'])>4000 or sum(r['bytes'] for r in record['files'].values())>MAX_INPUT_BYTES:
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
    with reservation(Path(record['study']),request.parent/'input-admission',30*1024**3,bytes_required=MAX_INPUT_BYTES):
        subprocess.run(['rsync',*arguments],check=True)
        verify(record)


def work(request):
    from shaping_rescue import resolve
    from shaping_rescue_dispatch import reserve_worker,release_worker,bind_worker
    from shaping_storage import reservation
    record=json.loads(request.read_text());verify(record)
    study=Path(record['study']);cell=record['cell'];key=cell['id']
    if resolve(study,key)!=record['resolution']:raise ValueError('worker cannot reproduce original rescue gate')
    if (study/'runs'/key).exists():raise ValueError('preserve existing attempt; never replay')
    while not reserve_worker(study,key)['can_start']:time.sleep(10)
    try:
        with reservation(study,study/'runs'/key/'attempt-001',30*1024**3):
            command=['python3','-B',str(REPO/'scripts/rev4_featpot/shaping_rescue_execution.py'),'--study',str(study),
                     '--binary-dir',record['binary_dir'],'--neutral-qualification',record['neutral_qualification'],'--cell',key]
            with request.with_name('worker.log').open('x') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                bind_worker(study,key,process.pid)
                status=dict(pid=process.pid,request_sha256=sha(request),stage='running',command=command)
                request.with_name('STATUS.json').write_bytes(encoded(status));code=process.wait()
            status.update(exit_code=code,stage='trained' if code==0 else 'failed')
            request.with_name('STATUS.json').write_bytes(encoded(status))
            if code:raise RuntimeError('rescue worker failed; preserve attempt')
            return status
    finally:release_worker(study,key)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','verify','transfer','receive','work'))
    for name in ('study','binaries','qualification','output','request','remote-source'):p.add_argument('--'+name,type=Path)
    p.add_argument('--cell');p.add_argument('--host',choices=HOSTS);p.add_argument('--allow-missing',action='store_true')
    a,remaining=p.parse_known_args()
    if a.mode=='receive':
        if remaining[:1]==['--']:remaining=remaining[1:]
        receive(a.request,remaining);return
    if remaining:p.error('unexpected arguments')
    if a.mode=='register':result=register(a.study,a.cell,a.host,a.binaries,a.qualification,a.output)
    elif a.mode=='transfer':result=transfer(a.request,a.remote_source)
    elif a.mode=='verify':result=verify(json.loads(a.request.read_text()),allow_missing=a.allow_missing)
    else:result=work(a.request)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
