"""Supply existing registered inner evaluation inputs without fitting or scoring."""
import argparse,json,subprocess
from pathlib import Path
from shaping_completion import sha,encoded

MAX_BYTES=256*1024**2
PARTS={f'inner-{outer}-{inner}' for outer in range(3) for inner in range(2)}


def register(study,output):
    from shaping_execution import load_registration
    from shaped_linear_study import OLD,PREP
    plan,_,parts=load_registration(study);files={};views=[]
    def add(path,pin=None):
        path=Path(path);actual=sha(path)
        if pin is not None and actual!=pin:raise ValueError('registered evaluation input drift')
        files[str(path)]=dict(sha256=actual,bytes=path.stat().st_size)
    for partition in sorted(PARTS):
        directory=study/'evaluation_views'/partition;receipt=directory/'RECEIPT.json';value=json.loads(receipt.read_text())
        if value['indices']!=parts[partition]['test']:raise ValueError('inner validation membership differs')
        add(receipt)
        for group in value['groups']:
            for name,pin in group['file_sha256'].items():add(name,pin)
        probes=directory/'PROBES.json';value=json.loads(probes.read_text());add(probes);add(value['path'],value['sha256']);views.append(str(directory))
    original=Path(plan['original_plan']).parent
    for path in [OLD/'data/y.npy',OLD/'data/pop.npy',original/'data/component.npy',original/'data/family.npy',PREP]:add(path,plan['input_pins'][str(path)])
    record=dict(schema='shaping-inner-evaluation-input-closure-v1',plan_sha256=sha(study/'PLAN.json'),files=files,views=views,outer_or_final_views_included=False)
    if sum(row['bytes'] for row in files.values())>MAX_BYTES:raise ValueError('evaluation input closure exceeds 256 MiB')
    with output.open('xb') as stream:stream.write(encoded(record))
    return record


def validate(study,record):
    if record['schema']!='shaping-inner-evaluation-input-closure-v1' or record['outer_or_final_views_included'] or record['plan_sha256']!=sha(study/'PLAN.json'):
        raise ValueError('evaluation input closure registration differs')
    views={Path(p) for p in record['views']}
    if views!={study/'evaluation_views'/p for p in PARTS}:raise ValueError('only all registered inner views may be transported')
    if sum(r['bytes'] for r in record['files'].values())>MAX_BYTES:raise ValueError('evaluation input closure exceeds 256 MiB')
    plan=json.loads((study/'PLAN.json').read_text())
    for name,row in record['files'].items():
        path=Path(name)
        if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('noncanonical evaluation input path')
        if not any(view in path.parents for view in views) and plan['input_pins'].get(name)!=row['sha256']:
            raise ValueError('unregistered evaluation label array')
    return views


def preflight(study,record,receipt):
    views=validate(study,record);backups=[]
    # The caller stops its evaluator first. Refuse any remaining work process
    # before moving an incomplete cache directory; native training stays active.
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=p.read_bytes().decode(errors='replace').split(chr(0))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if '--work' in args and any(a.endswith('shaping_evaluation_worker.py') for a in args) and str(study) in args:
            raise ValueError('evaluation process still active; preserve its inputs')
    for view in sorted(views):
        if view.exists() and not (view/'RECEIPT.json').exists():
            suffix=0
            while True:
                backup=view.with_name(view.name+f'.before-input-closure-{suffix}.bak')
                if not backup.exists():break
                suffix+=1
            view.rename(backup);backups.append(dict(original=str(view),preserved=str(backup)))
    for name,row in record['files'].items():
        path=Path(name)
        if path.exists() and (path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']):
            raise ValueError('preserve existing different evaluation input: '+name)
    result=dict(stage='ready_for_missing_input_copy',incomplete_views_preserved=backups,files=len(record['files']),driver_sha256=sha(__file__))
    with receipt.open('xb') as stream:stream.write(encoded(result))
    return result


def verify(study,record):
    validate(study,record)
    for name,row in record['files'].items():
        path=Path(name)
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('evaluation input copy drift: '+name)
    return dict(stage='all_registered_inner_inputs_verified',files=len(record['files']),bytes=sum(r['bytes'] for r in record['files'].values()),driver_sha256=sha(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','preflight','verify','receive'))
    for name in ('study','manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--receipt',type=Path);a,arguments=p.parse_known_args()
    if a.mode!='receive' and arguments:p.error('unexpected arguments')
    if a.mode=='preflight' and a.receipt is None:p.error('--receipt is required for preflight')
    if a.mode=='register':value=register(a.study,a.manifest)
    elif a.mode=='preflight':value=preflight(a.study,json.loads(a.manifest.read_text()),a.receipt)
    elif a.mode=='receive':
        from shaping_storage import reservation
        record=json.loads(a.manifest.read_text());validate(a.study,record)
        arguments=arguments[1:] if arguments[:1]==['--'] else arguments
        if arguments[:1]!=['--server']:raise ValueError('rsync receiver arguments required')
        with reservation(a.study,a.manifest.parent/'input-admission',30*1024**3,bytes_required=MAX_BYTES):
            subprocess.run(['rsync',*arguments],check=True);value=verify(a.study,record)
    else:value=verify(a.study,json.loads(a.manifest.read_text()))
    if a.mode!='receive':print(json.dumps(value),flush=True)
