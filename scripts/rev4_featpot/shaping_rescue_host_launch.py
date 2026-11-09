"""Fully transfer registered rescues, then launch independent bounded fit owners."""
import argparse,json,shlex,subprocess
from pathlib import Path
from shaping_completion import encoded,sha


def install_assignments(jobs,host,output):
    """The packer uses canonical study assignments, separate from job capsules."""
    paths={}
    for job in jobs:
        path=Path(job['assignment']);pin=sha(path)
        if job.get('assignment_sha256',pin)!=pin:raise ValueError('registered canonical assignment differs')
        row=json.loads(path.read_text())
        if row['host']!=host or row['cells']!=[job['cell']['id']]:raise ValueError('canonical assignment scope differs')
        paths[str(path)]=pin
    program='paths='+repr(paths)+'\n'+'''import hashlib,json
from pathlib import Path
for name,pin in paths.items():
 p=Path(name)
 if p.exists():
  with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
  assert actual==pin,'preserve different canonical assignment: '+name
print(json.dumps(dict(checked=len(paths))))
'''
    subprocess.run(['ssh',host,'python3 -B -'],input=program,text=True,check=True,capture_output=True)
    manifest=output/'ASSIGNMENT_PATHS.txt';manifest.write_bytes(b''.join(p.lstrip('/').encode()+b'\0' for p in paths))
    subprocess.run(['rsync','-rlptR','--omit-dir-times','--no-implied-dirs','--ignore-existing','--from0',
                    '--files-from='+str(manifest),'/',host+':/'],check=True)
    verify=program.replace('if p.exists():','if True:')
    subprocess.run(['ssh',host,'python3 -B -'],input=verify,text=True,check=True,capture_output=True)
    (output/'CANONICAL_ASSIGNMENTS_VERIFIED.json').write_bytes(encoded(dict(host=host,assignment_sha256=paths)))


def launch(waves,source,host,output):
    from shaping_rescue_worker import HOSTS,memory_allowance
    if host not in HOSTS:raise ValueError('approved native worker required')
    jobs=[]
    for wave in waves:
        record=json.loads((wave/'REGISTRATION.json').read_text())
        if record['stage']!='registered':raise ValueError('complete registered rescue wave required')
        if sha(source/'scripts/rev4_featpot/shaping_rescue_worker.py')!=record['source_driver_sha256']:
            raise ValueError('registered worker source differs')
        jobs.extend(j for j in record['jobs'] if j['host']==host)
    if not jobs or len({j['cell']['id'] for j in jobs})!=len(jobs):raise ValueError('distinct host jobs required')
    for job in jobs:
        if sha(job['request'])!=job['request_sha256']:raise ValueError('registered request differs')
    output.mkdir(parents=True,exist_ok=False)
    driver=source/'scripts/rev4_featpot/shaping_rescue_worker.py'
    # Await every transfer and the owner's complete destination verification.
    # A failed transfer starts no scientific job from this host launch.
    with (output/'transfer.log').open('x') as log:
        for job in jobs:
            subprocess.run(['python3','-B',str(driver),'transfer','--request',job['request'],
                            '--remote-source',str(source)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=source)
    install_assignments(jobs,host,output)
    (output/'INPUT_VERIFIED.json').write_bytes(encoded(dict(host=host,requests={j['request']:j['request_sha256'] for j in jobs})))
    rows=[]
    for job in jobs:
        request=Path(job['request']);unit='zensim-rescue-'+job['cell']['id']+'-'+output.name
        scientific=json.loads(request.read_text())
        runtime=Path(scientific['study']).parent/'python-runtime'
        memory_bytes=memory_allowance(scientific)
        memory=str(memory_bytes//1024**3)+'G'
        command=['env','TMPDIR=/home/lilith/tmp','PYTHONPATH='+str(runtime),
                 '/home/lilith/work/claudehints/scripts/run-heavy','--mem',memory,'--jobs','1','--',
                 'python3','-B',str(driver),'work','--request',str(request)]
        remote=['systemd-run','--user','--unit='+unit,'--property=MemoryMax='+memory,'--property=MemorySwapMax=0',
                '--property=Nice=19','--property=IOSchedulingClass=idle',
                '--property=StandardOutput=append:'+str(request.with_name('service.log')),
                '--property=StandardError=append:'+str(request.with_name('service.log')),*command]
        row=dict(host=host,cell=job['cell'],request=job['request'],unit=unit,command=command,
                 memory_max_bytes=memory_bytes,threads_per_fit=1)
        (output/(job['cell']['id']+'-LAUNCH.json')).write_bytes(encoded(row))
        subprocess.run(['ssh',host,shlex.join(remote)],check=True)
        rows.append(row)
    receipt=dict(host=host,rows=rows,inputs_verified_before_any_launch=True,
                 independent_memory_scopes=True,scientific_protocol_unchanged=True)
    (output/'LAUNCH.json').write_bytes(encoded(receipt));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave',type=Path,action='append',required=True)
    for name in ('source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--host',required=True);a=p.parse_args()
    result=launch(a.wave,a.source,a.host,a.output)
    print(json.dumps(dict(host=a.host,submitted_fits=len(result['rows']))),flush=True)
