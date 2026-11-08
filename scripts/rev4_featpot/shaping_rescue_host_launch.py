"""Fully transfer registered rescues, then launch independent bounded fit owners."""
import argparse,json,shlex,subprocess
from pathlib import Path
from shaping_completion import encoded,sha


def launch(waves,source,host,output):
    from shaping_rescue_worker import HOSTS
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
    (output/'INPUT_VERIFIED.json').write_bytes(encoded(dict(host=host,requests={j['request']:j['request_sha256'] for j in jobs})))
    rows=[]
    for job in jobs:
        request=Path(job['request']);unit='zensim-rescue-'+job['cell']['id']+'-'+output.name
        runtime=Path(json.loads(request.read_text())['study']).parent/'python-runtime'
        command=['env','TMPDIR=/home/lilith/tmp','PYTHONPATH='+str(runtime),
                 '/home/lilith/work/claudehints/scripts/run-heavy','--mem','2G','--jobs','1','--',
                 'python3','-B',str(driver),'work','--request',str(request)]
        remote=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0',
                '--property=Nice=19','--property=IOSchedulingClass=idle',
                '--property=StandardOutput=append:'+str(request.with_name('service.log')),
                '--property=StandardError=append:'+str(request.with_name('service.log')),*command]
        row=dict(host=host,cell=job['cell'],request=job['request'],unit=unit,command=command,
                 memory_max_bytes=2*1024**3,threads_per_fit=1)
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
