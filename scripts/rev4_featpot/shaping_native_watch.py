"""Bounded read-only fleet observations; never start, resume or signal a fit."""
import argparse,fcntl,json,os,shlex,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha

HOSTS=('r3500','r3800x','i134','r5600g','r7900x','i270')
MAX_OBSERVATION_BYTES=8*1024**2


def remote(host,observer,scope,expected):
    command=['python3','-B',str(observer),'--controllers','--scope',str(scope)]
    result=subprocess.run(['ssh',host,shlex.join(command)],capture_output=True,text=True,timeout=20,check=True)
    if len(result.stdout.encode())>MAX_OBSERVATION_BYTES:raise ValueError('native observation exceeds bounded metadata allowance')
    value=json.loads(result.stdout)
    if (value['schema']!='scoped-native-training-observation-v1' or value['host']!=host
        or value['driver_sha256']!=expected):raise ValueError('native observation host or source differs')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(value['checked_utc'])).total_seconds()
    if not 0<=age<=60:raise ValueError('native observation is not fresh')
    return value


def sweep(study,observer,output):
    expected=sha(observer);results={};errors={}
    with ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
        futures={host:pool.submit(remote,host,observer,study.parent,expected) for host in HOSTS}
        for host,future in futures.items():
            try:results[host]=future.result()
            except Exception as error:errors[host]=str(error)
    for host,value in results.items():
        path=output/(host+'.json');temporary=path.with_suffix('.writing');temporary.write_bytes(encoded(value))
        if path.exists():path.replace(path.with_suffix('.previous'))
        temporary.replace(path)
    status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),
        stage='observing_with_errors' if errors else 'watching',errors=errors,
        hosts={host:dict(native_jobs=len(value['native_jobs']),controller_jobs=len(value['controller_jobs']),
                        checked_utc=value['checked_utc']) for host,value in results.items()},
        observer_sha256=expected,driver_sha256=sha(__file__),training_processes_changed=False)
    path=output/'NATIVE_SUPERVISION_STATUS.json';temporary=path.with_suffix('.writing')
    temporary.write_bytes(encoded(status));temporary.replace(path);return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','observer','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--poll-seconds',type=int,default=60);a=p.parse_args()
    if a.poll_seconds<30:p.error('poll interval must be at least 30 seconds')
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'OBSERVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            status=sweep(a.study,a.observer,a.output)
            if not a.watch:print(json.dumps(status),flush=True);break
            # Latest and previous metadata stay bounded. Native fit logs/results
            # remain with their existing scientific owners; no per-poll log dump.
            time.sleep(a.poll_seconds)
