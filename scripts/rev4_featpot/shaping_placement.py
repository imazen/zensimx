"""Read live worker backlog for placement; existing admission remains authoritative."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess


def observation(study, host, capacity):
    program = 'study=' + repr(str(study)) + '\ncapacity=' + repr(capacity) + '\n' + '''
from pathlib import Path
import fcntl,json,os,shutil,socket
root=Path(study);claims_path=root/'cpu-admission/CLAIMS.json'
with (claims_path.parent/'transaction.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 claims=json.loads(claims_path.read_text()) if claims_path.exists() else {}
native=[];waiters=set()
for process in Path('/proc').glob('[0-9]*'):
 try:
  command=(process/'cmdline').read_bytes();argv=[v.decode() for v in command.split(b'\\0') if v]
  if (process/'comm').read_text().strip()=='zensim_mlp_trai':native.append(command)
  scripts=[i for i,v in enumerate(argv) if Path(v).name=='shaping_rescue_worker.py']
  if len(scripts)!=1 or argv[scripts[0]+1:scripts[0]+2]!=['work'] or argv.count('--request')!=1:continue
  q=json.loads(Path(argv[argv.index('--request')+1]).read_text())
  if q['study']==study and q['host']==socket.gethostname():waiters.add(q['cell']['id'])
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
own=sum(any(key.encode() in command for key in claims) for command in native)
unrepresented_claims=max(0,len(claims)-own)
unclaimed_waiters=len(waiters-set(claims))
cpus=len(os.sched_getaffinity(0))
if capacity>=cpus:raise ValueError('worker CPU affinity differs from approved placement budget')
memory=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
free=shutil.disk_usage(root).free
print(json.dumps(dict(host=socket.gethostname(),native=len(native),claims=len(claims),
 unclaimed_waiters=unclaimed_waiters,cpus=cpus,memory_available_bytes=memory,ssd_free_bytes=free,
 occupied=len(native)+unrepresented_claims+unclaimed_waiters)))
'''
    result = subprocess.run(['ssh', host, 'python3 -B -'], input=program, text=True,
                            capture_output=True, check=True, timeout=30)
    row = json.loads(result.stdout)
    if row['host'] != host:
        raise ValueError('placement observation host differs')
    return row


def loads(study, capacities):
    from shaping_compute_pool import allowed
    hosts=allowed(study)
    capacities={host:cap for host,cap in capacities.items() if hosts is None or host in hosts}
    if not capacities:raise ValueError('no currently authorized placement worker')
    with ThreadPoolExecutor(max_workers=len(capacities)) as pool:
        futures = {host: pool.submit(observation, study, host, cap) for host, cap in capacities.items()}
        observed = {host: future.result() for host, future in futures.items()}
    # This is a placement hint. Every fit still acquires the original CPU/RAM/SSD
    # guards, with their stronger accounting, after its immutable registration.
    eligible = {host: row for host, row in observed.items()
                if row['memory_available_bytes'] >= 6 * 1024**3 and row['ssd_free_bytes'] >= 36 * 1024**3}
    if not eligible:
        raise ValueError('no worker has placement memory and SSD headroom')
    return {host: row['occupied'] if host in eligible else 1000000 for host, row in observed.items()}, observed
