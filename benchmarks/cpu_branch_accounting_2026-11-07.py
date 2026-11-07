"""As-run short-window CPU accounting; preserves failed perf counters explicitly."""
from pathlib import Path
import os,json,subprocess,concurrent.futures,time
from datetime import datetime,timezone
rows={}
eval_names={'shaping_supervision.py','shaping_reexport_supervision.py','shaping_evaluation.py','shaping_bootstrap.py','shaping_convergence.py','shaping_outer_evaluation.py','shaping_variant_evaluation.py'}
archive_names={'shaping_archive_supervision.py','dataset_mirror.py','shaping_worker_watch.py','shaping_worker_collect.py','shaping_worker_collection_wave.py','shaping_worker_checkpoint_retention.py','shaping_worker_bundle_retention.py','shaping_checkpoint_retention.py','shaping_worker_input_install.py'}
for p in Path('/proc').glob('[0-9]*'):
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();args=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))[:-1];exe=os.readlink(p/'exe');scripts={Path(a).name for a in args if a.endswith('.py')}
  kind='native_training' if Path(exe).name=='zensim_mlp_train' else 'evaluation' if scripts&eval_names else 'archive_transport' if scripts&archive_names else 'reporting' if 'shaping_progress.py' in scripts else 'orchestration' if any(a.startswith('shaping_') for a in scripts) else None
  rows[int(p.name)]={'ppid':int(fields[1]),'kind':kind}
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
def classify(pid,seen=None):
 r=rows[pid]
 if r['kind']:return r['kind']
 seen=set() if seen is None else seen
 if pid in seen or r['ppid'] not in rows:return None
 seen.add(pid);r['kind']=classify(r['ppid'],seen);return r['kind']
groups={}
for pid in rows:
 kind=classify(pid)
 if kind:groups.setdefault(kind,[]).append(pid)
def collect(item):
 name,pids=item
 cmd=['sudo','-n','perf','stat','-x',';','--no-big-num','-e','task-clock','-p',','.join(map(str,pids)),'--','sleep','10']
 t=datetime.now(timezone.utc).isoformat();r=subprocess.run(cmd,capture_output=True,text=True)
 seconds=None
 for line in r.stderr.splitlines():
  v=line.split(';')
  if len(v)>2 and v[2]=='task-clock':
   try:seconds=float(v[0])/1000
   except ValueError:pass
 return name,{'pids':pids,'started_utc':t,'returncode':r.returncode,'cpu_seconds':seconds,'stdout':r.stdout,'stderr':r.stderr,'command':cmd}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=dict(pool.map(collect,groups.items()))
print(json.dumps({'host':os.uname().nodename,'checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'10-second perf task-clock samples of existing process branches, with child inheritance; no training restarted; native training includes internal statistics. Other orchestration may include newly started training children. Not cumulative study accounting.','groups':results},indent=2))
