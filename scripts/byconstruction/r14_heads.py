"""Run R12's frozen head instrument on i270 with the shared one-hour lock wait."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import r14_transforms as fixed

owner=fixed.owner
O=fixed.O
REMOTE='tmp/zensim-r14-heads-20261104'
BINARY_SHA='aa409c8e689323f7ed0f8ca4f07fa34565d673efa18e26806f98572677af3a33'


def worker():
    import os
    import platform
    import socket
    root=Path(sys.argv[2]).resolve()
    if socket.gethostname()!='i270':raise ValueError('R14 head timing is authorized on i270 only')
    cpu_names=sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')})
    (root/'HOST.json').write_text(json.dumps(dict(host=socket.gethostname(),cpu_models=cpu_names,
        kernel=platform.release(),machine=platform.machine(),affinity=sorted(os.sched_getaffinity(0)),
        checked=time.time(),role='observed inside the acquired timing resource scope'),indent=2)+'\n')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    os.environ.update(RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',ZENSIM_MAX_TIER='v3',TMPDIR=str(root/'scratch'))
    (root/'scratch').mkdir(exist_ok=True)
    binary=root/'prune_forward_bench'
    if owner_hash(binary)!=BINARY_SHA:raise ValueError('R12 head instrument changed')
    for name,row in manifest['models'].items():
        model=root/row['file']
        control=root/'A0.bin'
        if owner_hash(model)!=row['sha256'] or owner_hash(control)!=manifest['A0_sha256']:
            raise ValueError('frozen head pair changed')
        (root/'STAGE.json').write_text(json.dumps(dict(name=name,pid=os.getpid(),started=time.time()))+'\n')
        with (root/f'timing-{name}.log').open('x') as log:
            subprocess.run([str(binary),'--heads',str(control),str(model),'256',str(root/f'timing-{name}.json')],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/'COMPLETE.json').write_text(json.dumps(dict(finished=time.time(),models=list(manifest['models'])))+'\n')


def owner_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status():
    import os
    root=O/'heads'
    active=json.loads((root/'ACTIVE.json').read_text())
    sys.path.insert(0,str(fixed.X/'scripts/rev4_featpot'))
    from run_processes import process_tree
    live=any(p['pid']==active['pid'] and p['argv']==active['argv'] and p['state'] not in ('Z','X')
             for p in process_tree(active['pid']))
    probe='''import json,os,subprocess
from pathlib import Path
r=Path.home()/"tmp/zensim-r14-heads-20261104"
d={name:json.loads((r/name).read_text()) if (r/name).exists() else None
   for name in ("STAGE.json","COMPLETE.json","HOST.json")}
p=subprocess.check_output(["lslocks","-J","-o","PID,COMMAND,PATH,BLOCKER"],text=True)
d["locks"]=[v for v in json.loads(p)["locks"] if v.get("path","").endswith("/heavy.lock")]
stage=d["STAGE.json"]
d["worker_pid_exists"]=bool(stage and Path("/proc",str(stage["pid"])).exists())
print(json.dumps(d))'''
    remote=json.loads(subprocess.check_output(['ssh','i270','python3 -c '+shlex.quote(probe)],text=True))
    execution=json.loads((root/'EXECUTION.json').read_text()) if (root/'EXECUTION.json').exists() else None
    result=dict(checked=time.time(),local_coordinator_live=live,remote=remote,execution=execution,
        started=active['started'],wait_limit_seconds=active['wait_limit_seconds'])
    result['state']=execution['status'] if execution else ('timing/preflight' if remote['worker_pid_exists'] else
        'waiting for shared lock' if live else 'exited; inspect execution/logs')
    owner.write(root/'SUPERVISION.json',result)
    print(json.dumps(result,indent=2))


def main():
    if sys.argv[1:2]==['worker']:
        worker();return
    if sys.argv[1:]==['status']:
        status();return
    if sys.argv[1:]:raise ValueError('unexpected arguments')
    root=O/'heads'
    root.mkdir(exist_ok=False)
    frozen=json.loads((O/'FROZEN_FITS.json').read_text())
    models=json.loads((O/'MODELS.json').read_text())
    release=json.loads((O/'T2_RELEASE.json').read_text())
    expected=3+int(release['run'])
    if len(models)!=expected or len(frozen['models'])!=expected:
        raise ValueError('complete frozen T1/T2 inventory required')
    import shutil
    instrument=Path('/var/tmp/byc-linear-20261104/extra/heads/bin/prune_forward_bench')
    if owner.sha(instrument)!=BINARY_SHA:raise ValueError('frozen R12 binary differs')
    shutil.copy2(instrument,root/'prune_forward_bench')
    control=O/'controls/A0.bin'
    # The frozen controls manifest owns the source identity; no control refit.
    if not control.exists():
        control=owner.H/'tmp/e36test/r10/runs/mixF3-s0/pack_e059/refit/production-f16.bin'
    if owner.sha(control)!='185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088':
        raise ValueError('frozen A0 differs')
    shutil.copy2(control,root/'A0.bin')
    manifest=dict(binary_sha256=BINARY_SHA,A0_sha256=owner.sha(control),models={})
    for row in frozen['models']:
        name=f"{row['arm']}_s{row['seed']}"
        if owner.sha(models[name])!=row['packed_sha256']:raise ValueError('head model differs')
        filename=name+'.bin'
        shutil.copy2(models[name],root/filename)
        manifest['models'][name]=dict(file=filename,sha256=row['packed_sha256'])
    owner.write(root/'MANIFEST.json',manifest)
    shutil.copy2(__file__,root/'worker.py')
    # The remote worker uses only the standard library. Avoid importing repo owners there.
    worker_source=(root/'worker.py').read_text().replace('import r14_transforms as fixed','')
    worker_source=worker_source.replace('owner=fixed.owner\nO=fixed.O','')
    (root/'worker.py').write_text(worker_source)
    subprocess.run(['ssh','i270',f'test ! -e {REMOTE} && mkdir -p {REMOTE}'],check=True)
    subprocess.run(['scp','-p',*[str(p) for p in root.iterdir() if p.is_file()],f'i270:{REMOTE}/'],check=True)
    command=f'flock -w 3600 -E 75 ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- python3 ~/{REMOTE}/worker.py worker ~/{REMOTE}'
    started=time.time()
    with (root/'remote.log').open('x') as log:
        p=subprocess.Popen(['ssh','i270',command],stdout=log,stderr=subprocess.STDOUT)
        owner.write(root/'ACTIVE.json',dict(pid=p.pid,argv=['ssh','i270',command],started=started,
            wait_limit_seconds=3600,host='i270'))
        rc=p.wait()
    subprocess.run(['scp','-p','-r',f'i270:{REMOTE}/*',str(root)+'/'],check=True)
    owner.write(root/'EXECUTION.json',dict(rc=rc,started=started,finished=time.time(),
        status='not measured: shared lock busy for one hour' if rc==75 else 'complete' if rc==0 else 'failed; inspect retained logs'))
    if rc==75:return
    if rc:raise subprocess.CalledProcessError(rc,['ssh','i270',command])
    import r14_head_summary
    r14_head_summary.timing(O)


if __name__=='__main__':main()
