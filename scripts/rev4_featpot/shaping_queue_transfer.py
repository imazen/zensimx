"""Transfer only unopened requests from an initial concurrent rescue wave."""
from __future__ import annotations
import argparse,fcntl,json,os,signal,time
from contextlib import ExitStack
from pathlib import Path
from shaping_completion import encoded,sha


def transfer(plan_path):
    plan=json.loads(plan_path.read_text());manifest=Path(plan['wave']);wave=json.loads(manifest.read_text())
    status=json.loads(manifest.with_name('WAVE_STATUS.json').read_text());pid=status['pid']
    if sha(manifest)!=plan['wave_sha256'] or status['manifest_sha256']!=sha(manifest):raise ValueError('wave identity drift')
    if status['completed'] or status['failed']:raise ValueError('only initial waves with all original worker slots occupied or waiting are supported')
    protected=set(wave['requests'][:wave['workers']]);chosen=plan['requests']
    if not chosen or len(set(chosen))!=len(chosen) or not set(chosen)<=set(wave['requests']) or set(chosen)&protected:
        raise ValueError('transfer must exclude every initial active or waiting worker')
    proc=Path('/proc')/str(pid);arguments=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if (not any(Path(x).name=='shaping_rescue_worker.py' for x in arguments) or 'wave' not in arguments
        or arguments.count('--request')!=1 or arguments[arguments.index('--request')+1]!=str(manifest)):
        raise ValueError('wave process identity differs')
    first=json.loads(Path(wave['requests'][0]).read_text());study=Path(first['study'])
    if any((study/'runs'/json.loads(Path(p).read_text())['cell']['id']/'attempt-001/RESULT.json').exists() for p in protected):
        raise ValueError('an initial worker has finished; preserve and reconcile wave explicitly')
    destination=Path(plan['output'])
    if destination.exists():raise ValueError('preserve previous handoff')
    record=dict(schema='shaping-unopened-request-transfer-v1',plan_sha256=sha(plan_path),wave_sha256=sha(manifest),
                wave_pid=pid,files={},active_fits_interrupted=False)
    # Acquire the same short transactions used by admission. Stopping only the
    # Python queue while holding these locks cannot freeze it as a lock owner.
    # Native controllers and trainers are separate processes and keep running.
    with ExitStack() as stack:
        for relative in ('cpu-admission/transaction.lock','storage-admission/transaction.lock'):
            lock=stack.enter_context((study/relative).open('a'));fcntl.flock(lock,fcntl.LOCK_EX)
        os.kill(pid,signal.SIGSTOP)
        try:
            deadline=time.monotonic()+2
            while True:
                states=[p.read_text().rsplit(')',1)[1].split()[0] for p in (proc/'task').glob('*/stat')]
                if states and all(state=='T' for state in states):break
                if time.monotonic()>deadline:raise ValueError('queue did not reach its metadata handoff boundary')
                time.sleep(.01)
            claims=json.loads((study/'cpu-admission/CLAIMS.json').read_text())
            for name in chosen:
                path=Path(name);row=json.loads(path.read_text());cell=row['cell']['id'];saved=path.with_suffix('.transferred')
                if cell in claims or (study/'runs'/cell).exists() or saved.exists() or sha(path)!=wave['request_sha256'][name]:
                    raise ValueError('a proposed request has started, changed or transferred already')
                record['files'][name]=dict(preserved_path=str(saved),sha256=sha(path),cell=cell)
            with destination.open('xb') as out:out.write(encoded(record));out.flush();os.fsync(out.fileno())
            for name,row in record['files'].items():Path(name).rename(row['preserved_path'])
        finally:os.kill(pid,signal.SIGCONT)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();print(json.dumps(transfer(a.plan)),flush=True)
