"""Temporarily pin existing native fits, sample perf counters, then restore affinity."""
import argparse,json,os,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha


def identity(pid):
    root=Path('/proc')/str(pid)
    stat=(root/'stat').read_text().rsplit(')',1)[1].split()
    command=(root/'cmdline').read_bytes().split(b'\0')[:-1]
    if stat[0]=='Z' or not command or Path(os.fsdecode(command[0])).name!='zensim_mlp_train':
        raise ValueError('live native trainer required')
    return dict(pid=pid,start_ticks=stat[19],command=[os.fsdecode(a) for a in command])


def require_same(record):
    if identity(record['pid'])!=record:raise ValueError('native trainer identity changed')


def sample(records,output,seconds):
    for record in records:require_same(record)
    command=['sudo','-n','perf','stat','-x',';',
             '-e','task-clock,cycles,instructions,cache-references,cache-misses,cpu-migrations,context-switches',
             '-p',','.join(str(r['pid']) for r in records),'--','sleep',str(seconds)]
    with output.open('x') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    for record in records:require_same(record)
    return command


def probe(study,pids,cpus,output,seconds):
    from shaping_compute_pool import allowed
    from shaping_training_budget import training_limit
    hosts=allowed(study)
    if hosts is not None and socket.gethostname() not in hosts:raise ValueError('worker withdrawn from compute')
    if (len(pids)!=len(cpus) or len(pids)!=len(set(pids)) or len(cpus)!=len(set(cpus))
        or not pids or len(pids)>training_limit(study) or not set(cpus)<=os.sched_getaffinity(0)
        or not 1<=seconds<=20):raise ValueError('distinct trainers and admitted distinct CPUs required')
    records=[identity(pid) for pid in pids]
    for record in records:
        command=record['command']
        if '--out' not in command or not Path(command[command.index('--out')+1]).is_relative_to(study/'runs'):
            raise ValueError('native fit must belong to this study')
    output.mkdir(parents=True,exist_ok=False)
    threads=[]
    for record,cpu in zip(records,cpus):
        for path in (Path('/proc')/str(record['pid'])/'task').iterdir():
            original=sorted(os.sched_getaffinity(int(path.name)))
            if cpu not in original:raise ValueError('cannot broaden an existing thread affinity')
            threads.append(dict(pid=record['pid'],tid=int(path.name),original=original,cpu=cpu))
    (output/'ORIGINAL.json').write_bytes(encoded(dict(records=records,threads=threads)))
    commands=[];changed=[];restored=[]
    try:
        commands.append(sample(records,output/'baseline-perf.csv',seconds))
        for record in records:require_same(record)
        for thread in threads:
            require_same(next(r for r in records if r['pid']==thread['pid']))
            os.sched_setaffinity(thread['tid'],{thread['cpu']});changed.append(thread)
        time.sleep(2)
        commands.append(sample(records,output/'pinned-perf.csv',seconds))
    finally:
        errors=[]
        for thread in reversed(changed):
            try:
                require_same(next(r for r in records if r['pid']==thread['pid']))
                os.sched_setaffinity(thread['tid'],thread['original'])
                if sorted(os.sched_getaffinity(thread['tid']))!=thread['original']:raise ValueError('restore differs')
                restored.append(thread['tid'])
            except ProcessLookupError:errors.append(dict(thread=thread,error='thread ended; no replacement process touched'))
            except Exception as error:errors.append(dict(thread=thread,error=str(error)))
        (output/'RESTORED.json').write_bytes(encoded(dict(restored=restored,errors=errors)))
        if errors:raise ValueError('affinity restore requires inspection; see RESTORED.json')
    time.sleep(2)
    commands.append(sample(records,output/'restored-perf.csv',seconds))
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),
                driver_sha256=sha(__file__),commands=commands,records=records,
                files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
                original_affinities_restored=True,training_started=False,training_replayed=False,
                interpretation='Live scheduling observations, not isolated training throughput or causal proof.')
    (output/'RESULT.json').write_bytes(encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pid',type=int,action='append',required=True)
    p.add_argument('--cpu',type=int,action='append',required=True)
    p.add_argument('--seconds',type=int,default=10)
    a=p.parse_args();print(json.dumps(probe(a.study,a.pid,a.cpu,a.output,a.seconds)))
