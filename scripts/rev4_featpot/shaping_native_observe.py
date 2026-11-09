"""Read scoped native PIDs and snapshot-lock identities without scoring or fitting."""
import argparse,hashlib,json,os,re,socket
from datetime import datetime,timezone
from pathlib import Path


def controllers(scopes,proc=Path('/proc')):
    rows=[]
    for entry in proc.glob('[0-9]*/cmdline'):
        try:
            command=entry.read_bytes().decode().split(chr(0));command=[v for v in command if v]
            if not command or not Path(command[0]).name.startswith('python') or command.count('--study')!=1:continue
            scripts=[Path(v).name for v in command[1:] if v.endswith('.py') and Path(v).name.startswith('shaping_')]
            if len(scripts)!=1:continue
            position=command.index('--study')+1
            if position==len(command):continue
            study=Path(command[position])
            if not any(root==study or root in study.parents for root in scopes):continue
            fields=(entry.parent/'stat').read_text().rsplit(')',1)[1].split()
            rows.append(dict(pid=int(entry.parent.name),state=fields[0],start_ticks=int(fields[19]),
                script=scripts[0],command=command))
        except (FileNotFoundError,ProcessLookupError):pass
    return rows


def observe(scopes,*,proc=Path('/proc'),cgroups=Path('/sys/fs/cgroup')):
    rows=[];disappeared=[]
    for entry in proc.glob('[0-9]*/comm'):
        try:
            if entry.read_text().strip()!='zensim_mlp_trai':continue
            command=(entry.parent/'cmdline').read_bytes().decode().split(chr(0))
            if command.count('--out')!=1:continue
            output=Path(command[command.index('--out')+1])
            if not any(root in output.parents for root in scopes):continue
            run=output.parent.parent if output.parent.name=='fit' else output.parent
            metadata=run/'RUN.json';record=json.loads(metadata.read_text()) if metadata.exists() else {}
            group=(entry.parent/'cgroup').read_text().strip().split(':',2)[2]
            cap=cgroups/group.lstrip('/');fields=(entry.parent/'stat').read_text().rsplit(')',1)[1].split()
            snapshot=[]
            for fd in (entry.parent/'fd').iterdir():
                try:path=Path(os.readlink(fd))
                except FileNotFoundError:continue
                if (path.name=='.lock' and path.parent.parent.name=='shared-training-snapshots'
                    and re.fullmatch('[0-9a-f]{64}',path.parent.name)):snapshot.append(str(path.parent))
            snapshot=sorted(set(snapshot))
            if len(snapshot)>1:raise ValueError('ambiguous native snapshot directory')
            row=dict(pid=int(entry.parent.name),state=fields[0],start_ticks=int(fields[19]),
                run=str(run),cell=record.get('cell'),command=command[:-1] if command[-1]=='' else command,
                checkpoint_count=len(list((output.parent/'checkpoints').glob('ckpt_epoch*.bin'))),
                memory_max=(cap/'memory.max').read_text().strip(),swap_max=(cap/'memory.swap.max').read_text().strip(),
                snapshot_directory=snapshot[0] if snapshot else None)
            phase=output.parent/'RUN.json'
            if output.parent.name in ('fit','pretraining') and phase.exists():
                phase_identity=json.loads(phase.read_text())
                if phase_identity.get('snapshot_phase') is True:
                    row['snapshot_run']=str(output.parent)
            rows.append(row)
        except (FileNotFoundError,ProcessLookupError):disappeared.append(int(entry.parent.name))
    return dict(schema='scoped-native-training-observation-v1',host=socket.gethostname(),
        checked_utc=datetime.now(timezone.utc).isoformat(),native_jobs=rows,disappeared_during_observation=disappeared,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path);p.add_argument('--controllers',action='store_true');a=p.parse_args();result=observe(a.scope)
    if a.controllers:result['controller_jobs']=controllers(a.scope)
    payload=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.output:
        temporary=a.output.with_suffix('.writing');temporary.write_text(payload);temporary.replace(a.output)
    print(payload,end='')
