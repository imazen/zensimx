"""Summarize existing epoch logs across a recorded resource-change boundary."""
import argparse,json,re,statistics
from datetime import datetime
from pathlib import Path


def windows(points,original_applied,transition_started,priority_applied,limit=5):
    if not original_applied<transition_started<=priority_applied:raise ValueError('ordered resource timestamps required')
    selected={'original':[],'priority':[]}
    for left,right in zip(points,points[1:]):
        if right['epoch']!=left['epoch']+1 or right['elapsed']<=left['elapsed'] or right['mtime']<left['mtime']:
            raise ValueError('complete ordered epoch intervals required')
        row=dict(from_epoch=left['epoch'],to_epoch=right['epoch'],seconds=right['elapsed']-left['elapsed'])
        if original_applied<=left['mtime'] and right['mtime']<transition_started:selected['original'].append(row)
        elif priority_applied<=left['mtime']:selected['priority'].append(row)
    return {name:dict(intervals=rows[-limit:],n=len(rows[-limit:]),median_seconds=statistics.median(r['seconds'] for r in rows[-limit:]) if rows else None)
            for name,rows in selected.items()}


def summarize(run,original,transition,priority,output):
    times=[datetime.fromisoformat(json.loads(p.read_text())['checked_utc']).timestamp() for p in (original,transition,priority)]
    text=(run/'train.log').read_text();points=[]
    for epoch,elapsed in re.findall(r'^\s*epoch\s+(\d+)\s+\|[^\n]*\bt=([0-9.]+)s',text,re.M):
        model=run/'checkpoints'/f'ckpt_epoch{int(epoch):03}.bin'
        points.append(dict(epoch=int(epoch),elapsed=float(elapsed),mtime=model.stat().st_mtime))
    result=dict(run=str(run),resource_timestamps=times,points=points,windows=windows(points,*times),
                interpretation='Observed completed epoch intervals from one continuing fit. Epochs, model state and competing work differ; not an isolated benchmark or a causal speedup estimate.')
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','original','transition','priority','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(summarize(a.run,a.original,a.transition,a.priority,a.output)['windows']))
