"""Read completed control-batch resource receipts without rerunning fits."""
import argparse,json,math,re,shlex
from datetime import datetime
from pathlib import Path
from shaping_completion import encoded,sha


def batches(text):
    active=None;out=[]
    for line in text.splitlines():
        if line.startswith('run-heavy: mem-cap='):
            if active is not None:raise ValueError('overlapping or unfinished resource batches')
            command=shlex.split(line.split(' :: ',1)[1])
            if not any(x.endswith('/shaping_controls.py') for x in command):raise ValueError('control command required')
            partition=command[command.index('--partition')+1]
            if ('--tree' in command)==('--linear' in command):raise ValueError('one control family required')
            active=dict(partition=partition,family='extra_trees' if '--tree' in command else 'shaped_additive',started={},completed=[],command=command)
        elif line.startswith('run-heavy: done'):
            if active is None:raise ValueError('unmatched resource receipt')
            m=re.fullmatch(r'run-heavy: done rc=0 (\d+)s \| peak-RSS ([0-9.]+)GiB \| min-avail (\d+)MiB \| peak-load ([0-9.]+)',line)
            if not m:raise ValueError('one successful finite resource receipt required')
            elapsed,rss,available,load=m.groups()
            if not all(math.isfinite(float(x)) for x in m.groups()):raise ValueError('nonfinite resource receipt')
            if not active['completed'] or set(active['started'])!=set(active['completed']):raise ValueError('incomplete control cells')
            active.update(wrapper_elapsed_seconds=int(elapsed),wrapper_peak_rss_gib_reported=float(rss),wrapper_min_available_mib=int(available),wrapper_peak_load=float(load),wrapper_receipt_line=line)
            out.append(active);active=None
        elif active is not None and line.startswith('{'):
            row=json.loads(line)
            if row.get('stage')=='running':
                if row['cell'] in active['started']:raise ValueError('duplicate control start')
                active['started'][row['cell']]=row['pid']
            elif row.get('stage')=='complete':
                if row['cell'] not in active['started'] or row['cell'] in active['completed']:raise ValueError('unmatched or duplicate control completion')
                active['completed'].append(row['cell'])
    if active is not None or not out:raise ValueError('incomplete resource log')
    return out


def report(study,logs,output):
    from shaping_execution import load_registration
    if output.exists():raise ValueError('preserve control resource report')
    _,cells,_=load_registration(study);pins={str(study/'PLAN.json'):sha(study/'PLAN.json')};rows=[];seen=set()
    for log in logs:
        pins[str(log)]=sha(log)
        for batch in batches(log.read_text()):
            expected=[c for c in cells if c['stage']=='inner' and c['family']==batch['family'] and c['partition']==batch['partition']]
            if {c['id'] for c in expected}!=set(batch['completed']):raise ValueError('complete registered inner batch required')
            fits=[]
            for cell in expected:
                if cell['id'] in seen:raise ValueError('duplicate batch cell')
                seen.add(cell['id']);run=study/'runs'/cell['id']/'attempt-001'
                identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
                if identity['cell']!=cell or result['cell']!=cell or result['stage']!='complete' or result['exit_code']!=0:raise ValueError('matching completed control required')
                if identity['plan_sha256']!=pins[str(study/'PLAN.json')] or any(result[k]!=v for k,v in identity.items() if k!='stage'):raise ValueError('control identity drift')
                if identity['pid']!=batch['started'][cell['id']]:raise ValueError('control log PID differs')
                start=datetime.fromisoformat(identity['started_utc']);end=datetime.fromisoformat(result['finished_utc'])
                if start.utcoffset() is None or end.utcoffset() is None or end<start:raise ValueError('invalid control timestamps')
                for p in (run/'RUN.json',run/'RESULT.json'):pins[str(p)]=sha(p)
                fits.append(dict(cell=cell,recorded_utc_interval_seconds=(end-start).total_seconds(),individual_peak_rss_gib=None,
                                 scope='Recorded civil-clock interval includes fit, export, parity and evaluation; shared input preparation precedes RUN. Individual peak RSS was not measured.'))
            rows.append(dict(log=str(log),batch=batch,fits=fits,scope='Whole process batch includes shared preparation, fits, exports and evaluation; wrapper rounding is retained. Batch cost is not divided among fits.'))
    for path,pin in pins.items():
        if sha(path)!=pin:raise ValueError('resource evidence changed')
    result=dict(batches=rows,completed_control_cells=len(seen),source_pins=pins,driver_sha256=sha(__file__),training_replayed=False,
                scientific_eligibility_asserted=False,interpretation='Concurrent-run accounting, not an isolated benchmark. No per-model memory or training-only timing is inferred.')
    output.write_bytes(encoded(result));return dict(batches=len(rows),cells=len(seen),training_replayed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--log',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(report(a.study,a.log,a.output)))
