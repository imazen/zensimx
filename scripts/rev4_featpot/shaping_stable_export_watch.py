"""Drain terminal primary models into disjoint fixed scalar waves without fit replay."""
import argparse,fcntl,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_stable_pairs import FAMILIES
from shaping_execution import load_registration
from shaping_stable_export_prepare import prepare
from shaping_stable_export_launch import launch
from shaping_stable_export_collect import collect


def completed_collection(wave,q):
    receipts=list(wave.glob('collection-*/COLLECTION_VERIFIED.json'))
    if len(receipts)>1:raise ValueError('ambiguous scalar collection')
    if not receipts:return False
    row=json.loads(receipts[0].read_text())
    if (row['registration_sha256']!=sha(wave/'REGISTRATION.json') or row['models']!=len(q['jobs'])
        or not row['all_transport_sha256_verified'] or not row['all_candidate_provenance_verified']):
        raise ValueError('scalar collection receipt differs')
    return True


def worker_ready(wave,q):
    program='''import json,os
from pathlib import Path
q=REGISTRATION;wave=Path(WAVE);missing=[]
for job in q['jobs']:
 run=Path(job['run']);p=run.parents[2]/'reexport_max_pairs'/job['cell']['id']/run.name
 if not (p/'CONVERGENCE.json').exists():missing.append(job['cell']['id'])
if missing:
 for family in {j['cell']['family'] for j in q['jobs'] if j['cell']['id'] in missing}:
  paths=list((wave/'launch'/family).glob('STATUS.json'))
  if len(paths)!=1:raise ValueError('missing scalar owner status: '+family)
  row=json.loads(paths[0].read_text());pid=row['pid'];p=Path('/proc')/str(pid)
  if not p.exists() or b'shaping_reexport_supervision.py' not in (p/'cmdline').read_bytes() or row['stage']=='sweep_finished':
   raise ValueError('scalar owner stopped with incomplete models: '+family)
print(json.dumps({'ready':not missing,'missing':missing}))
'''
    program='REGISTRATION='+repr(q)+'\nWAVE='+repr(str(wave))+'\n'+program
    p=subprocess.run(['ssh',q['host'],'python3 -B -'],input=program,text=True,capture_output=True,check=True)
    return json.loads(p.stdout)


def required_scope(study,cells,stage):
    if stage=='inner':
        required={c['id'] for c in cells if c['stage']=='inner' and c['family'] in FAMILIES}
        if len(required)!=504:raise ValueError('complete qualified primary coverage required')
        return required,{}
    if stage!='rescue':raise ValueError('registered inner or rescue stage required')
    from shaping_reduced_scope import shortlist
    required=set();pins={}
    for outer in range(3):
        scope,path=shortlist(study,outer);pins[str(path)]=sha(path)
        families=set(scope['roles'].values())&set(FAMILIES)
        selected=[c for c in scope['cells'] if c['stage']=='rescue' and c['family'] in families]
        if len(selected)!=12*len(families) or any(c not in cells for c in selected):
            raise ValueError('complete shortlisted paired rescue scope required')
        required.update(c['id'] for c in selected)
    if not required:raise ValueError('nonempty qualified rescue scope required')
    return required,pins


def watch(args):
    args.output.mkdir(exist_ok=False)
    with (args.output/'OWNER.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return watch_locked(args)


def watch_locked(args):
    waves=[p.parent for p in args.previous_wave];_,cells,_=load_registration(args.study)
    stage=getattr(args,'stage','inner');required,scope_pins=required_scope(args.study,cells,stage)
    while True:
        status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),stage='watching',
            driver_sha256=sha(__file__),waves=list(map(str,waves)),required_models=len(required),training_replayed=False,
            cell_stage=stage,scope_sha256=scope_pins)
        try:
            if any(sha(name)!=pin for name,pin in scope_pins.items()):raise ValueError('registered rescue shortlist changed')
            assigned=set();waiting=[];registrations=[(wave,json.loads((wave/'REGISTRATION.json').read_text())) for wave in waves]
            for wave,q in registrations:
                ids={j['cell']['id'] for j in q['jobs']}
                if assigned&ids or not ids<=required:raise ValueError('overlapping or unrelated scalar waves')
                assigned|=ids
            for wave,q in registrations:
                if completed_collection(wave,q):continue
                if worker_ready(wave,q)['ready']:
                    attempt=1
                    while (wave/f'collection-{attempt}').exists():attempt+=1
                    collect(wave,args.source,wave/f'collection-{attempt}')
                else:waiting.append(str(wave))
            status.update(assigned_models=len(assigned),waiting_waves=waiting)
            if not waiting:
                trained={cid for cid in required if any(json.loads(p.read_text()).get('stage')=='trained'
                    for p in (args.study/'runs'/cid).glob('attempt-*/RESULT.json'))}
                status.update(trained_models=len(trained))
                if trained-assigned:
                    prefix='stable-export-wave' if stage=='inner' else 'stable-rescue-export-wave'
                    wave=args.study.parent/f'{prefix}-{len(waves)+1:04}-20261108'
                    options={} if stage=='inner' else dict(stage=stage,cell_ids=required,evaluation_inputs=args.evaluation_inputs)
                    prepare(args.study,args.source,args.host,list(FAMILIES),[p/'REGISTRATION.json' for p in waves],wave,**options)
                    # Record new immutable assignment before submission, so setup
                    # failure cannot make this condition silently eligible again.
                    waves.append(wave);status['waves']=list(map(str,waves));status['stage']='starting_fixed_wave'
                    (args.output/'STATUS.json').write_bytes(encoded(status));launch(wave)
                elif assigned==required:status['stage']='complete'
                else:status['stage']='waiting_for_remaining_native_fits'
        except Exception as error:
            status.update(stage='failed_with_preserved_evidence',error=str(error))
            (args.output/'STATUS.json').write_bytes(encoded(status));raise
        (args.output/'STATUS.json').write_bytes(encoded(status));print(json.dumps(status),flush=True)
        if status['stage']=='complete':return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--previous-wave',type=Path,action='append',default=[])
    p.add_argument('--stage',choices=('inner','rescue'),default='inner')
    p.add_argument('--evaluation-inputs',type=Path)
    watch(p.parse_args())
