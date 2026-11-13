"""Finite collection/evaluation/archive owner for the 15 registered new fits.

Runs on an eligible worker. Never starts, restarts, retunes or selects a fit.
Checks real native processes and terminal records; stops after verified closure.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from shaping_completion import encoded,sha


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.new');temporary.write_bytes(encoded(value));temporary.replace(path)


def census_program(root):
    return '''import json,os,re,shutil,subprocess
from pathlib import Path
r=Path(ROOT);rows=[];native=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if not (p/'comm').read_text().strip().startswith('zensim_mlp_trai'):continue
  argv=(p/'cmdline').read_bytes().split(bytes([0]));argv=[a.decode(errors='replace') for a in argv if a]
  native.append(dict(pid=int(p.name),argv=argv))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
for d in sorted((r/'runs').glob('*')):
 if not d.is_dir():continue
 a=d.name.startswith('a-');terminal=d/('RUN.json' if a else 'RESULT.json');v={}
 if terminal.exists():v=json.loads(terminal.read_text())
 live=[p['pid'] for p in native if any(str(d) in a for a in p['argv'])]
 stage=v.get('stage','preparing');trained=stage=='trained'
 logs=list(d.glob('*/writes/train.log')) if a else [d/'train.log'];epoch=None
 for log in logs:
  if not log.exists():continue
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-65536));text=f.read().decode(errors='replace')
  matches=re.findall(r'(?<![A-Za-z_])epoch(?:=|\\s+)([0-9]+)',text)
  if matches:epoch=int(matches[-1])
 if epoch is not None and not 0<=epoch<(120 if a else 240):raise ValueError('invalid epoch parsed from training log')
 unit='zensim-followup-'+d.name+'.service'
 active=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip()
 if live and stage=='preparing':stage='training'
 rows.append(dict(id=d.name,stage=stage,trained=trained,native_pids=live,last_epoch=epoch,unit_state=active,
  terminal_sha256=__import__('hashlib').sha256(terminal.read_bytes()).hexdigest() if terminal.exists() else None))
avail=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
print(json.dumps(dict(host=os.uname().nodename,rows=rows,native_processes=len(native),memory_available_bytes=avail,
 disk_free_bytes=shutil.disk_usage(r).free,cpus=len(os.sched_getaffinity(0)))))
'''.replace('ROOT',repr(str(root)))


def census(host,root):
    command=[sys.executable,'-B','-'] if host=='local' else ['ssh','-o','ConnectTimeout=8',host,'python3','-B','-']
    p=subprocess.run(command,input=census_program(root),capture_output=True,text=True,timeout=45)
    if p.returncode:raise RuntimeError('census failed '+host+': '+p.stderr)
    return json.loads(p.stdout)


def summary(root,results):
    import statistics
    human={};a={}
    baseline=json.loads((root/'a-evaluation-inputs/THREE_SEED_COMPARISON.json').read_text())
    for r in results:
        if r['recipe']=='human_ssim2':
            point=r['metrics']['selection_score'];control=r['matched_control']['selection_score']
            human.setdefault(r['arm'],[]).append(dict(seed=r['seed'],human_composite=point,matched_control=control,
                delta=None if point is None or control is None else point-control,rank_tails=r['rank_tails']))
        else:
            for old in baseline['rows']:
                dataset,pop=old['dataset'],old['population']
                point=r['datasets'][dataset][pop]['statistics']['srocc_signed']
                control=old['points']['unshaped'][r['seed']]
                a.setdefault(r['arm'],{}).setdefault(dataset+'|'+pop,[]).append(dict(seed=r['seed'],srocc=point,
                    matched_unshaped=control,delta=point-control,rank_p99=r['datasets'][dataset][pop]['rank_p99']))
    for arms in (human,):
        for arm,rows in list(arms.items()):
            arms[arm]=dict(paired_seeds=sorted(rows,key=lambda r:r['seed']),
                mean_delta=statistics.mean(r['delta'] for r in rows) if all(r['delta'] is not None for r in rows) else None,
                seed_sd=statistics.stdev(r['human_composite'] for r in rows) if len(rows)==3 and all(r['human_composite'] is not None for r in rows) else None)
    for arm,populations in a.items():
        for pop,rows in list(populations.items()):
            populations[pop]=dict(paired_seeds=sorted(rows,key=lambda r:r['seed']),mean_delta=statistics.mean(r['delta'] for r in rows),
                seed_sd=statistics.stdev(r['srocc'] for r in rows) if len(rows)==3 else None)
    return dict(stage='scientific_results_complete',fits=15,evaluations=len(results),human=human,a_recipe=a,
        scope='Development screen only. Human recipe uses one previously exposed source fold. A uses previously exposed CSIQ/LIVE; no fresh confirmation.',
        final_holdout_read=False,serving_model_promoted=False)


def close(root,study,archive,results):
    if subprocess.check_output(['findmnt','-no','FSTYPE','-T',str(archive)],text=True).strip() not in ('nfs','nfs4'):
        raise ValueError('durable archive must resolve to the verified NAS mount')
    closure=root/'closure';closure.mkdir(exist_ok=True)
    write(closure/'RESULTS.json',summary(root,results))
    (closure/'REPLAY.md').write_text('''# Replaying the saved scalar panels

Run `sha256sum -c SHA256SUMS` before extraction. Extract the `.tar.zst`
bundles into one directory with `tar --zstd -xf <bundle>`.

The native executable bundle is Linux x86-64. Each evaluation carries the
actual scalar prediction/target inputs for its pooled and within-source panels:

```
native-bin-v2/panel --batch evaluations/<model>/replay-batch.tsv --stats srocc --raw-errors
native-bin-v2/panel --batch evaluations/<A-model>/live/replay-batch.tsv --stats srocc --raw-errors
```

The adjacent `.labels.json` maps compact labels to their original populations
and sources. No new fit, calibration, image extraction or Python code is needed
for these panels. The JSON results retain rank tails and the paired-seed
comparison. Human composite scores average pooled/within-source SROCC, equally
over the four human populations; CID22 SSIM2 stays separate.

`REUSED_INPUT_CLOSURES.json` identifies the previously verified training-data
archives. `human-inputs/` contains the exact commands, shape/optimizer manifests,
row weights and input hashes. `a-inputs/` contains the registered A commands,
shared/fixed templates and bank provenance; its original recipe and runtime
are in the named original-A shared archive. Native executable/source identities
are in `native-bin-v2/BUILD.json`. All 15 conditions use fixed final epochs.
This is a development screen; no fresh confirmation or serving promotion occurred.
''')
    # Quiescent fit/evaluation trees are packed once. Do not include the live
    # supervisor or duplicate the previously archived shared training arrays.
    for name in ('runs','evaluations','native-source','native-bin-v2','human-inputs','a-inputs','a-evaluation-inputs','qualification','runtime'):
        bundle=closure/(name+'.tar.zst')
        if bundle.exists():continue
        if not (root/name).exists():raise ValueError('closure input absent: '+name)
        pending=bundle.with_name(bundle.name+'.pending')
        if pending.exists():raise ValueError('preserve interrupted packing: '+str(pending))
        cmd=['tar','--sort=name','--exclude=target','--exclude=.git','--exclude=.jj','--exclude=.workongoing',
            '--exclude=__pycache__','--exclude=*.pyc','-I','zstd -T1 -3','-cf',str(pending),'-C',str(root),name]
        with (root/(name+'-pack.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        pending.replace(bundle)
    # Snapshots are bounded stores (two retained generations plus a pending
    # write). Copy only after no registered native trainer remains anywhere.
    for name in ('human-snapshots','a-snapshots'):
        if (root/name).exists() and not (closure/(name+'.tar.zst')).exists():
            pending=closure/(name+'.tar.zst.pending')
            if pending.exists():raise ValueError('preserve interrupted snapshot packing')
            with (root/(name+'-pack.log')).open('x') as log:
                subprocess.run(['tar','--sort=name','-I','zstd -T1 -3','-cf',str(pending),
                    '-C',str(root),name],stdout=log,stderr=subprocess.STDOUT,check=True)
            pending.replace(closure/(name+'.tar.zst'))
    if not (closure/'MIRROR_VERIFIED.json').exists():
        owner=Path(__file__).resolve().parents[1]/'byconstruction/dataset_mirror.py'
        with (root/'closure-archive.log').open('a') as log:
            subprocess.run([sys.executable,'-B',str(owner),'--root',str(closure),'--nfs','--cold-readback',
                '--artifact-kind','training','--destination',str(archive/'closure')],stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((closure/'MIRROR_VERIFIED.json').read_text())
    if not receipt['all_artifact_sha256_verified'] or sha(closure/'SHA256SUMS')!=sha(archive/'closure/SHA256SUMS'):
        raise ValueError('final archive verification incomplete')
    return dict(destination=str(archive/'closure'),receipt_sha256=sha(closure/'MIRROR_VERIFIED.json'),manifest_sha256=sha(closure/'SHA256SUMS'))


def run(root,study,archive,interval):
    lock=(root/'FINISHER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected={f'human-{arm}-s{s}':('r3800x' if s==1 else 'local') for s in range(3)
        for arm in ('hard_case','negative_abs_init','negative_softplus')}
    expected.update({f'a-{arm}-s{s}':'r5600g' for s in range(3) for arm in ('slow_joint','common_frozen')})
    output=root/'evaluations';output.mkdir(exist_ok=True);collected=root/'collection';collected.mkdir(exist_ok=True)
    bins=root/'native-bin-v2';heavy=Path.home()/'work/claudehints/scripts/run-heavy'
    env=os.environ.copy();env.update(TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1')
    while True:
        state=dict(stage='supervising',pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),required_fits=15,
            fitted=0,evaluated=0,failures=[],hosts={},rows=[])
        observations={}
        for host in sorted(set(expected.values())):
            try:
                value=census(host,root);state['hosts'][host]=value
                observations.update({(host,r['id']):r for r in value['rows']})
            except Exception as e:state['failures'].append(dict(host=host,error=str(e)))
        results=[]
        for identifier,host in expected.items():
            row=observations.get((host,identifier));evaluation=output/identifier
            if row is None:
                state['failures'].append(dict(id=identifier,error='registered run missing from live host census'));continue
            state['rows'].append(dict(host=host,**row))
            if not row['trained']:
                if not row['native_pids'] and row['unit_state']!='active':
                    state['failures'].append(dict(id=identifier,error='no native trainer, no live owner and no completed record'))
                continue
            state['fitted']+=1;write(root/'STATUS.json',state)
            if row['native_pids'] or row['unit_state']=='active':
                continue
            try:
                run_dir=root/'runs'/identifier
                if host!='local' and not (collected/(identifier+'.json')).exists():
                    subprocess.run(['rsync','-a','--checksum','--partial-dir=.rsync-partial',host+':'+str(run_dir)+'/',str(run_dir)+'/'],check=True)
                    write(collected/(identifier+'.json'),dict(host=host,terminal_sha256=row['terminal_sha256']))
                journal=run_dir/'owner-journal.log'
                if not journal.exists():
                    command=['journalctl','--user','-u','zensim-followup-'+identifier+'.service','--no-pager','-o','cat']
                    if host!='local':command=['ssh',host,*command]
                    p=subprocess.run(command,capture_output=True,text=True,check=True)
                    if identifier.startswith('a-') and 'run-heavy: done rc=0' not in p.stdout:
                        raise ValueError('completed A resource-wrapper receipt is missing')
                    journal.write_text(p.stdout)
                if (evaluation/'ERROR.json').exists():raise RuntimeError('retained evaluation failure needs repair')
                if not (evaluation/'RESULT.json').exists():
                    kind='a' if identifier.startswith('a-') else 'human'
                    argv=[str(heavy),'--mem','2G','--jobs','1','--',sys.executable,'-B',str(Path(__file__).with_name('shaping_followup_evaluate.py')),
                        kind,'--run',str(run_dir),'--output',str(evaluation),'--binaries',str(bins),'--study',str(study),
                        '--a-inputs',str(root/'a-evaluation-inputs')]
                    with (output/(identifier+'.log')).open('x') as log:
                        code=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env=env).returncode
                    if code:
                        evaluation.mkdir(exist_ok=True);write(evaluation/'ERROR.json',dict(exit_code=code,log=str(output/(identifier+'.log'))))
                        raise RuntimeError('fixed-checkpoint evaluation failed; preserved')
                result=json.loads((evaluation/'RESULT.json').read_text())
                if result['stage']!='complete' or result['final_holdout_read'] or result['id']!=identifier:
                    raise ValueError('evaluation identity/scope/stage differs')
                results.append(result);state['evaluated']+=1
            except Exception as e:state['failures'].append(dict(id=identifier,error=str(e)))
        if state['evaluated']==15 and not state['failures']:
            # Collect the frozen A provenance and bounded snapshots once.
            for relative in ('a-snapshots','a-inputs/templates'):
                dest=root/relative;dest.mkdir(parents=True,exist_ok=True)
                subprocess.run(['rsync','-a','--checksum','r5600g:'+str(root/relative)+'/',str(dest)+'/'],check=True)
            subprocess.run(['rsync','-a','--include=/*.json','--include=/code/***','--include=/templates/***','--exclude=*',
                'r5600g:'+str(root/'a-inputs')+'/',str(root/'a-inputs')+'/'],check=True)
            write(root/'RESULTS.json',summary(root,results))
            state['stage']='archiving';write(root/'STATUS.json',state)
            state['archive']=close(root,study,archive,results);state['stage']='complete';write(root/'STATUS.json',state)
            print(json.dumps(state),flush=True);return
        write(root/'STATUS.json',state)
        print(json.dumps({k:state[k] for k in ('checked_utc','fitted','evaluated','failures')}),flush=True)
        if state['failures']:
            # Keep successful fits/evaluations; expose a real failure instead of
            # leaving an idle owner reporting progress forever.
            state['stage']='needs_repair';write(root/'STATUS.json',state)
            raise RuntimeError('finite finisher found failures; see STATUS.json')
        time.sleep(interval)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','study','archive'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--interval',type=int,default=120);a=p.parse_args()
    if not 60<=a.interval<=600:p.error('metadata interval must be 60..600 seconds')
    try:
        run(a.root,a.study,a.archive,a.interval)
    except Exception as e:
        path=a.root/'STATUS.json';state=json.loads(path.read_text()) if path.exists() else {}
        state.update(stage='needs_repair',checked_utc=datetime.now(timezone.utc).isoformat(),terminal_error=str(e))
        write(path,state)
        raise
