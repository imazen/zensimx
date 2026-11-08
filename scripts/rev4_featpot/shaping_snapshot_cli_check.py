"""Qualify fresh-directory native crash recovery with tiny frozen controls."""
from __future__ import annotations
import argparse,json,os,re,subprocess,time
from pathlib import Path
from shaping_completion import encoded,sha
from activation_curriculum import replay_controls


def qualify(study,frozen,binaries,output):
    from shaping_reduced_scope import admit_compute
    admit_compute(study)
    output.mkdir(parents=True,exist_ok=False)
    replay_controls(frozen,binaries,output/'neutral')
    original=json.loads((output/'neutral/deep-relu/COMMAND.json').read_text())
    original[original.index('--pairs-per-epoch')+1]='10000'
    environment=os.environ.copy()
    for key in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256',
                'ZENSIM_INPUT_SHAPING_MANIFEST','ZENSIM_INPUT_SHAPING_SHA256',
                'ZENSIM_TRAIN_RESUME','ZENSIM_TRAIN_RESUME_CHECKPOINT_ORIGIN','ZENSIM_TRAIN_SNAPSHOT_ROOT'):
        environment.pop(key,None)
    environment.update(RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    canonical={};prefix_before=None
    for phase in ('reference','interrupted','resumed'):
        root=output/phase;(root/'ckpt').mkdir(parents=True)
        command=list(original)
        for flag,value in [('--out',root/'best.bin'),('--dump-checkpoints-dir',root/'ckpt')]:
            command[command.index(flag)+1]=str(value)
        (root/'COMMAND.json').write_bytes(encoded(command))
        if phase!='reference':environment['ZENSIM_TRAIN_SNAPSHOT_ROOT']=str(output/'store')
        if phase=='resumed':environment.update(ZENSIM_TRAIN_RESUME='1',ZENSIM_TRAIN_RESUME_CHECKPOINT_ORIGIN=str(output/'interrupted/ckpt'))
        with (root/'train.log').open('x') as log:
            process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
            if phase=='interrupted':
                deadline=time.monotonic()+600;boundary=None
                while process.poll() is None and time.monotonic()<deadline:
                    for path in (output/'store').glob('*/state-*.snap'):
                        with path.open('rb') as stream:prefix=stream.read(512)[40:]
                        if b'"epoch":1,"pair":0,' in prefix:boundary=path;break
                    if boundary:break
                    time.sleep(.02)
                if not boundary:
                    if process.poll() is None:process.terminate();process.wait()
                    raise ValueError('mechanical fixture did not expose its committed epoch-one snapshot boundary')
                process.terminate();code=process.wait()
                (root/'INTERRUPTION.json').write_bytes(encoded(dict(pid=process.pid,exit_code=code,
                    snapshot=str(boundary),snapshot_sha256=sha(boundary),boundary='epoch one, pair zero',scientific_cell=False)))
                prefix_before=sha(root/'ckpt/ckpt_epoch000.bin')
                continue
            code=process.wait()
            if code:raise RuntimeError('native CLI qualification failed: '+phase)
        paths=[root/'best.bin',*sorted((root/'ckpt').glob('ckpt_epoch*.bin'))]
        if len(paths)!=3:raise ValueError('two epoch checkpoints and best bake required')
        if phase=='resumed':
            if re.search(r'epoch\s+0\s+\|',(root/'train.log').read_text()):raise ValueError('completed epoch was replayed')
            if sha(output/'interrupted/ckpt/ckpt_epoch000.bin')!=prefix_before:raise ValueError('interrupted prefix was modified')
        for path in paths:
            target=root/(path.stem+'.canonical.bin')
            with (root/(path.stem+'.strip.log')).open('x') as log:
                subprocess.run([str(binaries/'bake_dial_refit'),'strip','--in',str(path),'--out',str(target),'--key','zentrain.repro'],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            canonical[phase,path.name]=sha(target)
    if any(canonical['reference',name]!=canonical['resumed',name] for name in ('best.bin','ckpt_epoch000.bin','ckpt_epoch001.bin')):
        raise ValueError('resumed native CLI changed a canonical bake')
    states=[p for p in (output/'store').glob('*/state-*.snap')]
    if len(states)!=2 or sum(p.stat().st_size for p in states)+max(p.stat().st_size for p in states)>64*1024**2:
        raise ValueError('retention or simultaneous-write budget differs')
    record=dict(schema='zensim-snapshot-cli-control-v2',driver_sha256=sha(__file__),trainer_sha256=sha(binaries/'zensim_mlp_train'),
        frozen_control_sha256=sha(frozen/'RESULT.json'),neutral_control_count=3,pairs_per_epoch=10000,total_epochs=2,
        completed_epochs_replayed=0,original_prefix_preserved=True,all_canonical_bakes_equal=True,
        retained_states=2,retained_state_bytes=sum(p.stat().st_size for p in states),scientific_fits_replayed=False,
        canonical_bakes={name:canonical['resumed',name] for name in ('best.bin','ckpt_epoch000.bin','ckpt_epoch001.bin')})
    (output/'QUALIFICATION.json').write_bytes(encoded(record));return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','frozen','binaries','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(qualify(args.study,args.frozen,args.binaries,args.output)),flush=True)
