"""Recover the four service-killed, zero-epoch worker attempts under owner direction."""
from __future__ import annotations
import argparse,json,os,re,shutil,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha

CELLS=('bb69102081104506a747','5491e6c23514ed0c59ae',
       '1d3496c70f00c64f9f84','83e41509517d6ffb591a')
INSTRUCTION='If external forces - or a disk or memory leak - caused this, they can be resumed where they left off. But sort out what is killing resources and fix it first'
CAUSE='systemd user manager stopped after SSH session exit; lingering was disabled'
NAME='ZERO_EPOCH_RECOVERY_AUTHORIZATION.json'


def register(study,direction):
    message=json.loads(direction.read_text())
    if message['user_instruction']!=INSTRUCTION or message['conditional_resumption_authorized'] is not True:
        raise ValueError('owner conditional recovery instruction required')
    if subprocess.check_output(['loginctl','show-user',str(os.getuid()),'-p','Linger','--value'],text=True).strip()!='yes':
        raise ValueError('fix worker lingering before recovery')
    if socket.gethostname()!='r5600g':raise ValueError('original approved worker required')
    records={}
    for cell in CELLS:
        run=study/'runs'/cell/'attempt-001'
        old=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
        if (result['stage']!='interrupted' or result['reason']!=CAUSE or result['saved_checkpoints']!=0
            or old['execution_host']!='r5600g' or old['cell']!=result['cell']
            or list((run/'checkpoints').glob('ckpt_epoch*.bin'))
            or re.search(r'epoch\s+\d+\s+\|',(run/'train.log').read_text())):
            raise ValueError('only verified external zero-epoch interruption is recoverable here')
        records[cell]=dict(cell=old['cell'],previous_attempt='attempt-001',new_attempt='attempt-002',
            old_run_sha256=sha(run/'RUN.json'),old_result_sha256=sha(run/'RESULT.json'),
            old_result=result,shaping_sha256=sha(run/'SHAPING.json'),keep_sha256=sha(run/'keep.txt'),
            original_build_sha256=old['build_sha256'])
    record=dict(schema='shaping-zero-epoch-recovery-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        owner_instruction=INSTRUCTION,direction_sha256=sha(direction),host='r5600g',
        plan_sha256=sha(study/'PLAN.json'),cells=records,completed_epochs_replayed=0,
        recovery_boundary='original seeded initialization; no epoch checkpoint exists',
        resource_fix='user lingering enabled and verified; persistent worker service required')
    with (study/NAME).open('xb') as stream:stream.write(encoded(record))
    return record


def authorize_attempt(study,cell,attempt):
    path=study/NAME
    if not path.exists():raise ValueError('later worker attempts require explicit reconciliation')
    record=json.loads(path.read_text())
    if (record['schema']!='shaping-zero-epoch-recovery-v1' or record['owner_instruction']!=INSTRUCTION
        or record['completed_epochs_replayed']!=0 or set(record['cells'])!=set(CELLS)
        or record['plan_sha256']!=sha(study/'PLAN.json') or attempt!='attempt-002' or cell not in CELLS):
        raise ValueError('zero-epoch recovery authorization differs')
    row=record['cells'][cell];old=row['old_result']
    if (row['cell']['id']!=cell or row['new_attempt']!=attempt or old['stage']!='interrupted'
        or old['saved_checkpoints']!=0 or old['reason']!=CAUSE):
        raise ValueError('original interruption is not authorized for recovery')
    return row,sha(path)


def verify_attempt(study,run,attempt):
    if attempt=='attempt-001':return
    row,pin=authorize_attempt(study,run['cell']['id'],attempt)
    if (run['cell']!=row['cell'] or run.get('recovery_authorization_sha256')!=pin
        or run.get('recovery_completed_epochs_replayed')!=0
        or run['build_sha256']!=row['original_build_sha256']):
        raise ValueError('recovered run differs from its explicit authorization')


def recover(args):
    from shaping_reduced_scope import admit_cell
    from shaping_rescue_dispatch import reserve_worker,release_worker,bind_worker
    from shaping_storage import reservation
    row,pin=authorize_attempt(args.study,args.cell,'attempt-002');admit_cell(args.study,row['cell'])
    if socket.gethostname()!='r5600g':raise ValueError('original worker required')
    if subprocess.check_output(['loginctl','show-user',str(os.getuid()),'-p','Linger','--value'],text=True).strip()!='yes':
        raise ValueError('worker service lifecycle fix not active')
    old=args.study/'runs'/args.cell/'attempt-001';dest=old.parent/'attempt-002'
    if dest.exists():raise ValueError('preserve existing recovery attempt')
    for name,key in [('RUN.json','old_run_sha256'),('RESULT.json','old_result_sha256'),('SHAPING.json','shaping_sha256'),('keep.txt','keep_sha256')]:
        if sha(old/name)!=row[key]:raise ValueError('original recovery evidence drift')
    if list((old/'checkpoints').glob('ckpt_epoch*.bin')):raise ValueError('saved epoch requires full-state recovery, not an initial-state retry')
    original=json.loads((old/'RUN.json').read_text());command=list(original['command'])
    build=args.study.parent/'study-bin/BUILD.json'
    if sha(build)!=row['original_build_sha256']:raise ValueError('frozen binary registration differs')
    for name,expected in json.loads(build.read_text())['binary_sha256'].items():
        if sha(build.parent/name)!=expected:raise ValueError('frozen executable drift')
    while not reserve_worker(args.study,args.cell)['can_start']:time.sleep(10)
    try:
        minimum=json.loads((args.study/'PLAN.json').read_text())['minimum_ssd_free_bytes']
        with reservation(args.study,dest,minimum):
            dest.mkdir();(dest/'checkpoints').mkdir()
            for name in ('keep.txt','SHAPING.json','SHAPING.provenance.json'):shutil.copyfile(old/name,dest/name)
            for flag,name in [('--out','best.bin'),('--keep-features','keep.txt'),('--dump-checkpoints-dir','checkpoints')]:
                if command.count(flag)!=1:raise ValueError('ambiguous original output path')
                command[command.index(flag)+1]=str(dest/name)
            env=os.environ.copy()
            for key in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256','ZENSIM_INPUT_SHAPING_MANIFEST','ZENSIM_INPUT_SHAPING_SHA256'):env.pop(key,None)
            env.update(ZENSIM_INPUT_SHAPING_MANIFEST=str(dest/'SHAPING.json'),ZENSIM_INPUT_SHAPING_SHA256=row['shaping_sha256'])
            with (dest/'train.log').open('x') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                run=dict(original,command=command,pid=process.pid,driver_sha256=sha(__file__),
                     started_utc=datetime.now(timezone.utc).isoformat(),stage='running',
                     recovery_authorization_sha256=pin,recovery_completed_epochs_replayed=0)
                (dest/'RUN.json').write_bytes(encoded(run));bind_worker(args.study,args.cell,process.pid)
                code=process.wait()
            result=dict(run,stage='trained' if code==0 else 'failed',exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
            if code==0:
                expected={f'ckpt_epoch{e:03}.bin' for e in range(run['epochs'])}
                if {p.name for p in (dest/'checkpoints').glob('ckpt_epoch*.bin')}!=expected:raise ValueError('recovery checkpoint coverage differs')
                result['bake_sha256']=sha(dest/'checkpoints'/f"ckpt_epoch{run['epochs']-1:03}.bin")
            (dest/'RESULT.json').write_bytes(encoded(result))
            if code:raise RuntimeError('external recovery training failed')
            return result
    finally:release_worker(args.study,args.cell)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--register-direction',type=Path);p.add_argument('--cell',choices=CELLS)
    args=p.parse_args()
    if args.register_direction:print(json.dumps(register(args.study,args.register_direction)),flush=True)
    elif args.cell:print(json.dumps(recover(args)),flush=True)
    else:p.error('register direction or recover one authorized cell')


if __name__=='__main__':main()
