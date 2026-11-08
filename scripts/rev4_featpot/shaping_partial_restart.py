"""Explicit two-cell restart exception; preserve and compare every old checkpoint."""
from __future__ import annotations
import argparse,json,subprocess,time,shutil
from pathlib import Path
from shaping_completion import encoded,sha

CELLS={'c1b9fabc3c27b2b14fe7':19,'fd66858f8667dbad12c0':219}
NAME='PARTIAL_RESTART_AUTHORIZATION.json'


def register(study,proposal,approval):
    proposed=json.loads(proposal.read_text());approved=json.loads(approval.read_text())
    if (approved['answer']!='Restart only these two' or approved['proposal_sha256']!=sha(proposal)
        or approved['target_host']!='i134' or not approved['old_attempts_preserved']
        or not approved['original_binary_seeds_data_and_banks'] or not approved['compare_all_available_checkpoints']):
        raise ValueError('exact owner restart approval required')
    rows=proposed['scope']
    if len(rows)!=2 or {r['cell']['id'] for r in rows}!=set(CELLS):raise ValueError('only two approved cells')
    records={}
    for row in rows:
        cell=row['cell']['id'];old=study/'runs'/cell/'attempt-001'
        run=json.loads((old/'RUN.json').read_text());result=json.loads((old/'RESULT.json').read_text())
        checkpoints={p.name:sha(p) for p in (old/'checkpoints').glob('ckpt_epoch*.bin')}
        if (row['saved_epochs']!=CELLS[cell] or row['original_epochs']!=240 or row['new_epochs']!=240
            or row['target_host']!='i134' or row['new_attempt']!='attempt-002'
            or run['cell']!=row['cell'] or run['command']!=row['original_command']
            or run['build_sha256']!=row['frozen_build_sha256'] or result['stage']!='interrupted'
            or sha(old/'RUN.json')!=row['original_run_sha256'] or sha(old/'RESULT.json')!=row['original_result_sha256']
            or sha(old/'SHAPING.json')!=row['original_bank_sha256'] or checkpoints!=row['checkpoint_sha256']
            or set(checkpoints)!={f'ckpt_epoch{e:03}.bin' for e in range(CELLS[cell])}):
            raise ValueError('approved original partial evidence differs')
        inputs={run['command'][i+1].split(':')[1] for i,a in enumerate(run['command']) if a=='--group'}
        records[cell]=dict(input_sha256={name:sha(name) for name in inputs},cell=row['cell'],old_run_sha256=sha(old/'RUN.json'),old_result_sha256=sha(old/'RESULT.json'),
            shaping_sha256=sha(old/'SHAPING.json'),keep_sha256=sha(old/'keep.txt'),
            original_build_sha256=run['build_sha256'],completed_epochs_replayed=CELLS[cell],
            checkpoint_sha256=checkpoints,new_attempt='attempt-002',target_host='i134')
    record=dict(schema='shaping-partial-restart-v1',owner_answer=approved['answer'],
        proposal_sha256=sha(proposal),approval_sha256=sha(approval),plan_sha256=sha(study/'PLAN.json'),cells=records)
    with (study/NAME).open('xb') as stream:stream.write(encoded(record))
    return record


def authorize(study,cell,attempt):
    path=study/NAME
    if not path.exists():raise ValueError('partial restart requires explicit reconciliation and approval')
    record=json.loads(path.read_text())
    if (record['schema']!='shaping-partial-restart-v1' or record['owner_answer']!='Restart only these two'
        or set(record['cells'])!=set(CELLS) or record['plan_sha256']!=sha(study/'PLAN.json')
        or cell not in CELLS or attempt!='attempt-002'):
        raise ValueError('partial restart authorization differs')
    row=record['cells'][cell]
    if (row['cell']['id']!=cell or row['target_host']!='i134' or row['new_attempt']!=attempt
        or row['completed_epochs_replayed']!=CELLS[cell]):raise ValueError('partial restart scope differs')
    return row,sha(path)


def compare_available(old,dest,row,binary):
    """Use native post-write log receipts; preserve canonical bytes and mismatches."""
    directory=dest/'prefix_comparison_v2';directory.mkdir(exist_ok=True)
    path=dest/'PREFIX_COMPARISON_V2.json'
    record=json.loads(path.read_text()) if path.exists() else dict(expected=len(row['checkpoint_sha256']),compared={})
    receipts=set()
    for line in (dest/'train.log').read_text().splitlines():
        marker='checkpoint dump: '
        if marker not in line:continue
        receipt=json.loads(line.split(marker,1)[1]);receipts.add(Path(receipt['path']).name)
    for name,pin in row['checkpoint_sha256'].items():
        if name in record['compared'] or name not in receipts:continue
        if sha(old/'checkpoints'/name)!=pin:raise ValueError('old checkpoint drift')
        outputs=[]
        for label,source in [('original',old/'checkpoints'/name),('restart',dest/'checkpoints'/name)]:
            target=directory/(label+'-'+name)
            with (directory/(label+'-'+name+'.log')).open('x') as log:
                result=subprocess.run([str(binary),'strip','--in',str(source),'--out',str(target),'--key','zentrain.repro'],
                    stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                message=(directory/(label+'-'+name+'.log')).read_text()
                if result.returncode==1 and message.startswith('error: metadata key \"zentrain.repro\" not present in ') and not target.exists():
                    shutil.copyfile(source,target) # Interrupted bakes have no final provenance stamp.
                else:raise RuntimeError('native canonical comparison failed: '+message)
            outputs.append(sha(target))
        record['compared'][name]=dict(original_sha256=outputs[0],restart_sha256=outputs[1],equal=outputs[0]==outputs[1])
        path.write_bytes(encoded(record))
    return record


def supervise(process,old,dest,row,binary):
    while True:
        try:compare_available(old,dest,row,binary)
        except Exception as error:
            (dest/'PREFIX_COMPARISON_ERROR.json').write_bytes(encoded(dict(error=str(error))))
            code=process.wait()
            return code or 65
        try:code=process.wait(timeout=5);break
        except subprocess.TimeoutExpired:pass
    record=compare_available(old,dest,row,binary)
    if code==0 and (len(record['compared'])!=record['expected'] or not all(r['equal'] for r in record['compared'].values())):
        return 65 # Native budget completed; failed qualification is explicit.
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','proposal','approval'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();print(json.dumps(register(args.study,args.proposal,args.approval)),flush=True)
