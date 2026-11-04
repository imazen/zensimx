"""Finish the retained T1a pack after its old inspector refused transforms.

The packed bake already exists. Verify its recorded canonical command/hash,
then inspect/certify those same bytes; never rerun calibration or fitting.
"""
import json
from pathlib import Path
import re
import shlex
import sys

import r14_transforms as fixed

owner=fixed.owner
O=fixed.O
ARM='T1a'
CELL=O/'fits/T1a-s0'


def worker():
    dest=CELL/'continuation-r2/pack_e059'
    packed=dest/'refit/production-f16.bin'
    dense,=list((dest/'dense').glob('*.bin'))
    log=(dest/'pack.log').read_text()
    command=shlex.split(log.splitlines()[0].removeprefix('$ '))
    anchor=owner.R/'wide/main/real/cid22_fit.parquet'
    expected=[str(O/'bin/bake_dial_refit'),'pack','--in',str(dense),'--out',str(packed),
        '--dtype','f16','--zerobias-bulk','0','--protect-last','--neg-tail',
        '--identity-knot','100','--tail-extend','2','--no-prune','--anchor',str(anchor),
        '--target-col','human_score','--verify','none']
    digests=re.findall(r'^  sha256 ([0-9a-f]{64})$',log,re.M)
    if command!=expected or digests!=[owner.sha(packed)] or 'K1 K2 K3 PASS' not in log:
        raise ValueError('retained canonical pack command/result differs')
    original=json.loads((O/'SOURCE_PINS.json').read_text())
    if owner.sha(expected[0])!=original['binary_sha256']['bake_dial_refit']:
        raise ValueError('canonical pack instrument differs')
    failed=json.loads((O/'logs/T1a-s0-pack.receipt.json').read_text())
    if failed['rc']==0 or 'derived inputs: feature transforms are not supported' not in (O/'logs/T1a-s0-pack.log').read_text():
        raise ValueError('not the retained inspection-only failure')
    row=dict(packed_model=str(packed),packed_model_sha256=owner.sha(packed),
        dense_model_sha256=owner.sha(dense),TRAIN_calibration_table_sha256=owner.sha(anchor),
        output_stage='e33-identity-knot-tail2',
        deployment_order='final epoch; canonical densify; f16 quantization; TRAIN calibration',
        input_pruning='disabled; preserve derived-input positions',
        recovered_pack_log_sha256=owner.sha(dest/'pack.log'),
        retained_failed_receipt_sha256=owner.sha(O/'logs/T1a-s0-pack.receipt.json'))
    owner.write(CELL/'PACK_STAGE.json',row)
    fixed.inspect_pack(ARM,row)
    if owner.sha(packed)!=digests[0]:raise ValueError('inspection changed packed bytes')


def main():
    if sys.argv[1:]==['worker']:worker();return
    if sys.argv[1:] or (CELL/'FIT.json').exists():raise ValueError('preserve completed fit')
    for label in ('T1a-s0-base','T1a-s0-continuation-r2'):
        if json.loads((O/'logs'/f'{label}.receipt.json').read_text())['rc']:
            raise ValueError('both scientific fit stages must have succeeded')
    label='T1a-s0-pack-inspection-r2'
    owner.job(label,[sys.executable,'-B',__file__,'worker'])
    row=json.loads((CELL/'PACK.json').read_text())
    times=[]
    def seconds(label):
        raw=(O/'logs'/f'{label}.time.txt').read_text()
        elapsed=next(v.split(': ',1)[1].strip() for v in raw.splitlines() if 'Elapsed (wall clock)' in v)
        return sum(float(v)*60**i for i,v in enumerate(reversed(elapsed.split(':'))))
    for stage,job in (('base','T1a-s0-base'),('continuation','T1a-s0-continuation-r2'),('pack',label)):
        times.append(dict(stage=stage,label=job,seconds=seconds(job)))
    times[-1].update(retained_payload_label='T1a-s0-pack',retained_payload_seconds=seconds('T1a-s0-pack'))
    times[-1]['seconds']+=times[-1]['retained_payload_seconds']
    selection=json.loads((O/'ACTIVE_TRAINER.json').read_text())
    native_dir=Path(selection['trainer_dir'])
    native=json.loads((native_dir/'TRAINER_PINS.json').read_text())
    row.update(arm=ARM,seed=0,wall_seconds=sum(v['seconds'] for v in times),stage_times=times,
        native_stages=dict(base=json.loads((CELL/'BASE_NATIVE.json').read_text()),
            continuation=dict(path=str(native_dir/'zensim_mlp_train'),sha256=native['trainer_sha256'],source_commit=native['source_commit'])),
        driver_sha256=owner.sha(__file__),continuation_name='continuation-r2',finished=owner.now(),
        failed_attempts=[dict(label='T1a-s0-continuation',optimizer_steps=0,
            receipt_sha256=owner.sha(O/'logs/T1a-s0-continuation.receipt.json')),
            dict(label='T1a-s0-pack',stage='inspection after successful canonical packing',
                receipt_sha256=owner.sha(O/'logs/T1a-s0-pack.receipt.json'),
                included_in_successful_payload_wall=True)])
    owner.write(CELL/'FIT.json',row)
    print('PACKED '+json.dumps(dict(arm=ARM,wall_seconds=row['wall_seconds'],capacity=row['capacity'])),flush=True)


if __name__=='__main__':main()
