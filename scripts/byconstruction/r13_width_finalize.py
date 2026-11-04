"""Freeze all R13 packed bakes and derive A's metadata-stripped shipping form.

This completes the released recipe's last step without changing the packed
assessment bakes. Native inspections bind layers, scalers, derived inputs and dial.
"""
import json
from pathlib import Path
import subprocess
import sys

import r13_width as owner


def worker():
    root=owner.O
    frozen=json.loads((root/'FROZEN_FITS.json').read_text())
    control=owner.H/'tmp/e36test/r10/runs/mixF3-s0/pack_e059/refit/production-f16.bin'
    control_head=root/'FROZEN_CONTROL_HEAD.json'
    if control_head.exists():raise ValueError('preserve prior finalization control')
    subprocess.run([str(owner.HEAD),'--inspect',str(control),str(control_head)],check=True)
    declaration=json.loads(control_head.read_text())['derived_inputs']
    frozen['control_sha256']=owner.sha(control)
    for row in frozen['models']:
        packed=Path(row['packed_model'])
        if owner.sha(packed)!=row['packed_sha256']:
            raise ValueError('frozen packed bake changed')
        original=root/f"fits/{row['arm']}-s{row['seed']}/HEAD.json"
        inspection=json.loads(original.read_text())
        first,last=inspection['layers']
        if (inspection['derived_inputs']!=declaration or first['in_dim']!=984
                or first['out_dim']!=int(row['arm'][1:])
                or last['in_dim']!=first['out_dim'] or last['out_dim']!=1):
            raise ValueError('R13 input declaration or registered layer dimensions differ')
        no_repro=packed.with_name('without-repro.bin')
        shipped=packed.with_name('shipped.bin')
        for src,dest,key in ((packed,no_repro,'zentrain.repro'),
                             (no_repro,shipped,'zentrain.sample_coverage')):
            if dest.exists():raise ValueError(f'preserve prior stripped artifact: {dest}')
            subprocess.run([str(root/'bin/bake_dial_refit'),'strip','--in',str(src),
                            '--out',str(dest),'--key',key],check=True)
        head=packed.with_name('shipped-inspection.json')
        subprocess.run([str(owner.HEAD),'--inspect',str(shipped),str(head)],check=True)
        if head.read_bytes()!=original.read_bytes():
            raise ValueError('stripping changed inference parameters')
        row.update(shipped_model=str(shipped),shipped_sha256=owner.sha(shipped),
                   shipped_bytes=shipped.stat().st_size,
                   stripped_inference_inspection='exactly equal to packed native inspection')
    frozen['finalizer_source_sha256']=owner.sha(__file__)
    owner.write(root/'FROZEN_FITS.json',frozen)


def main():
    if sys.argv[1:]==['worker']:
        worker()
    elif sys.argv[1:]:
        raise ValueError('unexpected arguments')
    else:
        receipt=owner.O/'logs/metadata-strip.receipt.json'
        if receipt.exists():
            if json.loads(receipt.read_text())['rc']!=0:
                raise ValueError('preserve failed metadata finalization for investigation')
            frozen=json.loads((owner.O/'FROZEN_FITS.json').read_text())
            if len(frozen['models'])!=15 or any(owner.sha(r['shipped_model'])!=r['shipped_sha256'] for r in frozen['models']):
                raise ValueError('completed frozen shipping inventory differs')
            return
        owner.freeze()
        owner.job('metadata-strip',[sys.executable,'-B',__file__,'worker'])


if __name__=='__main__':main()
