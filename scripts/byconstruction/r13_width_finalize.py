"""Freeze all R13 packed bakes and derive A's metadata-stripped shipping form.

This completes the released recipe's last step without changing the packed
assessment bakes. Native inspection proves every inference parameter equal.
"""
import json
from pathlib import Path
import subprocess
import sys

import r13_width as owner


def worker():
    root=owner.O
    frozen=json.loads((root/'FROZEN_FITS.json').read_text())
    for row in frozen['models']:
        packed=Path(row['packed_model'])
        if owner.sha(packed)!=row['packed_sha256']:
            raise ValueError('frozen packed bake changed')
        no_repro=packed.with_name('without-repro.bin')
        shipped=packed.with_name('shipped.bin')
        for src,dest,key in ((packed,no_repro,'zentrain.repro'),
                             (no_repro,shipped,'zentrain.sample_coverage')):
            if dest.exists():raise ValueError(f'preserve prior stripped artifact: {dest}')
            subprocess.run([str(root/'bin/bake_dial_refit'),'strip','--in',str(src),
                            '--out',str(dest),'--key',key],check=True)
        head=packed.with_name('shipped-inspection.json')
        subprocess.run([str(owner.HEAD),'--inspect',str(shipped),str(head)],check=True)
        original=root/f"fits/{row['arm']}-s{row['seed']}/HEAD.json"
        if json.loads(head.read_text())!=json.loads(original.read_text()):
            raise ValueError('stripping changed inference parameters')
        row.update(shipped_model=str(shipped),shipped_sha256=owner.sha(shipped),
                   shipped_bytes=shipped.stat().st_size,
                   stripped_inference_inspection='exactly equal to packed native inspection')
    owner.write(root/'FROZEN_FITS.json',frozen)


def main():
    if sys.argv[1:]==['worker']:
        worker()
    elif sys.argv[1:]:
        raise ValueError('unexpected arguments')
    else:
        owner.freeze()
        owner.job('metadata-strip',[sys.executable,'-B',__file__,'worker'])


if __name__=='__main__':main()
