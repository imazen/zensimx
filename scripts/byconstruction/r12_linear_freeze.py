"""Freeze R12 packed-byte inventory and native certificates before EVAL reads."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root=Path('/var/tmp/byc-linear-20261104')
    decision=json.loads((root/'dev/DECISION.json').read_text())
    arms=['L1','L2']+(['L3'] if decision['L3_run'] else [])
    models={}
    rows=[]
    for arm in arms:
        for seed in range(3):
            cell=root/f'fits/{arm}-s{seed}'
            bake=cell/('pack_e059/refit/production-f16.bin' if arm=='L3' else 'refit/production-f16.bin')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            c=subprocess.run([str(root/'bin/bake_dial_refit'),'certify-monotone','--bake',str(bake)],
                             capture_output=True,text=True,check=True)
            cert=json.loads(c.stdout)
            if not cert['certified']: raise ValueError('uncertified linear head')
            result=json.loads((cell/'result.json').read_text())
            rows.append(dict(arm=arm,seed=seed,packed_sha256=sha(bake),packed_bytes=bake.stat().st_size,
                             raw_sha256=result['selected_bake_sha256'],
                             certificate=cert,epochs=result['epochs'],selected_epoch=result['selection']['selected_epoch']))
            models[f'{arm}_s{seed}']=str(bake)
    (root/'MODELS.json').write_text(json.dumps(models,indent=2)+'\n')
    (root/'FROZEN_FITS.json').write_text(json.dumps(dict(models=rows,development=decision),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
