"""Require complete frozen R14 read receipts before assembling descriptive tables."""
import json
from pathlib import Path
import subprocess
import sys

import r14_transforms as fixed


def main():
    root=fixed.O
    frozen=json.loads((root/'FROZEN_FITS.json').read_text())
    models=json.loads((root/'MODELS.json').read_text())
    release=json.loads((root/'T2_RELEASE.json').read_text())
    if len(models)!=3+int(release['run']) or len(frozen['models'])!=len(models):
        raise ValueError('complete conditional fit inventory required')
    pins=json.loads((root/'reads/MODEL_PINS.json').read_text())
    for row in frozen['models']:
        name=f"{row['arm']}_s{row['seed']}"
        if fixed.owner.sha(models[name])!=row['packed_sha256'] or pins[name]['sha256']!=row['packed_sha256']:
            raise ValueError('read/model freeze differs')
    receipts=list((root/'reads').glob('*.receipt.json'))
    if len(receipts)!=12+4*len(models) or any(json.loads(p.read_text())['rc'] for p in receipts):
        raise ValueError('every registered read must complete successfully')
    readout=json.loads((root/'reads/readout.json').read_text())
    if any('aic' in c.lower() and '2026' in c for columns in readout['per'].values() for c in columns):
        raise ValueError('AIC-2026 quality statistics cannot enter committed tables')
    destination=fixed.X/'benchmarks/r14_input_transform_reads_2026-11-04.md'
    subprocess.run([sys.executable,'-B',str(fixed.X/'scripts/byconstruction/r12_linear_tables.py'),
        '--root',str(root),'--arms',','.join(['F',*[r['arm'] for r in frozen['models']]]),
        '--candidate-seeds','0','--candidate-seed-note',
        'Candidate seeds 1–2 are not measured here. The registration requires seed-0 product/development eligibility and a separate owner cost decision.',
        '--addressability-details','--title','R14 measured read tables','--out',str(destination)],check=True)
    fixed.owner.write(root/'READ_TABLE_RECEIPT.json',dict(
        frozen_sha256=fixed.owner.sha(root/'FROZEN_FITS.json'),
        model_pins_sha256=fixed.owner.sha(root/'reads/MODEL_PINS.json'),
        table_sha256=fixed.owner.sha(destination),
        receipts={str(p.relative_to(root)):fixed.owner.sha(p) for p in sorted(receipts)},
        source_sha256=fixed.owner.sha(__file__)))


if __name__=='__main__':main()
