"""Assemble R13 measured tables only after every registered artifact exists."""
import argparse
import json
from pathlib import Path
from statistics import mean

import r13_width as owner

O=owner.O


def load(p):
    return json.loads(Path(p).read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    fits=load(O/'FROZEN_FITS.json')['models']
    if len(fits)!=15: raise ValueError('all fifteen frozen fits required')
    models=load(O/'MODELS.json')
    cap=load(O/'CAPACITY.json')
    dev=load(O/'DEVELOPMENT.json')
    primary=load(O/'reads/PRIMARY.json')
    readout=load(O/'reads/readout.json')
    chroma=load(O/'reads/chroma/G_CHROMA_L.json')['candidates']
    timing=load(O/'heads/SUMMARY.json')
    for p in (O/'reads').glob('*.receipt.json'):
        if load(p)['rc']!=0: raise ValueError(f'failed assessment: {p}')
    # Four shared scoring stages plus every model's steering/dial/NEARID.
    receipts=list((O/'reads').glob('*.receipt.json'))
    if len(receipts)!=72: raise ValueError(f'incomplete assessment receipt inventory: {len(receipts)}/72')
    lines=['# R13: A at proportional hidden widths (2026-11-04)','',
           'Registration `25e77292`; admission/execution `ccdb5c7a`; native read-owner reuse `d6e429d3`.',
           'The control A is the frozen adopted W128 mixF3, seeds 0–2; no A refit.',
           'Owner defines width as pairs = width/4: half the units form max pairs, half retain A’s ordinary threshold role.',
           'W16 and W32 cannot keep 32 pairs, so this comparison is proportional by construction.','',
           'All fifteen fits use A’s 984 inputs, admitted TRAIN mixture and weights, 120 base epochs (fixed epoch 119),',
           'then public_train ×256 continuation with original rehearsal, 60 epochs at 1e-5 (fixed epoch 59).',
           'No EVAL result selects a width, checkpoint or setting. No AIC-2026 assessment statistics are committed.',
           'Packing is canonical densify → f16 with protected final layer → CID22 TRAIN dial, E33 identity/tail2,',
           'with all 984 inputs preserved and a native monotone certificate for each final bake.','',
           '| Width | Seed | Base final dev geomean3 | TRAIN dev MAE | E36 inversions (%) | All-nine MOS error (%) | Chroma level | Dead / residual / live | Largest pair (%) | Inputs 50/90/99% | Head (µs) | Fit wall (s) |',
           '|---|---:|---:|---:|---:|---:|---|---|---:|---|---:|---:|']
    compact=[]
    for width in (16,32,64,128,256,512):
        arm='F' if width==128 else f'W{width}'
        for seed in range(3):
            name=f'{arm}_s{seed}'
            mos=100*mean(v['med'] for v in readout['per'][f'{arm}@{seed}'].values())
            share=100*primary[arm]['seeds'][name]
            if width==128:
                # A's audit already covered the same full TRAIN bank. Classify
                # stored weights with exactly R13's disjoint count definition.
                root=Path('/mnt/v/output/zensim/a-weight-audit-2026-11-04')
                w=load(root/f'audit-A{seed}/weights.json')
                import numpy as np
                first,last=w['layers']; h=first['out_dim']
                matrix=np.array(first['weights'],dtype=np.float32).reshape(-1,h)
                output=np.array(last['weights'],dtype=np.float32)
                dead=(np.max(abs(matrix),axis=0)==0)&(np.array(first['biases'])<=0)
                residual=~dead&(abs(output)<1e-3*max(abs(output)))
                audit=load(root/f'audit-A{seed}/audit.json')
                contrib={u['unit']:abs(float(output[u['unit']]))*u['mean_abs_activation'] for u in audit['hidden_units']}
                shares=[sum(contrib[j] for j in pair)/sum(contrib.values()) for pair in audit['constraints']['max_pairs']]
                effects=sorted((r['mean_abs_delta'] for r in load(root/f'bank-A{seed}/summary.json')['input_attribution']),reverse=True)
                mass=np.cumsum(effects)
                c=dict(dead=int(dead.sum()),residual=int(residual.sum()),live=int((~dead&~residual).sum()),
                       largest_pair_share=max(shares),concentration={str(p):int(np.searchsorted(mass,p*sum(effects))+1) for p in (.5,.9,.99)})
                level='reference seed'
                head='paired controls below'
                wall='not refitted'
            else:
                c=cap[f'cid22/{name}']
                level='pass' if chroma[name]['passes'] else 'fail'
                head=f"{timing[name]['candidate_ns']/1000:.6f}"
                fit=next(r for r in fits if r['arm']==arm and r['seed']==seed)
                wall=f"{fit['wall_seconds']:.2f}"
                compact.append(dict(width=width,seed=seed,dev=dev['base_final_dev_geomean3'][name],
                                    dev_mae=dev['pooled'][name],e36_percent=share,mos_error_percent=mos,
                                    chroma_level=chroma[name]['passes'],capacity=c,
                                    head_ns=timing[name]['candidate_ns'],fit_wall_seconds=fit['wall_seconds']))
            concentration='/'.join(str(c['concentration'][str(p)]) for p in (.5,.9,.99))
            lines.append(f"| {'A (128)' if width==128 else width} | {seed} | {dev['base_final_dev_geomean3'][name]:.6f} | {dev['pooled'][name]:.6f} | {share:.6f} | {mos:.5f} | {level} | {c['dead']}/{c['residual']}/{c['live']} | {100*c['largest_pair_share']:.8f} | {concentration} | {head} | {wall} |")
    lines += ['', 'Base development geomean3 is recorded at the fixed final epoch; it does not select a different epoch.',
              'TRAIN dev MAE pools SafeSyn and CID22 development rows in stored score-point units.',
              'All-nine error averages collection-wise median residual / target p1–p99 span after assessment-only 4PL.',
              'MDID is a training source; the full read tables separately show the eight other collections and common three.',
              'Dead units have exact-zero encoder columns and nonpositive bias. Residual excludes dead units and has',
              '|output coefficient| < 1e-3 of the largest. Largest pair share uses measured mean absolute raw contributions.',
              'Concentration counts damage slots out of 492, jointly intervening on their direct and product paths;',
              'single-slot effects overlap and do not form an additive score decomposition.','',
              '| Width | Matched A head (µs), seeds 0/1/2 | Clean paired rounds /30, seeds 0/1/2 |',
              '|---|---|---|']
    for width in (16,32,64,256,512):
        values=[timing[f'W{width}_s{s}'] for s in range(3)]
        matched=' / '.join(format(v['baseline_ns']/1000,'.6f') for v in values)
        clean=' / '.join(str(v['clean']) for v in values)
        lines.append(f"| {width} | {matched} | {clean} |")
    lines += ['', 'Native head owner: R12’s pinned `Predictor::predict_transformed` paired zenbench instrument,',
              '256 fixed positive logical rows, one call thread, default native dispatch, same clean-round indices.',
              'Includes scaling/network forward; excludes extraction, derived products and dial. These are paired',
              'descriptive means; no pooled confidence interval or full-image speed claim is implied.','',
              'Full EVAL, E36, chroma, steering, C1–C6, identity, NEARID and five-view HDR tables:',
              '[R13 measured reads](r13_hidden_width_reads_2026-11-04.md). HDR uses the registered cube-root:203 banks,',
              'raw historical native scores before the dial, exactly as R12. HDR is report-only; no HDR leg was fitted.',
              'These are reused mix/HC1/R12 assessments, not untouched confirmation. CLIC/CIQA/MMSPG and public_test stay closed.','',
              '## Frozen final bakes','', '| Width | Seed | Packed SHA-256 | Bytes |','|---|---:|---|---:|']
    for row in fits:
        lines.append(f"| {row['arm']} | {row['seed']} | `{row['packed_sha256']}` | {row['packed_bytes']} |")
    lines += ['', 'Complete fit checkpoints, source/binary identities, capacities on CID22 and HDR TRAIN banks,',
              'per-job GNU time and run-heavy receipts, and raw read JSON are preserved under',
              '`/var/tmp/byc-width-20261104/` (verified on dev). Mirror:',
              '`/mnt/tower/output/zensim/r13-width-20261104/`; verification receipt `ARCHIVE_VERIFIED.json`.',
              'Every base/continuation/pack/audit/read job takes and releases the shared heavy lock individually.','',
              '## Per-job resource receipts','']
    for path in sorted((O/'logs').glob('*.log')):
        for line in path.read_text(errors='replace').splitlines():
            if line.startswith('run-heavy: done'):
                lines.append(f"- `{path.name}`: {line}")
    lines += ['', '## Verdict', '', 'Measured interpretation is written after reviewing every table; no model is adopted by this report.', '']
    a.out.write_text('\n'.join(lines))
    owner.write(O/'REPORT_ROWS.json',compact)


if __name__=='__main__':main()
