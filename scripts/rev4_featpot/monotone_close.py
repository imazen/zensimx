#!/usr/bin/env python3
"""Close the registered study without converting failed scientific gates to passes."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from activation_curriculum import sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--figures',type=Path,required=True)
    args=parser.parse_args();root=args.root
    if (root/'CLOSURE.json').exists():raise ValueError('preserve existing closure')
    totals={};pins={};failed=[]
    for stage,required in [('screen',24),('long',24),('training',66),('leaky',6),('isolated-order',9)]:
        plan=json.loads((root/stage/'PLAN.json').read_text())
        terminal=json.loads((root/stage/'SUMMARY.json').read_text())
        if terminal['required']!=required or terminal['complete']+len(terminal['failures'])!=required:
            raise ValueError('unfinished registered stage: '+stage)
        failures={(r['arm'],r['seed_index']) for r in terminal['failures']}
        for arm in plan['arms']:
            for i in range(3):
                cell=root/stage/arm/f's{i}';log=(cell/'train.log').read_text()
                if not re.search(r'run-heavy: done rc=0\b',log):raise ValueError('unfinished native fit')
                bake=cell/f"ckpt/ckpt_epoch{plan['epochs']-1:03d}.bin"
                probes=json.loads((cell/'PROBES.json').read_text())
                passed=probes['identity']==100. and not probes['violations_over_1e_5'] and not probes['ceiling_violations']
                if passed==((arm,i) in failures):raise ValueError('gate ledger mismatch')
                if not passed:failed.append(f'{stage}/{arm}/s{i}')
                if passed and json.loads((cell/'RESULT.json').read_text())['bake_sha256']!=sha(bake):
                    raise ValueError('final native bake drift')
                pins[str(bake.relative_to(root))]=sha(bake)
        totals[stage]=dict(fits_finished=required,original_feature_gates_passed=terminal['complete'],
            original_feature_gates_failed=len(failures))
        pins[f'{stage}/PLAN.json']=sha(root/stage/'PLAN.json')
        pins[f'{stage}/SUMMARY.json']=sha(root/stage/'SUMMARY.json')
    expected={f'long/{arm}/s{i}' for arm in ('shared_spline','single_ramps') for i in range(3)}
    if set(failed)!=expected:raise ValueError('unexpected scientific gate outcomes')
    for directory,required in [('training-qualified',66),('leaky-qualified',6),('isolated-order-qualified',9)]:
        path=root/directory/'QUALIFICATION.json';q=json.loads(path.read_text())
        if q['required']!=required or q['complete']!=required:raise ValueError('incomplete event qualification')
        for cell,r in q['cells'].items():
            if not r['retained_sections_identical'] or sha(root/directory/cell/'final.bin')!=r['qualified_sha256']:
                raise ValueError('qualification model drift')
        pins[str(path.relative_to(root))]=sha(path)
    stages=json.loads((root/'stages/SUMMARY.json').read_text())
    if stages['complete']!=5 or stages['exact_final_replay_cells']!=15 or stages['checkpoint_coverage']!=1800:
        raise ValueError('incomplete retained-checkpoint replay')
    if len(list((root/'stage-replay').glob('*/s*/ckpt/ckpt_epoch*.bin')))!=1800:
        raise ValueError('retained checkpoint count drift')
    pins['stages/SUMMARY.json']=sha(root/'stages/SUMMARY.json')
    repairs=[]
    for directory in ('screen-export-audit-v2','long-export-audit'):
        path=root/directory/'AUDIT.json';audit=json.loads(path.read_text())
        plan=json.loads((root/directory/'PLAN.json').read_text())
        for cell,receipt in audit['repair_receipts'].items():
            if audit['feature_gates'][cell]!='PASS' or receipt['fitting_steps']!=0:
                raise ValueError('unresolved repaired spline gate')
            model=root/directory/cell/f"ckpt/ckpt_epoch{plan['epochs']-1:03d}.bin"
            if sha(model)!=receipt['output_sha256']:raise ValueError('repaired spline bake drift')
            repairs.append(f'{directory}/{cell}')
        pins[str(path.relative_to(root))]=sha(path)
    if len(repairs)!=6:raise ValueError('missing repaired spline exports')
    for name in ('screen-bootstrap','training-bootstrap','leaky-bootstrap','isolated-order-bootstrap','long-export-bootstrap'):
        path=root/name/'SOURCE_BOOTSTRAP.json';b=json.loads(path.read_text())
        for value in b['populations'].values():
            if value['references']>1 and value['draws']!=2000:raise ValueError('bootstrap budget drift')
        pins[str(path.relative_to(root))]=sha(path)
    figure_receipt=args.figures/'MIRROR_VERIFIED.json';mirror=json.loads(figure_receipt.read_text())
    if not mirror['all_artifact_sha256_verified']:raise ValueError('figures not mirrored')
    write(root/'CLOSURE.json',dict(schema='monotone-study-closure-v1',stages=totals,
        registered_fits_finished=129,original_feature_gates_passed=123,original_feature_gate_failures=failed,
        exact_checkpoint_replays=15,retained_checkpoints=1800,repaired_spline_exports=repairs,
        source_commit=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip(),
        evidence_sha256=pins,figure_mirror_receipt_sha256=sha(figure_receipt),
        remaining=['three original ramp exports fail the unchanged native feature gate',
            'softplus/integrated-Gaussian/rational/linear+tanh are scalar-only; exact native operators absent',
            'pixel-domain, strict global f32 monotonicity and cross-target shipping qualification not established'],
        interpretation='completed registered experiment matrix; failed candidates preserved, not accepted; no serving promotion'))
    print(json.dumps(dict(registered_fits_finished=129,original_gates_passed=123,
        closure_sha256=sha(root/'CLOSURE.json'))))


if __name__=='__main__':main()
