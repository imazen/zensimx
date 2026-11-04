"""R13 local experiment jobs. Each heavy job acquires and releases the shared lock.

Reuses A's fit/pack owners and R12's native inspection/read owners. Receipts
retain failed jobs; continuation never silently retries a scientific fit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

X = Path(__file__).resolve().parents[2]
O = Path('/var/tmp/byc-width-20261104')
R = Path('/var/tmp/byc-fchroma-20261029/r3/w100/root')
H = Path.home()
SCRATCH = H / 'tmp/zensim-width'
ORDER = (32, 256, 64, 512, 16)
HEAD = Path('/var/tmp/byc-linear-20261104/extra/heads/bin/prune_forward_bench')
EXPORT = Path('/mnt/v/output/zensim/a-weight-audit-2026-11-04/native/zensim-viz-export-d9e1ca3e9624036f567c29dbc5cd79668a697fe4a63b4bb33c9cf30fdaed2f6f')
BANKS = {
    'cid22': (R / 'wide/main/real/cid22_fit.parquet', '07e2f4b71fe0ccc67848f2896344e6d0b6a255b0e91a7da9062d471ea3e11100'),
    'hdr': (Path('/mnt/v/output/zensim/byc-hdr-20261104/banks-cube-root-203/teacher-train-r1/features.parquet'), '742fd8f314dcc5d1f812faa9924ac8b3fb048ba07dbc5f53a0b7f2b984857101'),
}


def registered_seeds():
    cut = O / 'OWNER_COST_CUT.json'
    if not cut.exists():
        return (0, 1, 2)
    decision = json.loads(cut.read_text())
    expected_widths = [4,16,32,64,256,512] if decision.get('schema') == 'r13-owner-cost-cut-v2' else [16,32,64,256,512]
    if (decision.get('schema') not in ('r13-owner-cost-cut-v1','r13-owner-cost-cut-v2') or decision.get('seeds') != [0]
            or decision.get('not_run_seeds') != [1, 2]
            or decision.get('widths') != expected_widths):
        raise ValueError('unrecognized R13 owner cut')
    return (0,)


def registered_widths():
    registered_seeds()  # validate the owner decision before applying its addition
    cut=O/'OWNER_COST_CUT.json'
    return ORDER+(4,) if cut.exists() and json.loads(cut.read_text()).get('additional_arm')=='W4-s0' else ORDER


def required_fits():
    return len(registered_widths()) * len(registered_seeds())


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def env():
    return dict(os.environ, ZENSIM_FORMULA_REV='7', ZENSIM_MAX_TIER='v3',
                RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                TMPDIR=str(SCRATCH), BYC_SCRATCH=str(SCRATCH),
                REV4_V2_BIN_DIR=str(O / 'bin'), ZEN_PANEL_BIN=str(O / 'bin/panel'),
                VIZ_CERTIFIER=str(O / 'bin/bake_dial_refit'),
                REV4_V2_ROOT=str(R), BYC_ROOT=str(R),
                BYC_CELLS='/var/tmp/byc-fchroma-20261029/r3/w100/cells',
                PYTHONPATH=f'{X}/scripts/rev4_featpot:{X}/scripts/byconstruction:{X}/scripts')


def job(label, argv):
    """One resource scope; no lock spans separate jobs or idle reporting."""
    receipt = O / 'logs' / f'{label}.receipt.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get('rc') == 0:
            return old
        raise ValueError(f'failed job retained; investigate before another attempt: {label}')
    argv = [str(v) for v in argv]
    start = now()
    write(O / 'SUPERVISION.json', dict(stage=label, coordinator_pid=os.getpid(),
          started=start, state='waiting-for-shared-lock', completed_fits=len(list((O/'fits').glob('*/FIT.json'))),
          required_fits=required_fits(), source=sha(Path(__file__)), next_action=argv))
    command = ['flock', str(H/'tmp/heavy.lock'), str(H/'work/claudehints/scripts/run-heavy'),
               '--mem', '24G', '--jobs', '8', '--', '/usr/bin/time', '-v',
               '-o', str(O/'logs'/f'{label}.time.txt'), *argv]
    with (O/'logs'/f'{label}.log').open('x') as log:
        log.write(json.dumps(command) + '\n'); log.flush()
        t = time.monotonic()
        p = subprocess.Popen(command, cwd=X, env=env(), stdout=log, stderr=subprocess.STDOUT)
        write(O/'ACTIVE_JOB.json', dict(stage=label, pid=p.pid, coordinator_pid=os.getpid(),
              started=start, log=str(log.name), command=command))
        rc = p.wait()
    result = dict(label=label, started=start, finished=now(), rc=rc,
                  wall_including_lock_seconds=time.monotonic()-t, argv=argv,
                  log=str(O/'logs'/f'{label}.log'), time_receipt=str(O/'logs'/f'{label}.time.txt'))
    write(receipt, result)
    with (O/'PROGRESS.jsonl').open('a') as f:
        f.write(json.dumps(result)+'\n')
    print(json.dumps(result), flush=True)
    if rc:
        write(O/'SUPERVISION.json', dict(stage=label,state='failed',receipt=str(receipt),rc=rc,last_check=now()))
        raise subprocess.CalledProcessError(rc, command)
    return result


def fit_args(width, seed, dest):
    return [sys.executable, '-B', X/'scripts/rev4_featpot/r13_width_fit.py', '--r13-width', width,
            '--root', R, '--spec', f'sel:45acffa603e8@h32:H{width}:cv16:cf98:fx2:mono:thr:mp{width//4}',
            '--columns', (H/'tmp/hc1-profile/cols_fx2.txt').read_text().strip(),
            '--head', 'N', '--seed-index', seed, '--strict-admission',
            '--data-role-decision', R/'human_role_decision.json', '--dest', dest, '--train-only',
            '--human-sources', 'tid2013', '--exclude-datasets',
            'pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022', '--mos-share', '36.035']


def quick_capacity(path):
    d = json.loads(Path(path).read_text())
    first, last = d['layers']
    width = first['out_dim']
    dead = [j for j in range(width) if first['biases'][j] <= 0
            and all(first['weights'][i*width+j] == 0 for i in range(first['in_dim']))]
    cutoff = max(abs(v) for v in last['weights']) * 1e-3
    residual = [j for j in range(width) if j not in dead and abs(last['weights'][j]) < cutoff]
    return dict(hidden=width, dead=len(dead), residual=len(residual),
                live=width-len(dead)-len(residual), dead_units=dead, residual_units=residual)


def pack(width, seed):
    from v2_production_pack import pack_production
    cell = O / f'fits/W{width}-s{seed}'
    dest = cell/'continuation/pack_e059'
    (dest/'refit').mkdir(parents=True, exist_ok=False)
    result = pack_production(cell/'continuation/ckpt/ckpt_epoch059.bin', dest,
                             R/'wide/main/real/cid22_fit.parquet', e33_output_stage=True, no_prune=True)
    bake = dest/'refit/production-f16.bin'
    cert = subprocess.check_output([str(O/'bin/bake_dial_refit'), 'certify-monotone', '--bake', str(bake)])
    cert = json.loads(cert)
    if not cert['certified']:
        raise ValueError('R13 monotonicity certificate failed')
    subprocess.run([str(HEAD), '--inspect', str(bake), str(cell/'HEAD.json')], check=True)
    result.update(certificate=cert, capacity=quick_capacity(cell/'HEAD.json'))
    write(cell/'PACK.json', result)


def fits():
    for seed in registered_seeds():
        for width in registered_widths():
            name = f'W{width}-s{seed}'
            cell = O/'fits'/name
            cell.mkdir(parents=True, exist_ok=True)
            if (cell/'FIT.json').exists():
                continue
            args = fit_args(width, seed, cell/'base')
            job(name+'-base', args+['--pack-production','--e33-output-stage'])
            args = fit_args(width, seed, cell/'continuation')
            job(name+'-continuation', args+['--init-from',cell/'base/refit/last.bin',
                '--fine-tune-epochs',60,'--fine-tune-lr','0.00001','--aic-public-train',
                '/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7/aic_full_fit.parquet',
                '--aic-weight',256,'--diagnostic-population','mixed-aic256-xref'])
            job(name+'-pack', [sys.executable,'-B',__file__,'pack','--width',width,'--seed',seed])
            row = json.loads((cell/'PACK.json').read_text())
            row.update(arm=f'W{width}', seed=seed, finished=now())
            # GNU time excludes waiting for the shared lock.
            times = []
            for stage in ('base','continuation','pack'):
                raw = (O/'logs'/f'{name}-{stage}.time.txt').read_text()
                elapsed = next(v.split(': ',1)[1].strip() for v in raw.splitlines() if 'Elapsed (wall clock)' in v)
                parts = [float(v) for v in elapsed.split(':')]
                seconds = sum(v*60**i for i,v in enumerate(reversed(parts)))
                times.append(dict(stage=stage, seconds=seconds, elapsed=elapsed))
            row['wall_seconds'] = sum(v['seconds'] for v in times)
            row['stage_times'] = times
            write(cell/'FIT.json',row)
            print('PACKED '+json.dumps(dict(name=name,wall_seconds=row['wall_seconds'],capacity=row['capacity'])),flush=True)
    write(O/'SUPERVISION.json',dict(stage='fits-complete',state='complete',completed_fits=required_fits(),required_fits=required_fits(),last_check=now()))


def freeze():
    models = {}
    rows = []
    for seed in registered_seeds():
        for width in registered_widths():
            cell = O/f'fits/W{width}-s{seed}'
            row = json.loads((cell/'FIT.json').read_text())
            bake = Path(row['packed_model'])
            if sha(bake) != row['packed_model_sha256']:
                raise ValueError('packed bake changed')
            row['packed_sha256'] = row['packed_model_sha256']
            row['packed_bytes'] = bake.stat().st_size
            rows.append(row)
            models[f'W{width}_s{seed}'] = str(bake)
    write(O/'MODELS.json', models)
    cut = json.loads((O/'OWNER_COST_CUT.json').read_text()) if (O/'OWNER_COST_CUT.json').exists() else None
    write(O/'FROZEN_FITS.json',dict(models=rows,source=sha(__file__),owner_cost_cut=cut))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('fits','pack','freeze','job'))
    ap.add_argument('--width',type=int,choices=ORDER+(4,))
    ap.add_argument('--seed',type=int,choices=range(3))
    ap.add_argument('--label')
    a, argv = ap.parse_known_args()
    if argv and a.command != 'job':
        ap.error('unexpected arguments')
    for p in (O/'logs',SCRATCH): p.mkdir(parents=True,exist_ok=True)
    if a.command == 'fits': fits()
    elif a.command == 'pack': pack(a.width,a.seed)
    elif a.command == 'freeze': freeze()
    else: job(a.label, argv[1:] if argv[:1] == ['--'] else argv)


if __name__ == '__main__': main()
