"""R12 frozen-model assessments through the mix's registered read owners.

Caller holds heavy.lock/run-heavy. Every stage has its own complete log and
receipt; failure is recorded and does not erase other assessments. No fit or
model choice occurs here. Existing mix outputs remain immutable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--models', type=Path, required=True)
    ap.add_argument('--only', default='eval,e36,chroma,steer,dial,nearid,hdr')
    ap.add_argument('--bin-dir', type=Path, help='explicit rebuilt serving tools; historical defaults when absent')
    ap.add_argument('--include-model', help='run pixel gates only for this frozen model; retain full model pins')
    a = ap.parse_args()
    x = Path(__file__).resolve().parents[2]
    bdir = a.bin_dir.resolve() if a.bin_dir else None
    def binary(name, historical):
        return str(bdir/name) if bdir else historical
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = json.loads(a.models.read_text())
    baseline = {f'F_s{s}': str(Path.home()/f'tmp/e36test/r10/runs/mixF3-s{s}/pack_e059/refit/production-f16.bin') for s in range(3)}
    all_models = dict(baseline, **models)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    pins = {n: dict(path=p, sha256=digest(p)) for n,p in all_models.items()}
    frozen = json.loads((a.models.parent/'FROZEN_FITS.json').read_text())
    expected = {f"{r['arm']}_s{r['seed']}":r['packed_sha256'] for r in frozen['models']}
    if set(models)!=set(expected) or any(pins[n]['sha256']!=expected[n] for n in models):
        raise ValueError('assessment models differ from frozen fits')
    pin_path=out/'MODEL_PINS.json'
    if pin_path.exists() and json.loads(pin_path.read_text())!=pins:
        raise ValueError('preserve original model pins')
    pin_path.write_text(json.dumps(pins, indent=2)+'\n')
    env = dict(os.environ, ZENSIM_FORMULA_REV='7', ZENSIM_MAX_TIER='v3', RAYON_NUM_THREADS='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TMPDIR=str(out/'scratch'),
               ZEN_PANEL_BIN=binary('panel','/var/tmp/byc-hdr-20261104/bin/panel'),
               PYTHONPATH=str(x/'scripts/rev4_featpot')+':'+str(x/'scripts/byconstruction')+':'+str(x/'scripts'))
    (out/'scratch').mkdir(exist_ok=True)
    py = sys.executable
    scripts = x/'scripts/byconstruction'
    pred = binary('predict_features_with_bake','/var/tmp/byc-fchroma-20261029/r3/tools/predict_features_with_bake')
    fitbin = binary('bake_dial_refit','/var/tmp/byc-hdr-20261104/bin/bake_dial_refit')
    byc = binary('byc_predict','/var/tmp/zensim-bin-10d087cd/byc_predict')
    steer = binary('steer_instrument','/var/tmp/byc-fx2-20261028/stage2/gate-tools-d02d2e07/steer_instrument')
    serve = binary('serve_custom_bake',str(Path.home()/'tmp/e36test/nearid/bin/serve_custom_bake'))
    tools=[pred,fitbin,env['ZEN_PANEL_BIN'],byc,steer,serve]
    if bdir: tools.append(str(bdir/'bake_verdict'))
    tool_pins={p:digest(p) for p in tools}
    tool_pin_path=out/'TOOL_PINS.json'
    if tool_pin_path.exists() and json.loads(tool_pin_path.read_text())!=tool_pins:
        raise ValueError('assessment instruments differ from original stage pins')
    tool_pin_path.write_text(json.dumps(tool_pins,indent=2)+'\n')
    records = []

    def run(name, argv, extra=None):
        argv = [str(v) for v in argv]
        receipt = out/f'{name}.receipt.json'
        if receipt.exists():
            raise ValueError(f'preserve prior stage, use --only for outstanding stages: {name}')
        now = datetime.now(timezone.utc).isoformat()
        (out/'STAGE').write_text(f'{now} {name} pid={os.getpid()}\n')
        with (out/f'{name}.log').open('x') as log:
            log.write(json.dumps(argv)+'\n'); log.flush()
            p = subprocess.run([str(v) for v in argv], cwd=x, env=dict(env, **(extra or {})), stdout=log, stderr=subprocess.STDOUT)
        row = dict(name=name, argv=[str(v) for v in argv], started=now, rc=p.returncode)
        receipt.write_text(json.dumps(row, indent=2)+'\n')
        records.append(row)
        (out/'PROGRESS.json').write_text(json.dumps(records,indent=2)+'\n')
        print(name, p.returncode, flush=True)
        return p.returncode == 0

    todo = a.only.split(',')
    if 'eval' in todo or 'eval-reports' in todo:
        ev = out/'eval'
        ev.mkdir(exist_ok='eval-reports' in todo)
        parent = Path.home()/'tmp/e36test/eval'
        for name in ('features.parquet','eval_rows.json','LABEL_READS.json','judge_pairs.tsv'):
            if not (ev/name).exists(): shutil.copy2(parent/name, ev/name)
        args = [py,scripts/'dataset_eval.py','score','--out',ev,'--predict',fitbin]
        for n,p in all_models.items(): args += ['--model',f'{n}={p}']
        if 'eval-reports' in todo or run('eval-score',args):
            spec = ','.join(f'{n.rsplit("_s",1)[0]}@{n.rsplit("_s",1)[1]}={n}' for n in all_models)
            # Import owners from this checkout before the historical scratch readout
            # inserts its old checkout paths. The readout itself is preserved verbatim.
            readout = Path.home()/'tmp/e36test/exp5/readout.py'
            code = 'import sys,runpy; import dataset_eval,aic_panel; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
            run('readout',[py,'-c',code,readout,ev,spec,out/'readout.json'])
            for rows in ('compression','noncodec'):
                run(f'ceval-{rows}',[py,scripts/'e36_exp56/ceval.py',ev,out/f'ceval-{rows}.json',spec,'--rows',rows])
    if 'e36' in todo:
        bank = Path('/mnt/v/zensim/e36-features/2026-11-01')
        idx = Path.home()/'tmp/e36feat/full'
        args = [py,scripts/'e36_features.py','score',bank/'bank',out/'scores','--predict',pred]
        for n,p in all_models.items(): args += ['--bake',f'{n}={p}']
        if run('e36-score',args):
            args = [py,scripts/'e36_features.py','inversions',out/'scores',idx,
                    '/mnt/v/output/zensim/e36-bulk-2026-10-17/pools-full-v1',out/'inversions']
            for n in all_models: args += ['--model',n]
            if run('e36-inversions',args):
                arms = ','.join(dict.fromkeys(n.rsplit('_s',1)[0] for n in all_models))
                seed_inventory={arm:sorted(int(n.rsplit('_s',1)[1]) for n in all_models if n.rsplit('_s',1)[0]==arm)
                                for arm in arms.split(',')}
                run('primary',[py,scripts/'e36_exp56/primary.py',out/'inversions/pairs_scored.parquet',
                               idx/'selection.jsonl',out/'PRIMARY.json',arms],
                    {'CONTROL':'F','ARM_SEEDS':json.dumps(seed_inventory)})
    if 'chroma' in todo:
        args = [py,scripts/'chroma_ladder_gate.py','score','--bank','/var/tmp/byc-fchroma-20261029/c/merged-bank',
                '--predict',pred,'--roles','dev','--out',out/'chroma','--drop-long-edge-at-most','64',
                '--bake',f'F_s0={baseline["F_s0"]}']
        for n,p in models.items(): args += ['--bake',f'{n}={p}']
        if run('chroma-score',args):
            args = [py,scripts/'chroma_ladder_gate.py','level','--out',out/'chroma','--baseline','F_s0']
            for n in models: args += ['--candidate',n]
            run('chroma-level',args)
            run('chroma-report',[py,scripts/'chroma_ladder_gate.py','report','--out',out/'chroma'])
    tools = str(bdir) if bdir else '/var/tmp/byc-fx2-20261028/stage2/gate-tools-d02d2e07'
    instruments = '/var/tmp/byc-fx2-20261028/stage2/fx2-gate-instruments-run02'
    for n,p in models.items():
        if a.include_model and n != a.include_model:
            continue
        if 'steer' in todo:
            run(f'steer-{n}',[py,scripts/'gates_steer_sharded.py','--name',n,'--model',p,'--out',out/f'gates/{n}',
                              '--bin',steer,'--formula-rev','7','--shards','9'])
        if 'dial' in todo:
            run(f'dial-{n}',[py,scripts/'gates.py','--name',n,'--model',p,'--out',out/f'gates/{n}',
                             '--bin-dir',tools,'--instruments',instruments,'--corpus-root',instruments+'/no-human-corpus',
                             '--formula-rev','7','--steps','verdict,identity'],
                {'PYTHONPATH':'/var/tmp/byc-chroma-20261028/a2/cluster-libs:'+env['PYTHONPATH']})
            run(f'gate-summary-{n}',[py,scripts/'gates.py','--summary','--out',out/f'gates/{n}','--name',n])
        if 'nearid' in todo:
            run(f'nearid-{n}', ['just','dataset-nearid-candidate',serve,
                              '/mnt/tower/output/zensim/mlic-nearid-20261027-run01/PACKET.json',out/f'nearid/{n}',
                              n.replace('_','').lower(),p,digest(p)])
    if 'hdr' in todo:
        banks = Path('/mnt/v/output/zensim/byc-hdr-20261104/banks-cube-root-203-aliases')
        args = [f'{n}={p}:fx2' for n,p in all_models.items()]
        run('hdr-panel',[py,scripts/'hdr_panel.py','--models',*args,'--byc-predict',byc,
                         '--banks',banks,'--out',out/'hdr-panel'])
        run('hdr-upiq',[py,scripts/'hdr_panel_upiq_all.py','--models',*args,'--byc-predict',byc,
                        '--upiq-bank',banks/'upiq-r2-bank','--out',out/'hdr-upiq'])
    (out/'STAGE').write_text(f'{datetime.now(timezone.utc).isoformat()} finished\n')
    if any(r['rc'] for r in records): raise SystemExit(1)


if __name__ == '__main__':
    main()
