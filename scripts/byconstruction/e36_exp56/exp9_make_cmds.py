"""R7 (zensimx record, registered 2026-11-03): CLIC 2024 as a training leg. Every command is its twin's with only the
root (the R7 copy with the CLIC 2024 leg appended, `clic.py leg`) and the base checkpoint changed.
  base fits:     f3clic-s{0,1,2}   = F3's base-fit step (experiment 5's copy, bin-walk uniform: reproduces F3 bit-exactly)
  continuations: p30L-clic-s{seed}    = arms/p30L-s{seed}                      from f3clic-s{seed}
                 control-clic-s{seed} = F3's control continuation (control-s{seed}) from f3clic-s{seed}  (amendment 1)
  AIC2026 folds: expaic p30Lclic-k{0..6}, controlclic-k{0..6} from the seed-0 continuations, and the twin folds
                 control-k{0..6} (F3's control recipe, original root and base; none existed before)
Prints the base tags, then the continuation and fold tags, one line each."""
import json
from pathlib import Path
H = Path.home() / 'tmp/e36test'
OUT = H / 'exp9'
OLD = '/var/tmp/byc-fchroma-20261029/r3/w100/root'
NEW = '/var/tmp/byc-r7-clic24-20261103/root'
F = Path('/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7')
SRC = H / 'src-exp5/scripts'
PY = str(Path.home() / '.cache/zensim-training-env/py3147-20261031/bin/python')
step = json.load(open(Path.home() / 'work/zen/zensim-training/recipes/F3.json'))['steps'][0]
MIX = ['--human-sources', 'tid2013', '--exclude-datasets',
       'pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022', '--mos-share', '36.035']


def reroot(argv, env):
    argv = [a.replace(OLD, NEW) for a in argv]
    env = {k: (v.replace(OLD, NEW) if isinstance(v, str) else v) for k, v in env.items()}
    assert not any(OLD in a for a in argv) and not any(OLD in str(v) for v in env.values())
    return argv, env


def save(exp, tag, argv, env):
    (H / exp / 'cmds').mkdir(parents=True, exist_ok=True)
    (H / exp / f'scratch/{tag}').mkdir(parents=True, exist_ok=True)
    env = dict(env, TMPDIR=str(H / exp / f'scratch/{tag}'))
    (H / exp / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return f'{exp}:{tag}'


base_tags, cont_tags = [], []
for seed in (0, 1, 2):
    for name, flags in (('f3clic', []),):
        tag = f'{name}-s{seed}'
        argv = [PY if a == '{python}' else a for a in step['argv']]
        argv[2] = str(SRC / 'rev4_featpot/v2_confirm_fit.py')
        argv[argv.index('--seed-index') + 1] = str(seed)
        argv[argv.index('--dest') + 1] = str(OUT / 'runs' / tag)
        argv += flags
        env = dict(step['env'], REV4_V2_BIN_DIR=str(H / 'bin-walk'), ZEN_PANEL_BIN=str(H / 'bin-fast/panel'),
                   BYC_SCRATCH=str(OUT / 'scratch'), PYTHONPATH=f"{SRC / 'rev4_featpot'}:{SRC}")
        base_tags.append(save('exp9', tag, *reroot(argv, env)))
    control = H / ('control-s0.cmd.json' if seed == 0 else f'arms/control-s{seed}.cmd.json')
    for name, twin, base in (('p30L-clic', H / 'arms' / f'p30L-s{seed}.cmd.json', f'f3clic-s{seed}'),
                             ('control-clic', control, f'f3clic-s{seed}')):
        src = json.loads(twin.read_text())
        argv, env = reroot(src['argv'], src['env'])
        argv[argv.index('--init-from') + 1] = str(OUT / 'runs' / base / 'refit/last.bin')
        tag = f'{name}-s{seed}'
        argv[argv.index('--dest') + 1] = str(OUT / 'runs' / tag)
        cont_tags.append(save('exp9', tag, argv, env))
        if seed == 0:
            assert argv[argv.index('--aic-public-train') + 1] == str(F / 'aic_full_fit.parquet')
            short = 'p30Lclic' if name == 'p30L-clic' else 'controlclic'
            for k in range(7):
                a2 = list(argv)
                a2[a2.index('--aic-public-train') + 1] = str(F / f'aic_k{k}_fit.parquet')
                a2[a2.index('--dest') + 1] = str(H / 'expaic' / 'runs' / f'{short}-k{k}')
                cont_tags.append(save('expaic', f'{short}-k{k}', a2, env))
# twin folds for F3's control recipe: the original command, only the fold table and destination changed
src = json.loads((H / 'control-s0.cmd.json').read_text())
assert src['argv'][src['argv'].index('--aic-public-train') + 1] == str(F / 'aic_full_fit.parquet')
for k in range(7):
    a2 = list(src['argv'])
    a2[a2.index('--aic-public-train') + 1] = str(F / f'aic_k{k}_fit.parquet')
    a2[a2.index('--dest') + 1] = str(H / 'expaic' / 'runs' / f'control-k{k}')
    cont_tags.append(save('expaic', f'control-k{k}', a2, src['env']))
print(' '.join(base_tags))
print(' '.join(cont_tags))
