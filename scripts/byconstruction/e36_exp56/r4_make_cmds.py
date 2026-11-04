"""R4 (zensimx record; registration waits on the owner's choice of DVIFM preset, decision 5): p30L + map 300 with
DVIFM's painted map as the map-term target (arm dv; the consensus arm cs was dropped by the owner's option b1). Every command is its twin's, experiment 3's p30L-map300-s{seed} (exp3m), with only the map pool,
the destination and the scratch dir changed; each seed-0 command also gets the seven AIC2026 fold twins of
expaic map300-k{k} (fold table and destination changed), kept in expaic/ where its scorer reads them.
    make_cmds.py PRESET POOLS_DIR    # POOLS_DIR/PRESET/manifest.json and POOLS_DIR/PRESET-consensus/manifest.json
Prints the continuation tags, then the fold tags, one line each."""
import json
import os
import sys
from pathlib import Path
H = Path.home() / 'tmp/e36test'
OUT = H / 'r4'
F = Path('/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7')
OLD_MAP = '/mnt/v/zensim/e36-features/2026-11-01/map-pool-16/manifest.json'
preset, pools = sys.argv[1], Path(sys.argv[2]).resolve()
# Owner, 2026-11-03, option (b1): the DVIFM-only arm. The consensus arm is not run (its loop's only independent judge
# would have been CVVDP).
ARMS = {'dv': pools / preset / 'manifest.json'}
for m in ARMS.values():
    assert json.loads(m.read_text())['provenance']['new_target']['preset'] == preset, m


def save(exp, tag, argv, env):
    (H / exp / 'cmds').mkdir(parents=True, exist_ok=True)
    (H / exp / 'scratch' / tag).mkdir(parents=True, exist_ok=True)
    env = dict(env, TMPDIR=str(H / exp / 'scratch' / tag))
    (H / exp / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return f'{exp}:{tag}'


cont, folds = [], []
for seed in (0, 1, 2):
    twin = json.loads((H / 'exp3m/cmds' / f'p30L-map300-s{seed}.cmd.json').read_text())
    wrapper = (Path(twin['env']['REV4_V2_BIN_DIR']) / 'zensim_mlp_train').read_text()
    assert wrapper.count(f'--map-pool {OLD_MAP} ') == 1, wrapper
    for arm, pool in ARMS.items():
        tag = f'p30L-map300{arm}-s{seed}'
        bindir = OUT / f'bin-{tag}'
        bindir.mkdir(parents=True, exist_ok=True)
        for f in Path(twin['env']['REV4_V2_BIN_DIR']).iterdir():
            if f.name != 'zensim_mlp_train' and not (bindir / f.name).exists():
                os.symlink(f.resolve(), bindir / f.name)
        w = bindir / 'zensim_mlp_train'
        assert wrapper.count(f'# E36 experiment 3 arm p30L-map300-s{seed}\n') == 1, wrapper
        w.write_text(wrapper.replace(f'# E36 experiment 3 arm p30L-map300-s{seed}\n', f'# E36 R4 arm {tag}\n')
                     .replace(f'--map-pool {OLD_MAP} ', f'--map-pool {pool} '))
        w.chmod(0o755)
        argv = list(twin['argv'])
        argv[argv.index('--dest') + 1] = str(OUT / 'runs' / tag)
        env = dict(twin['env'], REV4_V2_BIN_DIR=str(bindir), ZEN_PANEL_BIN=str(bindir / 'panel'))
        cont.append(save('r4', tag, argv, env))
        if seed == 0:
            assert argv[argv.index('--aic-public-train') + 1] == str(F / 'aic_full_fit.parquet')
            for k in range(7):
                a2 = list(argv)
                a2[a2.index('--aic-public-train') + 1] = str(F / f'aic_k{k}_fit.parquet')
                a2[a2.index('--dest') + 1] = str(H / 'expaic' / 'runs' / f'map300{arm}-k{k}')
                folds.append(save('expaic', f'map300{arm}-k{k}', a2, env))
print(' '.join(cont))
print(' '.join(folds))
