"""Revised plan R3 (zensimx record): the map term on views at quality setting <= 60 only (map-pool-16-q0-60, 1,310 of
1,967 views). Two arms, each as its ungated twin with only the pool path changed:
  p30L-map300q60              = experiment 3's p30L-map300 (exp3m/make_cmds.py)
  tid2013_mdid-p30L-map300q60 = experiment 5 stage 3's mix + map 300 (exp5c/make_cmds.py)
Full data at seeds 0-2 (exp8/cmds) and the AIC2026 public_train folds k0-k6 of each seed-0 command (expaic/cmds)."""
import json, os
from pathlib import Path
H = Path.home() / 'tmp/e36test'
O = Path('/mnt/v/zensim/e36-features/2026-11-01')
OUT = H / 'exp8'
GATED = O / 'map-pool-16-q0-60/manifest.json'
UNGATED = O / 'map-pool-16/manifest.json'
F = Path('/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7')
TWINS = {'p30L-map300q60': ('exp3m', 'p30L-map300'), 'tid2013_mdid-p30L-map300q60': ('exp5c', 'tid2013_mdid-p30L-map300')}
tags, aic = [], []
for name, (exp, twin) in TWINS.items():
    for seed in (0, 1, 2):
        src = json.loads((H / exp / 'cmds' / f'{twin}-s{seed}.cmd.json').read_text())
        tag = f'{name}-s{seed}'
        bindir = OUT / f'bin-{tag}'
        bindir.mkdir(parents=True, exist_ok=True)
        old = Path(src['env']['REV4_V2_BIN_DIR'])
        for f in old.iterdir():
            if f.name != 'zensim_mlp_train' and not (bindir / f.name).exists():
                os.symlink(f.resolve(), bindir / f.name)
        wrapper = (old / 'zensim_mlp_train').read_text()
        assert wrapper.count(str(UNGATED)) == 1, tag
        (bindir / 'zensim_mlp_train').write_text(wrapper.replace(str(UNGATED), str(GATED)).replace(
            f'arm {twin}-s{seed}', f'arm {tag} (R3: map term on views at quality <= 60)'))
        (bindir / 'zensim_mlp_train').chmod(0o755)
        argv = list(src['argv'])
        argv[argv.index('--dest') + 1] = str(OUT / 'runs' / tag)
        env = dict(src['env'], REV4_V2_BIN_DIR=str(bindir), ZEN_PANEL_BIN=str(bindir / 'panel'),
                   TMPDIR=str(OUT / f'scratch/{tag}'))
        (OUT / f'scratch/{tag}').mkdir(parents=True, exist_ok=True)
        (OUT / 'cmds').mkdir(exist_ok=True)
        (OUT / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
        tags.append(f'exp8:{tag}')
        if seed == 0:
            assert argv[argv.index('--aic-public-train') + 1] == str(F / 'aic_full_fit.parquet')
            short = 'map300q60' if name.startswith('p30L') else 'mixmap300q60'
            for k in range(7):
                at = f'{short}-k{k}'
                a2 = list(argv)
                a2[a2.index('--aic-public-train') + 1] = str(F / f'aic_k{k}_fit.parquet')
                a2[a2.index('--dest') + 1] = str(H / 'expaic' / 'runs' / at)
                e2 = dict(env, TMPDIR=str(H / 'expaic' / f'scratch/{at}'))
                (H / 'expaic' / f'scratch/{at}').mkdir(parents=True, exist_ok=True)
                (H / 'expaic' / 'cmds' / f'{at}.cmd.json').write_text(json.dumps(dict(argv=a2, env=e2), indent=1) + '\n')
                aic.append(f'expaic:{at}')
print(' '.join(tags + aic))
