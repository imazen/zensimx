"""Experiment 5 commands: F3's base-fit step through the experiment copy of the scripts, bin-fast trainer."""
import json, os, sys
from pathlib import Path
H = Path.home() / 'tmp/e36test'; OUT = H / 'exp5'
SRC = H / 'src-exp5/scripts'
step = json.load(open(Path.home() / 'work/zen/zensim-training/recipes/F3.json'))['steps'][0]
PY = str(Path.home() / '.cache/zensim-training-env/py3147-20261031/bin/python')
DS = ['pipal', 'mdid', 'live-md', 'cid-iq', 'cid-oa', 'irccyn-ivc', 'colourlab-rating-variability-2022']
HS = ['kadid', 'tid2013', 'konfig', 'cid22_a25']
SHARE = '36.035'
# Reruns after 2026-11-02 14:40 use the walk build (zensimx, uniform mode, experiment5_scale admission); experiment 6's
# control-wb checks that it reproduces the bin-fast control bit-exactly.
BIN = os.environ.get('EXP5_BIN', 'bin-walk')


def arm_flags(name):
    if name == 'control': return []
    if name == 'allsub': return ['--human-sources', ','.join(HS)]
    if name == 'none': return ['--human-sources', 'none', '--exclude-datasets', ','.join(DS)]
    if name == 'all-equal': return ['--mos-share', SHARE]
    srcs = name.split('+')
    hs = [s for s in srcs if s in HS]; ds = [s for s in srcs if s in DS]
    assert len(hs) + len(ds) == len(srcs), name
    return (['--human-sources', ','.join(hs) if hs else 'none'] + (['--exclude-datasets', ','.join(d for d in DS if d not in ds)] if len(ds) < len(DS) else []) + ['--mos-share', SHARE])


def command(name, seed):
    tag = name.replace('+', '_').replace('colourlab-rating-variability-2022', 'colourlab')
    dest = OUT / 'runs' / f'{tag}-s{seed}'
    argv = [PY if a == '{python}' else a for a in step['argv']]
    argv[2] = str(SRC / 'rev4_featpot/v2_confirm_fit.py')
    argv[argv.index('--seed-index') + 1] = str(seed)
    argv[argv.index('--dest') + 1] = str(dest)
    argv += arm_flags(name)
    env = dict(step['env'], REV4_V2_BIN_DIR=str(H / BIN), ZEN_PANEL_BIN=str(H / 'bin-fast/panel'),
               BYC_SCRATCH=str(OUT / 'scratch'), TMPDIR=str(OUT / f'scratch/{tag}-s{seed}'),
               PYTHONPATH=f"{SRC / 'rev4_featpot'}:{SRC}")
    (OUT / f'scratch/{tag}-s{seed}').mkdir(parents=True, exist_ok=True)
    (OUT / 'cmds').mkdir(exist_ok=True)
    (OUT / 'cmds' / f'{tag}-s{seed}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return f'{tag}-s{seed}'


seed = int(sys.argv[1]); names = sys.argv[2].split(',')
print(' '.join(command(n, seed) for n in names))
