#!/usr/bin/env python3
"""Judge frozen seed0 allocation pixels with serving B; not a B-map baseline."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')
ROOT = OUT/'steercodec'


def main():
    pins = json.loads((OUT/'PROGRAM_MODEL_PINS.json').read_text())
    for p, h in pins.items():
        assert hashlib.sha256((OUT/p).read_bytes()).hexdigest() == h
    receipt = ROOT/'B_JUDGE_PREREAD.json'
    assert not receipt.exists(), 'preserve existing qualification before rerunning'
    receipt.write_text(json.dumps({'model': pins['models/B.bin'], 'formula_revision': 1,
        'inputs': 'same fixed 264 pairs.tsv rows; 192 composites selected by frozen seed0',
        'pairs_sha256': hashlib.sha256((ROOT/'score/pairs.tsv').read_bytes()).hexdigest(),
        'statistic_owner': 'unchanged STEERCODEC full-upgrade share',
        'meaning': 'B judges seed0 maps; not B-selected composites or a B spatial baseline',
        'pass_rule': 'pending'}, indent=2)+'\n')
    env = dict(os.environ, ZENSIM_FORMULA_REV='1', RAYON_NUM_THREADS='1',
        TMPDIR=str(Path.home()/'tmp/prodqual-b'))
    cmd = [str(OUT/'bin/serve_custom_bake'), '--pairs', str(ROOT/'score/pairs.tsv'), str(OUT/'models/B.bin')]
    (ROOT/'B_JUDGE_COMMAND.json').write_text(json.dumps(cmd, indent=2)+'\n')
    with (ROOT/'score/zensim-B.tsv').open('w') as output, (ROOT/'score/B.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=output, stderr=log, check=True)
    owner = REPO/'benchmarks/steercodec_2026-10-04/steer_codec.py'
    spec = importlib.util.spec_from_file_location('codec_owner', owner)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT = ROOT
    # The owner tags its own-score column by the map's bake name. Bind that
    # input column to B scores of those exact pixels; keep its arithmetic intact.
    scores = module.load_scores()
    module.load_scores = lambda: {'zensim:seed0': scores['zensim:B']}
    module.cmd_summary(argparse.Namespace(out=str(ROOT/'summary-B-judge.json')))


if __name__ == '__main__':
    main()
