#!/usr/bin/env python3
"""Bind the canonical intervention analyzer to explicitly frozen model bytes.

The historical default SHA guard remains in its owner. This scoped adapter
parameterizes only that guard and parser, then executes the unchanged analysis.
It refuses invalid pins before opening intervention metadata or measurements.
"""
import argparse
import hashlib
from pathlib import Path
import re
import sys

HISTORICAL = 'cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6'
OWNER = Path(__file__).with_name('rd_probe_analyze_2026-07-18.py')


def bound_source():
    source = OWNER.read_text()
    parser = '    root = ap.parse_args().interventions.resolve()'
    guard = f'    assert inp["model_sha256"] == "{HISTORICAL}"'
    if source.count(parser) != 1 or source.count(guard) != 1:
        raise RuntimeError('canonical analyzer changed; overlay must be reviewed')
    source = source.replace(parser, '''    ap.add_argument("--expected-model-sha256", required=True)
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_model_sha256):
        ap.error("expected model pin must be 64 lowercase hexadecimal digits")
    root = args.interventions.resolve()''')
    return source.replace(guard,
        '    assert inp["model_sha256"] == args.expected_model_sha256, "intervention model differs from frozen pin"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interventions', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--expected-model-sha256', required=True)
    ap.add_argument('--archive-owner', type=Path, required=True)
    args = ap.parse_args()
    if not re.fullmatch(r'[0-9a-f]{64}', args.expected_model_sha256):
        ap.error('expected model pin must be 64 lowercase hexadecimal digits')
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != args.expected_model_sha256:
        ap.error('model bytes differ from frozen pin')
    source = bound_source()
    args.archive_owner.write_text(source)
    sys.argv = [str(OWNER), '--interventions', str(args.interventions),
                '--expected-model-sha256', args.expected_model_sha256]
    scope = {'__name__': '__main__', '__file__': str(OWNER), 're': re}
    exec(compile(source, str(OWNER), 'exec'), scope)


if __name__ == '__main__':
    main()
