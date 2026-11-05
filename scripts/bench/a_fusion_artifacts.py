"""Resolve derived artifacts independently of immutable benchmark captures."""
import json
from pathlib import Path


def parsed_path(cell, filename):
    ledger = cell / 'PARSE_RUNS.jsonl'
    if ledger.exists():
        for row in reversed([json.loads(line) for line in ledger.read_text().splitlines()]):
            if row['status'] == 0 and filename in row['artifacts']:
                path = Path(row['output']) / filename
                assert path.is_file(), path
                return path
    return cell / filename  # Historical captures keep their original derived files.
