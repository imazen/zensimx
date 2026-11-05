#!/usr/bin/env python3
"""Summarize admitted AMD captures without treating multiplexed groups as gates."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from a_family_cost_perf import summarize
from a_fusion_artifacts import parsed_path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def receipt(cell):
    assert (cell / 'COMPLETE.txt').is_file(), cell
    admission = json.loads((cell / 'admission.json').read_text())
    assert admission['status'] == 'accepted', (cell, admission)
    dispatch = json.loads((cell / 'DISPATCH.json').read_text())
    assert dispatch['selected_token'].endswith('X64V4xToken'), dispatch
    return dict(path=str(cell), command=admission['command'],
                admission_sha256=digest(cell / 'admission.json'),
                as_run_sha256_manifest=(cell / 'SHA256SUMS').read_text())


def counter_rows(cell, strict):
    result = dict(receipt=receipt(cell), counters=[], metrics={},
                  counters_sha256=digest(cell / 'counters.csv'))
    with (cell / 'counters.csv').open() as stream:
        for row in csv.reader(line for line in stream if not line.startswith('#')):
            if len(row) < 7:
                continue
            event = ','.join(row[2:-4])
            if event:
                supported = bool(row[0]) and not row[0].startswith('<')
                coverage = float(row[-3]) if row[-3] else None
                if strict:
                    assert supported and coverage is not None and coverage >= 99, (cell, row)
                result['counters'].append(dict(event=event, supported=supported,
                    count=float(row[0]) if supported else None, running_percent=coverage))
            if row[-2]:
                unit, name = row[-1].strip().rsplit(' ', 1) if ' ' in row[-1].strip() else ('', row[-1])
                result['metrics'][name] = dict(value=float(row[-2]), unit=unit.strip())
    assert result['counters'], cell
    return result


def ibs(cell):
    result = dict(receipt=receipt(cell), extraction_load_samples=0, symbols={},
                  load_data_sha256=digest(parsed_path(cell, 'load-data.txt')))
    counts, positive_counts, latency_weights = Counter(), Counter(), Counter()
    sources = defaultdict(Counter)
    for block in parsed_path(cell, 'load-data.txt').read_text().strip().split('\n\n'):
        lines = block.splitlines()
        header = lines[0]
        if '|OP LOAD|' not in header:
            continue
        frames = []
        for line in lines[1:]:
            match = re.fullmatch(r'\s*[0-9a-f]+\s+(.+)', line)
            assert match, line
            frames.append(match[1])
        if not any('zensim::fold_engine::compute_fold_backed_stoppable' in f for f in frames):
            continue
        symbol = frames[0]
        weight = int(header.rsplit(None, 1)[-1])
        source = re.search(r'\|LVL ([^|]+)\|', header)[1]
        counts[symbol] += 1
        sources[symbol][source] += 1
        if weight:
            positive_counts[symbol] += 1
            latency_weights[symbol] += weight
    assert counts, ('no extraction load samples', cell)
    result['extraction_load_samples'] = sum(counts.values())
    result['latency_units'] = 'IBS reported cycle weights; zero-weight samples are counted separately, not assigned a latency'
    for symbol, count in counts.most_common():
        positive = positive_counts[symbol]
        result['symbols'][symbol] = dict(samples=count, sources=dict(sources[symbol]),
            nonzero_latency_samples=positive, zero_latency_samples=count-positive,
            nonzero_latency_weight_sum=latency_weights[symbol],
            mean_nonzero_latency_cycles=latency_weights[symbol]/positive if positive else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.study
    metadata = json.loads((root / 'PERF_METRICS.json').read_text())
    groups = {'PipelineL1', 'PipelineL2', 'l3_cache'}
    required = [m for m in metadata if m.get('MetricGroup') in groups]
    selection = json.loads((root / 'AMD_SELECTION.json').read_text())
    result = dict(host='dev', tier='v4x', cpu=0, images=20, repeats=10, sizes={},
        selection=selection, selection_sha256=digest(root / 'AMD_SELECTION.json'),
        uncore_scope='CCD0/package shared PMUs during the FIFO window, not process-private events',
        units='cycles attribution uses extraction self samples; IBS samples are not cycle shares')
    for size in ['256', '1024', 'large']:
        original = root / ('phase1' if size == '256' else 'phase1-v6')
        cycle = original / f'{size}-cycles'
        metrics = {}
        for row in required:
            name = row['MetricName']
            cell = root / selection['metric_labels'][size] / f'{size}-metric-{name}'
            capture = counter_rows(cell, strict=True)
            if name == 'l3_read_miss_latency_for_remote_dram' and name not in capture['metrics']:
                requests = [v for v in capture['counters']
                            if v['event'] == 'l3_xi_sampled_latency_requests.dram_far']
                assert len(requests) == 1 and requests[0]['count'] == 0, (cell, capture)
                capture['metrics'][name] = dict(value=None, unit='ns',
                    finding='not measured: zero sampled remote-DRAM requests')
            assert name in capture['metrics'], (cell, capture['metrics'])
            scoped = 'duration_time' not in row['MetricExpr']
            metrics[name] = dict(metadata=row, capture=capture,
                qualifies_for_scoring_window=scoped,
                scope_note='FIFO-controlled hardware events' if scoped else
                    'diagnostic only: tool duration includes disabled setup/teardown')
        ibs_cell = root / 'phase1-v6' / f'{size}-ibs'
        result['sizes'][size] = dict(cycles=dict(receipt=receipt(cycle), **summarize(parsed_path(cycle, 'samples.txt'))),
            counters=counter_rows(original / f'{size}-counters', strict=False),
            fills=counter_rows(original / f'{size}-fills', strict=True),
            metrics=metrics, ibs=ibs(ibs_cell))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(host='dev', tier='v4x', sizes=list(result['sizes']),
                         individual_metric_cells=3*len(required), output=str(args.output))))


if __name__ == '__main__':
    main()
