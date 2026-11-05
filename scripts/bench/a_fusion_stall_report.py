#!/usr/bin/env python3
"""Summarize admitted Intel stall captures; sampled periods count events, not cycles."""
import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import socket

from a_family_cost_perf import bucket


def sampled_events(path):
    total = extraction = samples = extraction_samples = 0
    weights, counts = Counter(), Counter()
    symbols, symbol_samples = defaultdict(Counter), defaultdict(Counter)
    outside = Counter()
    for block in path.read_text().strip().split("\n\n"):
        lines = block.strip().splitlines()
        period = int(lines[0].strip())
        frames = []
        for line in lines[1:]:
            match = re.fullmatch(r"\s*[0-9a-f]+\s+(.+)", line)
            assert match, line
            frames.append(match[1])
        assert frames
        samples += 1
        total += period
        symbol = frames[0]
        if not any("zensim::fold_engine::compute_fold_backed_stoppable" in f for f in frames):
            outside[symbol] += period
            continue
        extraction_samples += 1
        extraction += period
        key = bucket(symbol)
        weights[key] += period
        counts[key] += 1
        symbols[key][symbol] += period
        symbol_samples[key][symbol] += 1
    assert sum(weights.values()) == extraction
    return dict(samples=samples, extraction_samples=extraction_samples,
                all_event_count_estimate=total, extraction_event_count_estimate=extraction,
                buckets={key: dict(event_count_estimate=value, samples=counts[key],
                    symbols=dict(symbols[key].most_common()), symbol_samples=dict(symbol_samples[key]))
                    for key, value in weights.most_common()},
                outside_extraction_symbols=dict(outside.most_common()))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def admitted(cell):
    receipt = json.loads((cell / 'admission.json').read_text())
    assert receipt['status'] == 'accepted', (cell, receipt['status'])
    return dict(admission_sha256=digest(cell / 'admission.json'), command=receipt['command'])


def counters(cell):
    result = dict(metrics={}, raw=[], receipt=admitted(cell))
    previous_core = False
    with (cell / 'counters.csv').open() as stream:
        for row in csv.reader(line for line in stream if not line.startswith('#')):
            if len(row) < 7:
                continue
            event = ",".join(row[2:-4])
            if event:
                previous_core = event.startswith('cpu_core/')
                if previous_core:
                    assert not row[0].startswith('<'), (cell, row)
                    assert float(row[-3]) >= 99, ('multiplexed counter cannot qualify', cell, row)
                    result['raw'].append(dict(event=event, count=float(row[0]), running_percent=float(row[-3])))
            if previous_core and row[-2]:
                match = re.search(r'\btma_\w+', row[-1])
                if match:
                    result['metrics'][match[0]] = dict(value=float(row[-2]), unit=row[-1].strip())
    assert result['raw'] and result['metrics'], cell
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert socket.gethostname() == 'i270'
    root = args.capture
    assert (root / 'COMPLETE.txt').is_file()
    result = dict(host='i270', tier='v3', cpu=2, images=20, repeats=10,
                  event_units='period-weighted sampled event-count estimates; not cycles', sizes={})
    for size in ['256', '1024', 'large']:
        groups = [counters(root / f'{size}-stat-{i}') for i in range(4)]
        metrics = {name: value for group in groups for name, value in group['metrics'].items()}
        required = ['tma_l1_bound', 'tma_l2_bound', 'tma_l3_bound', 'tma_dram_bound',
                    'tma_store_bound', 'tma_mem_latency', 'tma_mem_bandwidth',
                    'tma_ports_utilization', 'tma_divider']
        assert all(name in metrics for name in required), (size, metrics)
        raw = {row['event']: row['count'] for row in groups[0]['raw']}
        slots = raw['cpu_core/slots/']
        frontend = 100 * raw['cpu_core/topdown-fe-bound/'] / slots
        bad_speculation = 100 * raw['cpu_core/topdown-bad-spec/'] / slots
        events = {}
        for name in ['mem_load_retired.l1_miss', 'mem_load_retired.l2_miss',
                     'mem_load_retired.l3_miss', 'exe_activity.bound_on_loads']:
            cell = root / f'{size}-{name}'
            original = cell
            replacements = {('256', 'mem_load_retired.l3_miss'): '256-l3-period1',
                            ('256', 'mem_load_retired.l2_miss'): '256-l2-period101',
                            ('1024', 'mem_load_retired.l3_miss'): '1024-l3-period101'}
            replacement = root / replacements.get((size, name), '__no_replacement__')
            if replacement.is_dir():
                cell = replacement
            receipt = admitted(cell)
            confirmation = {}
            if cell != original:
                assert json.loads((cell / 'count-admission.json').read_text())['status'] == 'accepted'
                confirmation = dict(direct_count_csv=(cell / 'count.csv').read_text(),
                    direct_count_admission_sha256=digest(cell / 'count-admission.json'))
            if not (cell / 'samples.txt').read_text().strip():
                events[name] = dict(samples=0, attribution='not measured: no sampled events',
                                    receipt=receipt, **confirmation)
                continue
            summary = sampled_events(cell / 'samples.txt')
            summary['receipt'] = receipt
            summary['samples_sha256'] = digest(cell / 'samples.txt')
            summary['original_capture'] = str(original)
            summary['selected_capture'] = str(cell)
            summary.update(confirmation)
            events[name] = summary
        result['sizes'][size] = dict(metrics=metrics, metric_groups=groups,
            frontend_slot_percent=frontend, bad_speculation_slot_percent=bad_speculation, events=events)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(host='i270', sizes=list(result['sizes']), metric_cells=12,
                          event_cells=12, output=str(args.output))))


if __name__ == '__main__':
    main()
