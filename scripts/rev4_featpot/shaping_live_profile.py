"""Attach profiling to an existing trainer; never start or replay training."""
import argparse, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
from shaping_completion import encoded, sha


def identity(pid):
    root = Path('/proc')/str(pid)
    stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
    if stat[0] == 'Z':
        raise ValueError('trainer is no longer running')
    executable = (root/'exe').resolve()
    if executable.name != 'zensim_mlp_train':
        raise ValueError('attach only to the existing native trainer')
    return dict(pid=pid, start_ticks=stat[19], executable=str(executable),
                executable_sha256=sha(executable),
                command=(root/'cmdline').read_bytes().split(b'\0')[:-1])


def profile(pid, output):
    if output.exists():
        raise ValueError('preserve existing live profile')
    before = identity(pid)
    output.mkdir(parents=True)
    commands = [
        ['perf', 'record', '-e', 'cycles:u', '-F', '99', '--call-graph', 'dwarf,8192',
         '-p', str(pid), '-o', str(output/'perf.data'), '--', 'sleep', '10'],
        ['perf', 'stat', '-x', ';', '-e',
         'task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations,page-faults',
         '-p', str(pid), '--', 'sleep', '10'],
        ['perf', 'report', '--stdio', '--no-children', '--sort', 'symbol',
         '--percent-limit', '1', '--call-graph', 'none', '-i', str(output/'perf.data')],
    ]
    for name, command in zip(('record.log', 'counters.txt', 'hotspots.txt'), commands):
        if identity(pid) != before:
            raise ValueError('trainer identity changed during observation')
        with (output/name).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError('profiler failed; evidence preserved in '+str(output/name))
    if identity(pid) != before:
        raise ValueError('trainer identity changed during observation')
    before['command'] = [arg.decode() for arg in before['command']]
    receipt = dict(checked_utc=datetime.now(timezone.utc).isoformat(), trainer=before,
                   commands=commands, driver_sha256=sha(__file__),
                   files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
                   training_started=False, training_replayed=False,
                   interpretation='live sampled hotspots and counters only; profiler adds overhead; not an isolated throughput benchmark')
    (output/'PROFILE.json').write_bytes(encoded(receipt))
    return dict(output=str(output), pid=pid, profile_sha256=sha(output/'PROFILE.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(profile(a.pid, a.output)))
