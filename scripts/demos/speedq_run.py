#!/usr/bin/env python3
"""SPEEDQ parity, quiet-gated paired rounds and fresh-process peak RSS."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import os
from pathlib import Path
import subprocess
import time
import tempfile
import shutil
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from collections import deque

GEOMETRIES = ['64x64','128x128','256x256','512x512','1024x1024','2048x2048','4096x4096','1920x1080']
TIERS = ['v4x','v4','v3','scalar']
THREADS = [1,2,4,8,16,32]
ARMS = ['by_v2fy_r3','by_v2fy_r4','by_v2fy_r5','zensim_B','fast_ssim2','butteraugli','ssimulacra2_rs']
BAKE = '/var/tmp/rev4-featpot/d1-results/confirm/cells/sel:59f0bbc2f290@h32:H128:cv16:cf98__N/full_s0/refit/production-f16.bin'
SOURCE_SHA = 'f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400'
CLEAN_ROUND_RULE = 'first-32-clean-of-<=64 (owner 2026-10-08)'
LEGACY_ROUND_RULE = 'all-32-clean'
TRAIN_NAMES = {'zensim_mlp_train','train_hybrid.py','v2_lodo_mlp.py','zen-train','zen_train','e28_recipe.py','e28_simplex.py','bake_dial_refit'}
CPUSETS = {1:'2',2:'2,3',4:'0-3',8:'0-7',16:'0-15',32:'0-31'}


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def freeze_binary(build_log, dest):
    """Select Cargo's artifact receipt, never an old filename in target/."""
    messages=[json.loads(line) for line in Path(build_log).read_text().splitlines() if line.startswith('{')]
    assert any(m.get('reason')=='build-finished' and m.get('success') is True for m in messages), 'successful Cargo build receipt required'
    paths={m['executable'] for m in messages if m.get('reason')=='compiler-artifact' and m.get('executable') and m.get('target',{}).get('name')=='ssim2_speed_bar' and m['target'].get('kind')==['bench']}
    assert len(paths)==1, 'one unambiguous speed-matrix artifact required'
    source=Path(paths.pop());dest=Path(dest)
    with dest.open('xb') as out,source.open('rb') as inp: shutil.copyfileobj(inp,out)
    dest.chmod(source.stat().st_mode & 0o777)
    dependencies={}
    for m in messages:
        if m.get('reason')!='compiler-artifact' or m.get('target',{}).get('name') not in ['fast_ssim2','butteraugli','ssimulacra2','archmage','rayon'] or m['target'].get('kind')!=['lib']:
            continue
        name=m['target']['name']
        if name=='fast_ssim2' and m['package_id'].startswith('git+'):
            name='fast_ssim2_main'
        assert name not in dependencies or dependencies[name]['package_id']==m['package_id'], 'distinct packages cannot share an artifact inventory name'
        dependencies[name]=dict(package_id=m['package_id'],features=m['features'])
    write(dest.with_name(dest.name+'.artifact.json'),dict(cargo_artifact=str(source),build_log=str(build_log),binary_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),dependencies=dependencies))
    print(dest,flush=True)


def training_processes(proc=Path('/proc')):
    found = []
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():
            continue
        try:
            parts = (p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError):
            continue
        names = [Path(arg).name for arg in parts[:4] if arg]
        matches = TRAIN_NAMES.intersection(names)
        if matches:
            found.append({'pid':int(p.name),'program':sorted(matches)})
    return found


def quiet_state():
    load = float(Path('/proc/loadavg').read_text().split()[0])
    foreign = {}
    for name in ['cargo','rustc']:
        run = subprocess.run(['pgrep','-x',name],capture_output=True,text=True)
        if run.returncode not in (0,1):
            raise RuntimeError(f'pgrep {name} failed')
        foreign[name] = [int(p) for p in run.stdout.split()]
    training = training_processes()
    return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'load1':load,'foreign':foreign,'training':training,
            'admitted':load<2.0 and not any(foreign.values()) and not training}


def quiet_gate(log):
    start = time.monotonic()
    with Path(log).open('a') as f:
        while True:
            state = quiet_state()
            state['wait_seconds'] = time.monotonic()-start
            f.write(json.dumps(state)+'\n'); f.flush()
            if state['admitted']:
                return state
            print(f"quiet wait: load1={state['load1']} cargo/rustc={state['foreign']} training={state['training']}",flush=True)
            refresh_activity("quiet wait")
            time.sleep(10)


def environment(geometry, tier, threads):
    return {**os.environ, 'TMPDIR':str(Path.home()/'tmp'),
            'ZEN_S2_SPEEDQ':'1','ZEN_S2_GEOMETRY':geometry,'ZEN_S2_TIER':tier,
            'ZEN_S2_SPEEDQ_BAKE':BAKE,'RAYON_NUM_THREADS':str(threads),
            'ZENBENCH_NO_SAVE':'1','ZENBENCH_NO_CALIBRATE':'1'}


def worker_revision(name):
    if name.startswith('by_v2fy_r'):
        revision = name.removeprefix('by_v2fy_r').split('_')[0]
        assert revision in ('3', '4', '5'), 'unknown formula revision'
        return int(revision)
    if name.startswith(('e33_', 'e34_')) or name == 'zensim_B_rev5':  # Registered candidate bakes and default B use Rev5.
        return 5
    if name.startswith('r7_'):  # The two curated revision-7 models.
        return 7
    return 1


def worker_run(binary, arm, revision, geometry, tier, threads, log):
    env = environment(geometry,tier,threads)
    env.update(ZEN_S2_PARITY_FEATURES="1",ZEN_S2_SPEEDQ_WORKER=arm,ZENSIM_FORMULA_REV=str(revision),ZEN_S2_RSS_ONLY='1')
    run = subprocess.run(['taskset','-c',CPUSETS[threads],'nice','-n19','ionice','-c3',str(binary)],
                         env=env,capture_output=True,text=True)
    Path(log).write_text(run.stdout+'\nSTDERR:\n'+run.stderr)
    if run.returncode:
        raise RuntimeError(f'worker failed: {log}')
    return json.loads(run.stdout)


class LegacyToleranceFailure(RuntimeError):
    def __init__(self, difference):
        super().__init__('Rev3 feature difference exceeds documented golden tolerance')
        self.difference = difference


def compare_parity(old, rec, revision):
    if rec['input_sha256'] != old['input_sha256']:
        raise RuntimeError('input identity changed')
    if revision in (4, 5):
        if rec['score_bits'] != old['score_bits']:
            raise RuntimeError('strict Rev4/Rev5 score parity bug')
        left,right=old['feature_values'],rec['feature_values']
        assert len(left)==len(right)==420
        if any(not math.isfinite(v) for v in left+right) or any(struct.pack('>d',a)!=struct.pack('>d',b) for a,b in zip(left,right)):
            raise RuntimeError('strict Rev4/Rev5 feature-bit parity bug')
        return None
    left, right = old['feature_values'], rec['feature_values']
    assert len(left) == len(right) == 420, 'Rev3 must check all consumed features'
    assert all(math.isfinite(v) for v in left+right), 'nonfinite features cannot pass tolerance'
    differences = [abs(a-b) for a,b in zip(left,right)]
    normalized = [d/max(1e-6,1e-5*max(abs(a),abs(b))) for a,b,d in zip(left,right,differences)]
    difference = {'score_difference':rec['score']-old['score'], 'max_abs_feature_difference':max(differences),
            'max_tolerance_fraction':max(normalized), 'differing_features':sum(v!=0 for v in differences),
            'tolerance_violations':sum(v>1 for v in normalized),
            'geometry':f"{rec['width']}x{rec['height']}", 'tier':rec['tier'], 'threads':rec['threads']}
    if difference['tolerance_violations']:
        raise LegacyToleranceFailure(difference)
    return difference


def preflight(binary, dest, collect_legacy_failures=False):
    """Run the amended correctness gate before any timing segment."""
    root=Path(dest); root.mkdir(parents=True,exist_ok=False)
    records=[]; baseline={}; legacy=[]
    for geometry in GEOMETRIES:
        for threads in THREADS:
            for tier in TIERS:
                for revision in [5,4,3]:
                    tag=f'{geometry}-{tier}-t{threads}-r{revision}'
                    rec=worker_run(binary,'by_v2fy',revision,geometry,tier,threads,root/f'{tag}.log')
                    rec['revision']=revision
                    records.append(rec)
                    key=(geometry,revision)
                    old=baseline.setdefault(key,rec)
                    try:
                        difference=compare_parity(old,rec,revision)
                    except LegacyToleranceFailure as exc:
                        if not collect_legacy_failures:
                            write(root/'PARITY_FAILED.json',{'status':'STOP_REV3_FEATURE_TOLERANCE','baseline':old,'different':rec,'checked':records,'timing_started':False,'reason':str(exc)})
                            raise
                        difference = exc.difference
                    except RuntimeError as exc:
                        report={'status':'STOP_SCORE_PARITY_BUG','baseline':old,'different':rec,
                                'checked':records,'timing_started':False,'reason':str(exc)}
                        write(root/'PARITY_FAILED.json',report)
                        raise RuntimeError(f'parity failure at {tag}: {exc}') from exc
                    if difference is not None:
                        legacy.append(difference)
        print(f'parity passed {geometry}',flush=True)
    legacy_ok = not any(r['tolerance_violations'] for r in legacy)
    status = 'PASS' if legacy_ok else 'STRICT_PASS_LEGACY_TOLERANCE_FAIL'
    write(root/('PARITY_PASS.json' if legacy_ok else 'PARITY_STRICT_PASS.json'),
          {'status':status,'records':records,'strict_revisions':[4,5],
           'rev3_feature_tolerance':'max(1e-6 abs, 1e-5*scale)',
           'rev3_all_within_tolerance':legacy_ok,'rev3_differences':legacy})




def refresh_activity(activity):
    # The lane that owns the workspace lock; SPEEDQ's own lane by default (E33 runtime: SPEEDQ_LOCK_OWNER).
    owner=os.environ.get('SPEEDQ_LOCK_OWNER','codex SPEEDQ')
    path=Path(__file__).resolve().parents[2]/'.workongoing'
    if not path.exists() or f' {owner}' not in path.read_text():
        raise RuntimeError('SPEEDQ workspace lock ownership changed')
    path.write_text(datetime.now(timezone.utc).isoformat()+f' {owner} '+activity+'\n')


def parity_receipt(path):
    receipt=json.loads(Path(path).read_text())
    assert len(receipt['records']) == 576 and receipt['strict_revisions'] == [4,5]
    assert receipt['status'] in ('PASS','STRICT_PASS_LEGACY_TOLERANCE_FAIL')
    rows={(f"{r['width']}x{r['height']}",r['tier'],int(r['threads']),r['revision']):r for r in receipt['records']}
    expected={(g,t,n,r) for g in GEOMETRIES for t in TIERS for n in THREADS for r in [3,4,5]}
    assert set(rows)==expected, 'exact parity grid must cover every required cell'
    assert {r['model']['source_sha256'] for r in rows.values()} == {SOURCE_SHA}, 'parity must bind the frozen production model'
    for g in GEOMETRIES:
        for revision in (4,5):
            records=[rows[(g,t,n,revision)] for t in TIERS for n in THREADS]
            assert len({r['score_bits'] for r in records})==1, 'strict parity receipt differs'
            assert len({r['input_sha256'] for r in records})==1, 'strict input identity differs'
            for rec in records:
                assert len(rec['feature_values'])==420 and all(math.isfinite(v) for v in rec['feature_values'])
            assert len({b''.join(struct.pack('>d',v) for v in r['feature_values']) for r in records})==1, 'strict feature-bit receipt differs'
    return rows


class TimingNoise(RuntimeError):
    """Preserve a contaminated attempt, then admit a fresh quiet segment."""


def select_clean_rounds(inner, rounds=32):
    """Select one shared index vector without modifying any raw arm or flag."""
    assert rounds == 32, 'SPEEDQ requires 32 retained paired rounds'
    flags = inner['gate_clean']
    values = inner['paired_rounds']
    assert 0 < len(flags) <= 64, 'SPEEDQ attempt must contain at most 64 rounds'
    assert values and all(len(v) == len(flags) for v in values.values()), 'unaligned raw paired rounds'
    if inner['zenbench_unreliable']:
        raise TimingNoise('zenbench marks the whole attempt unreliable')
    retained = [i for i, flag in enumerate(flags) if flag is True][:rounds]
    if len(retained) < rounds:
        raise TimingNoise('fewer than 32 clean rounds by the attempt cap')
    assert retained[-1] == len(flags)-1, 'collector continued past the first 32 clean rounds'
    excluded = [i for i, flag in enumerate(flags) if flag is not True]
    selection = dict(rule=CLEAN_ROUND_RULE, rounds_total=len(flags),
                     rounds_excluded=len(excluded), excluded_indices=excluded,
                     retained_indices=retained)
    return {arm:[series[i] for i in retained] for arm,series in values.items()}, selection


def run_segment(binary, root, geometry, tier, threads, rounds, parity, analyzer, arms=ARMS,
                baseline='by_v2fy_r4', ready_check=None, worker_binaries=None):
    tag=f'{tier}-t{threads}-{geometry}'
    dest=Path(root)/tag
    if (dest/'COMPLETE.json').exists(): return
    dest.mkdir(parents=True,exist_ok=False)
    refresh_activity('quiet gate before '+tag)
    quiet_gate(dest/'quiet-waits.jsonl')
    owners=[]; sockets={}; worker_info={}; streams=[]
    env=environment(geometry,tier,threads)
    with tempfile.TemporaryDirectory(prefix='speedq-',dir=Path.home()/'tmp') as ipc:
        try:
            for name in arms:
                arm='by_v2fy' if name.startswith('by_v2fy_r') else name
                revision=worker_revision(name)
                socket=str(Path(ipc)/name)
                sockets[name]=socket
                log=(dest/f'{name}.worker.log').open('x');streams.append(log)
                worker_env={**env,'ZEN_S2_SPEEDQ_WORKER':arm,'ZENSIM_FORMULA_REV':str(revision),'ZEN_S2_SOCKET':socket}
                proc=subprocess.Popen(['taskset','-c',CPUSETS[threads],'nice','-n19','ionice','-c3',str((worker_binaries or {}).get(name,binary))],env=worker_env,stdout=subprocess.PIPE,stderr=log,text=True)
                owners.append(proc)
                line=proc.stdout.readline()
                if not line: raise RuntimeError(f'{name} failed before READY: {dest}')
                rec=json.loads(line);worker_info[name]=rec
                if arm=='by_v2fy':
                    old=parity[(geometry,tier,threads,revision)]
                    assert rec['model']['source_sha256']==old['model']['source_sha256']==SOURCE_SHA, 'STOP: worker model differs from production parity receipt'
                    assert rec['score_bits']==old['score_bits'] and rec['input_sha256']==old['input_sha256'], 'STOP: worker differs from parity receipt'
                if ready_check is not None:
                    ready_check(name,geometry,tier,threads,rec)
            # Warmup belongs to setup. Check the quiet gate again immediately
            # before measured rounds, when every declared owner is idle.
            gate=quiet_gate(dest/'quiet-waits.jsonl')
            refresh_activity('paired timing '+tag)
            header={'quiet_gate':gate,'uptime':subprocess.check_output(['uptime'],text=True).strip(),
                    'nproc':int(subprocess.check_output(['nproc'],text=True)),
                    'governors':sorted({p.read_text().strip() for p in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_governor')}),
                    'tier':tier,'threads':threads,'cpuset':CPUSETS[threads],'geometry':geometry,
                    'rounds':rounds,'round_cap':64,'round_rule':CLEAN_ROUND_RULE,
                    'arms':arms,'worker_pids':[p.pid for p in owners],
                    'gate_trace':'ZENBENCH_GATE_TRACE' in env,
                    'model_source_sha256':worker_info[baseline]['model']['source_sha256'],
                    'binary_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest()}
            if worker_binaries:
                header['worker_binary_sha256']={name:hashlib.sha256(Path(worker_binaries.get(name,binary)).read_bytes()).hexdigest() for name in arms}
            write(dest/'header.json',header)
            raw=dest/'zenbench.json'
            bench_env={**env,'ZEN_S2_ARMS':','.join(arms),'ZEN_S2_ROUNDS':str(rounds),
                       'ZENBENCH_RESULT_PATH':str(raw),'ZEN_S2_WORKER_SOCKETS':json.dumps(sockets),
                       'ZENBENCH_LAUNCHER_PIDS':','.join(str(p.pid) for p in owners)+','+str(os.getpid())}
            print('paired timing '+tag,flush=True)
            foreign=[]
            with (dest/'zenbench.log').open('x') as log:
                proc=subprocess.Popen(['taskset','-c',CPUSETS[threads],'nice','-n19','ionice','-c3',str(binary)],env=bench_env,stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    state=quiet_state()
                    if any(state['foreign'].values()) or state['training']: foreign.append(state)
                    refresh_activity('paired timing '+tag)
                    time.sleep(1)
                if proc.returncode: raise RuntimeError(f'coordinator failed: {dest}')
            inner=json.loads(raw.with_suffix('.inner.json').read_text())
            try:
                if foreign: raise TimingNoise('foreign build/training during the attempt')
                values, selection = select_clean_rounds(inner, rounds)
            except TimingNoise:
                write(dest/'interference.json',{'foreign':foreign,'admitted':False})
                raise
            write(dest/'interference.json',{'foreign':foreign,'admitted':True})
            candidates=['by_v2fy_r5'] if baseline=='by_v2fy_r4' else [a for a in arms if a!=baseline]
            packets=[{'baseline':values[baseline],'candidate':values[a],
                      'iterations':[1]*rounds,'timer_resolution_ns':inner['timer_resolution_ns']}
                     for a in candidates]
            result=json.loads(subprocess.check_output([str(analyzer)],input=json.dumps(packets),text=True))
            assert len(result)==len(candidates)
            analysis=result[0] if baseline=='by_v2fy_r4' else dict(baseline_arm=baseline,comparisons=dict(zip(candidates,result)))
            write(dest/'paired_analysis.json',analysis)
            write(dest/'COMPLETE.json',{'status':'PASS','rounds':rounds,'paired_alignment_verified':True,'zenbench_gate_clean':True,**selection})
        finally:
            for proc in owners:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=30)
            for log in streams: log.close()


@contextmanager
def segment_lock(path):
    """Coordinate a complete segment with other local heavy-work owners."""
    if path is None:
        yield
        return
    with Path(path).open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                refresh_activity('waiting for segment lock')
                time.sleep(1)
        print(f'segment lock acquired: {path}', flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            print(f'segment lock released: {path}', flush=True)


def timing(binary, root, rounds, parity_path, analyzer, only=None, arms=ARMS, lock=None,
           cells=None, parity_loader=None, baseline='by_v2fy_r4', ready_check=None, worker_binaries=None):
    parity=(parity_loader or parity_receipt)(parity_path)
    grid=cells if cells is not None else [(g,t,n) for t in TIERS for n in THREADS for g in GEOMETRIES]
    pending=deque((g,t,n) for g,t,n in grid if not only or f'{t}-t{n}-{g}' in only)
    while pending:
        geometry,tier,threads=pending.popleft()
        tag=f'{tier}-t{threads}-{geometry}'
        try:
            with segment_lock(lock):
                options={} if baseline=='by_v2fy_r4' and ready_check is None else dict(baseline=baseline,ready_check=ready_check)
                if worker_binaries:
                    options['worker_binaries']=worker_binaries
                run_segment(binary,root,geometry,tier,threads,rounds,parity,analyzer,arms=arms,**options)
        except TimingNoise as exc:
            dest=Path(root)/tag
            archived=dest.with_name(tag+f'.noise-{time.time_ns()}.bak')
            dest.rename(archived)
            print(f'{exc}; excluded attempt preserved at {archived}; retry after remaining segments',flush=True)
            refresh_activity('retry after contaminated '+tag)
            pending.append((geometry,tier,threads))
            time.sleep(10)


# Shared-load RSS (no quiet wait) was the coordinator's instruction on 2026-10-09.
# The 36 REV5PERF5 records written that day carry the earlier tag, which
# misattributes it to the owner; readers accept it, writers never emit it.
RSS_UNDER_LOAD_POLICY = 'coordinator-instructed fresh-process RSS under shared load, 2026-10-09'
RSS_UNDER_LOAD_POLICY_RECORDED = 'owner-approved fresh-process RSS under shared load, 2026-10-09'
# Only the 36 records observed 2026-10-10T01:16:32Z..01:16:52Z may carry it.
RSS_UNDER_LOAD_POLICY_RECORDED_UNTIL = '2026-10-10T01:17:00Z'


def rss(binary, root, parity_path, arms=ARMS, parity_loader=None, ready_check=None,
        geometries=None, thread_counts=None, worker_binaries=None, require_quiet=True):
    parity=(parity_loader or parity_receipt)(parity_path)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for geometry in (geometries if geometries is not None else GEOMETRIES):
        for threads in (thread_counts if thread_counts is not None else [1,32]):
            for name in arms:
                arm='by_v2fy' if name.startswith('by_v2fy_r') else name
                revision=worker_revision(name)
                tag=f'v4x-t{threads}-{geometry}-{name}'
                if (root/f'{tag}.json').exists():continue
                refresh_activity('quiet gate before RSS '+tag)
                gate=quiet_gate(root/'quiet-waits.jsonl') if require_quiet else quiet_state()
                env=environment(geometry,'v4x',threads)
                env.update(ZEN_S2_SPEEDQ_WORKER=arm,ZENSIM_FORMULA_REV=str(revision),ZEN_S2_RSS_ONLY='1')
                executable=(worker_binaries or {}).get(name,binary)
                with (root/f'{tag}.log').open('x') as log:
                    out=subprocess.check_output(['/usr/bin/time','-v','taskset','-c',CPUSETS[threads],'nice','-n19','ionice','-c3',str(executable)],env=env,stderr=log,text=True)
                rec=json.loads(out)
                # Builds since REV5PERF5's floor grid report the pool size in RSS runs.
                assert rec.get('actual_rayon_threads') in (None, threads), 'STOP: RSS worker thread count differs'
                if arm=='by_v2fy':
                    old=parity[(geometry,'v4x',threads,revision)]
                    assert rec['model']['source_sha256']==old['model']['source_sha256']==SOURCE_SHA, 'STOP: RSS model differs from production parity'
                    assert rec['score_bits']==old['score_bits'] and rec['input_sha256']==old['input_sha256'], 'STOP: RSS score/input differs from parity'
                if ready_check is not None:
                    ready_check(name,geometry,'v4x',threads,rec)
                maxrss=next(int(l.rsplit(':',1)[1]) for l in (root/f'{tag}.log').read_text().splitlines() if 'Maximum resident set size (kbytes)' in l)
                write(root/f'{tag}.json',{'geometry':geometry,'arm':name,'tier':'v4x','threads':threads,'max_rss_kib':maxrss,'quiet_gate':gate,'quiet_required':require_quiet,'rss_policy':'quiet-gated' if require_quiet else RSS_UNDER_LOAD_POLICY,'worker':rec,'binary_sha256':hashlib.sha256(Path(executable).read_bytes()).hexdigest()})
                print('RSS '+tag+' '+str(maxrss)+' KiB',flush=True)


def collection_status(root, expected_timing=None, expected_rss=None):
    """Inspect collection markers and the most recent logged quiet check."""
    root=Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    waits=list(root.glob('timing/*/quiet-waits.jsonl'))+list(root.glob('rss/quiet-waits.jsonl'))
    latest=None
    for path in sorted(waits,key=lambda p:p.stat().st_mtime,reverse=True):
        lines=path.read_text().splitlines()
        if lines:
            latest={'path':str(path),'check':json.loads(lines[-1])}
            break
    print(json.dumps({'raw_dir':str(root),
                      'timing_completion_markers':len(list(root.glob('timing/*/COMPLETE.json'))),
                      'expected_timing_segments':expected_timing if expected_timing is not None else len(TIERS)*len(THREADS)*len(GEOMETRIES),
                      'rss_records':len(list(root.glob('rss/v4x-*.json'))),
                      'expected_rss_records':expected_rss if expected_rss is not None else len(GEOMETRIES)*2*len(ARMS),
                      'excluded_attempts':len(list(root.glob('timing/*.bak'))),
                      'latest_quiet_check':latest},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['parity','timing','rss','freeze','status','diagnose'])
    ap.add_argument('--binary'); ap.add_argument('--dest',required=True)
    ap.add_argument('--build-log')
    ap.add_argument('--rounds',type=int,default=32)
    ap.add_argument('--parity'); ap.add_argument('--analyzer'); ap.add_argument('--only',help='comma-separated segment tags')
    ap.add_argument('--collect-legacy-failures',action='store_true',help='complete the strict grid while recording Rev3 tolerance failures; does not admit timings')
    ap.add_argument('--arms',nargs='+',choices=ARMS,default=None,
                    help='timing workers (default: all seven); only ssimulacra2_rs may be omitted')
    ap.add_argument('--lock',type=Path,help='exclusive heavy-work flock held from each timing segment quiet gate through worker cleanup')
    args=ap.parse_args()
    if args.lock is not None and args.mode!='timing':
        ap.error('--lock applies only to timing')
    if args.mode=='timing' and args.rounds!=32:
        ap.error('SPEEDQ timing requires 32 retained clean rounds')
    if args.arms is not None:
        if args.mode!='timing': ap.error('--arms applies only to timing')
        if len(set(args.arms))!=len(args.arms) or not set(ARMS[:-1]).issubset(args.arms):
            ap.error('--arms requires each core arm once; only ssimulacra2_rs may be omitted')
    if args.mode=='status':
        collection_status(args.dest);return
    if args.mode=='freeze':
        if not args.build_log: ap.error('--build-log required for freeze')
        freeze_binary(args.build_log,args.dest);return
    if not args.binary: ap.error('--binary required')
    if args.mode in ('timing','rss','diagnose') and not args.parity: ap.error('--parity required')
    if args.mode in ('timing','diagnose') and not args.analyzer: ap.error('--analyzer required')
    if args.mode=='parity': preflight(args.binary,args.dest,args.collect_legacy_failures)
    elif args.mode=='timing': timing(args.binary,args.dest,args.rounds,args.parity,args.analyzer,args.only.split(',') if args.only else None,args.arms if args.arms is not None else ARMS,lock=args.lock)
    elif args.mode=='diagnose':
        cells={f'{t}-t{n}-{g}':(g,t,n) for t in TIERS for n in THREADS for g in GEOMETRIES}
        if args.only not in cells: ap.error('diagnose requires --only with one grid tag')
        os.environ['ZENBENCH_GATE_TRACE']='1'
        g,t,n=cells[args.only]
        run_segment(args.binary,args.dest,g,t,n,args.rounds,parity_receipt(args.parity),args.analyzer)
    else: rss(args.binary,args.dest,args.parity)



if __name__=='__main__': main()
