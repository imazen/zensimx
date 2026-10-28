#!/usr/bin/env python3
"""COSTCMP configuration for SPEEDQ's existing collectors and admission rules."""
import argparse
import hashlib
import json
import math
import os
import struct
import re
from pathlib import Path
import speedq_run as owner

ARMS = ['by_v2fy_r5', 'zensim_A', 'zensim_B', 'fast_ssim2_main', 'fast_ssim2']
TIERS = ['v4x', 'v3']
THREADS = [1, 4, 16, 32]
FAST_MAIN = '09ec3e7c78bd230b1545572a188e162313c59fda'

SCALING_ARMS = ['by_v2fy_r5', 'by_v2fy_r4', 'zensim_A', 'zensim_B', 'by_v2fy_r5_before']
SCALING_THREADS = [1, 2, 4, 8, 16, 32]
SCALING_GEOMETRIES = ['1024x1024', '2048x2048', '4096x4096', '1920x1080']
BUDGET_ARMS = ['by_v2fy_r5_before', 'by_v2fy_r5_b64', 'by_v2fy_r5_b128', 'by_v2fy_r5_b256']
# Floor grid (timing, its preflight and floor-arm RSS): the four measured arms
# plus the 128 MiB source with REV5_MIN_JOB_SLOTS = 3 (the shipped floor) and
# = 17 (never queues: the original route). Widths 4608 and 6144 give 4 and 3
# slots at 128 MiB, 4096 gives 5, 8192 gives 2 (frozen arm) or the original route.
BUDGET_FLOOR_SLOTS = {'by_v2fy_r5_floor3': 3, 'by_v2fy_r5_floor17': 17}
BUDGET_FLOOR_ARM = 'by_v2fy_r5_floor3'
BUDGET_ORIGINAL_ARM = 'by_v2fy_r5_floor17'
BUDGET_FLOOR_ARMS = BUDGET_ARMS + list(BUDGET_FLOOR_SLOTS)
# The exact floor hunk; a floor arm's source must be the frozen 128 MiB
# candidate with only this change applied (any slot count).
FLOOR_BLOCK = """// Below this many slots the walk keeps the original route. That route runs a
// strip's three channels in parallel (`fuse_channels`) and each channel's v1
// bands in parallel (`band_parallel`); a queued job runs its bands serially.
// Which is faster at a given slot count is measured, not assumed: see
// benchmarks/rev5perf5_budget_design_2026-10-09.md. Both routes produce
// identical bits; the floor changes scheduling only.
#[cfg(feature = "threads")]
const REV5_MIN_JOB_SLOTS: usize = {slots};

"""
FLOOR_LIMIT = ('    let limit = rev5_job_bytes(max_n).map_or(0, |bytes| threads.min(16).min(budget / bytes));\n'
               '    if limit < REV5_MIN_JOB_SLOTS {{ 0 }} else {{ limit }}\n')
CAP_LIMIT = '    rev5_job_bytes(max_n).map_or(0, |bytes| threads.min(16).min(budget / bytes))\n'
BUDGET_THREADS = [8, 16, 32]
BUDGET_GEOMETRIES = ['1024x1024', '4096x4096', '8192x4096']
BUDGET_FLOOR_GEOMETRIES = ['1024x1024', '4096x4096', '4608x4096', '6144x4096', '8192x4096']
# E33 registration section 9.3: candidates vs production seed 0, v4x and v3, one thread; RSS at 1024^2.
E33_ARMS = ['by_v2fy_r5', 'e33_a', 'e33_c']
E33_GEOMETRIES = ['64x64', '256x256', '1024x1024']
# QUAL-A (MODEL_SELECTION_SCORECARD scalar performance + spatial/memory cost): one pinned worker on v4x, uncached
# complete score at 1024^2 and 2048^2 against frozen D and fast-ssim2, plus cached-reference score+map arms.
QUAL_ARMS = ['by_v2fy_r5', 'e33_a', 'e33_c', 'zensim_D', 'fast_ssim2', 'fast_ssim2_main',
             'e33_a_map', 'e33_c_map', 'e33_seed0_map']
QUAL_GEOMETRIES = ['1024x1024', '2048x2048']
# Build: zensim-bench/r7-runtime/Cargo.toml (`just costcmp-r7-build`).
# The two curated revision-7 models (byconstruction record, "Runtime gate for the two curated models"): QUAL-A's
# scorecard rows (one pinned worker, 1024^2 and 2048^2), inside a size sweep (64^2 to 4096^2) at one and eight threads.
# Frozen D is the paired baseline, fast-ssim2 0.8.2 and main are the peer bars, butteraugli's 3-norm and the current
# default B are report-only peers. Cached-reference scalar arms are reported separately; the map arms are the spatial
# row's cached-reference score + map; the baseline arms (no scoring) anchor incremental RSS.
R7_ARMS = ['zensim_D', 'r7_general', 'r7_codec', 'fast_ssim2', 'fast_ssim2_main', 'zensim_B_rev5', 'e34_butter_scalar',
           'r7_general_cached', 'r7_codec_cached', 'r7_general_map', 'r7_codec_map']
R7_GEOMETRIES = ['64x64', '256x256', '1024x1024', '2048x2048', '4096x4096']
R7_THREADS = [1, 8]
# Fresh-process RSS: every geometry at one thread, the scorecard sizes and 4096^2 at eight.
R7_RSS = [(R7_GEOMETRIES, [1]), (['1024x1024', '2048x2048', '4096x4096'], [8])]
R7_BASELINES = ['r7_baseline_general', 'r7_baseline_codec']
R7_SURFACES = {'r7_general_cached': 'r7_general', 'r7_codec_cached': 'r7_codec',
               'r7_general_map': 'r7_general', 'r7_codec_map': 'r7_codec'}
DEFAULT_B = '89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e'


def configuration(scaling=False, budget=False, floor=False, e33=False):
    assert not (scaling and budget), 'select one COSTCMP grid'
    if e33 == 'qual':
        assert not (scaling or budget or floor), 'select one COSTCMP grid'
        return ['v4x'], [1], QUAL_GEOMETRIES, QUAL_ARMS, 'costcmp-qual-preflight-v1'
    if e33 == 'r7':
        assert not (scaling or budget or floor), 'select one COSTCMP grid'
        return ['v4x'], R7_THREADS, R7_GEOMETRIES, R7_ARMS, 'costcmp-r7-preflight-v1'
    if e33:
        assert not (scaling or budget or floor), 'select one COSTCMP grid'
        return TIERS, [1], E33_GEOMETRIES, E33_ARMS, 'costcmp-e33-preflight-v1'
    assert budget or not floor, 'the floor arm belongs to the budget grid'
    if budget:
        if floor:
            return ['v4x'], BUDGET_THREADS, BUDGET_FLOOR_GEOMETRIES, BUDGET_FLOOR_ARMS, 'costcmp-budget-preflight-v3'
        return ['v4x'], BUDGET_THREADS, BUDGET_GEOMETRIES, BUDGET_ARMS, 'costcmp-budget-preflight-v1'
    return (TIERS, SCALING_THREADS if scaling else THREADS,
            SCALING_GEOMETRIES if scaling else owner.GEOMETRIES,
            SCALING_ARMS if scaling else ARMS,
            'costcmp-scaling-preflight-v1' if scaling else 'costcmp-preflight-v1')


def cells(scaling=False, budget=False, floor=False, e33=False):
    tiers, threads, geometries, _, _ = configuration(scaling, budget, floor, e33)
    return [(g,t,n) for t in tiers for n in threads for g in geometries]


def floor_base(source):
    """(the 128 MiB candidate source a floor arm was made from, its slot floor), or an AssertionError."""
    found = re.findall(rb'const REV5_MIN_JOB_SLOTS: usize = (\d+);', source)
    assert len(found) == 1, 'floor hunk missing'
    block = FLOOR_BLOCK.format(slots=int(found[0])).encode()
    limit = FLOOR_LIMIT.format().encode()
    assert source.count(block) == 1 and source.count(limit) == 1, 'floor hunk missing'
    return source.replace(block, b'').replace(limit, CAP_LIMIT.encode()), int(found[0])


def budget_inventory(path, floor=False):
    manifest = json.loads(Path(path).read_text())
    assert manifest['schema'] == ('rev5perf5-binaries-v3' if floor else 'rev5perf5-binaries-v1')
    assert set(manifest['arms']) == set(BUDGET_FLOOR_ARMS if floor else BUDGET_ARMS)
    dependencies = []
    candidate_sources = []
    binaries = {}
    sources = {}
    for arm, rec in manifest['arms'].items():
        binary = Path(rec['binary'])
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == rec['binary_sha256']
        artifact = json.loads(Path(rec['artifact']).read_text())
        assert artifact['binary_sha256'] == rec['binary_sha256']
        dependencies.append(artifact['dependencies'])
        source = Path(rec['runtime_source']).read_bytes()
        assert hashlib.sha256(source).hexdigest() == rec['runtime_source_sha256']
        sources[arm] = source
        binaries[arm] = binary
        if arm in BUDGET_FLOOR_SLOTS:
            continue
        caps = re.findall(rb'const REV5_JOB_BUDGET_BYTES: usize = (\d+) \* 1024 \* 1024;', source)
        expected = [] if arm.endswith('_before') else [arm.rsplit('b',1)[1].encode()]
        assert caps == expected, 'source byte cap does not match candidate identity'
        if caps:
            candidate_sources.append(re.sub(rb'const REV5_JOB_BUDGET_BYTES: usize = \d+ \* 1024 \* 1024;',
                                            b'const REV5_JOB_BUDGET_BYTES: usize = CANDIDATE;', source))
    if floor:
        for arm, slots in BUDGET_FLOOR_SLOTS.items():
            assert floor_base(sources[arm]) == (sources['by_v2fy_r5_b128'], slots), f'{arm} differs from the 128 MiB candidate beyond a floor of {slots}'
    assert all(value == dependencies[0] for value in dependencies), 'candidate dependencies differ'
    assert len(set(candidate_sources)) == 1, 'candidate source differs beyond the private byte cap'
    return binaries

def signature(rec):
    return (rec['input_sha256'], rec['score_bits'], rec['model'])

def validate_ready(rows, name, geometry, tier, threads, rec):
    old = rows[(geometry, tier, threads, name)]
    assert (f"{rec['width']}x{rec['height']}",rec['tier'],int(rec['threads'])) == (geometry,tier,threads), 'STOP: worker coordinates differ from COSTCMP grid cell'
    assert signature(rec) == signature(old), 'STOP: worker input/score/model differs from COSTCMP preflight'
    assert rec['arm'] == ('by_v2fy' if name.startswith('by_v2fy_r') else name)
    if name.startswith('by_v2fy_r'):
        assert rec['model']['source_sha256'] == owner.SOURCE_SHA
        assert rec['model']['revision'] == str(owner.worker_revision(name))
    elif name.startswith('e33_'):
        label = name[4:].removesuffix('_map').upper()
        assert rec['model']['source_sha256'] == os.environ[f"ZEN_S2_E33_SHA_{label}"]
        assert rec['model']['revision'] == '5'
    elif name.startswith('r7_'):
        label = 'GENERAL' if 'general' in name else 'CODEC'
        assert rec['model']['source_sha256'] == os.environ[f"ZEN_S2_R7_SHA_{label}"]
        assert rec['model']['revision'] == '7'
    elif name == 'zensim_B_rev5':
        assert (rec['model']['revision'], rec['model']['source_sha256']) == ('5', DEFAULT_B)
    elif name == 'e34_butter_scalar':
        assert (rec['model']['peer'], rec['model']['score']) == ('butteraugli0.9.3', 'pnorm3')
    elif name in ('zensim_A', 'zensim_B', 'zensim_D'):
        assert rec['model']['revision'] == '1'
        assert len(rec['model']['source_sha256']) == 64
    elif name == 'fast_ssim2_main':
        assert rec['model']['commit'] == FAST_MAIN
    else:
        assert rec['model']['version'] == '0.8.2'

def receipt(path, scaling=False, budget=False, floor=False, e33=False):
    value = json.loads(Path(path).read_text())
    tiers, threads, geometries, arms, schema = configuration(scaling, budget, floor, e33)
    assert value['schema'] == schema and value['status'] == 'PASS'
    rows = {(f"{r['width']}x{r['height']}", r['tier'], int(r['threads']), r['name']): r for r in value['records']}
    assert len(rows) == len(value['records']) == len(cells(scaling,budget,floor,e33))*len(arms)
    assert set(rows) == {(g,t,n,a) for g,t,n in cells(scaling,budget,floor,e33) for a in arms}
    if e33:  # each candidate serves bit-identically across dispatch tiers, as production does
        for g in geometries:
            for a in arms[1:]:
                assert len({rows[(g,t,n,a)]['score_bits'] for t in tiers for n in threads}) == 1, f'strict {a} score parity differs'
    if e33 == 'qual':  # the cached-reference map arm serves the same score as its model's uncached arm
        for g,t,n in cells(scaling,budget,floor,e33):
            for m,u in (('e33_a_map','e33_a'),('e33_c_map','e33_c'),('e33_seed0_map','by_v2fy_r5')):
                assert rows[(g,t,n,m)]['score_bits'] == rows[(g,t,n,u)]['score_bits'], f'{m} score differs from {u}'
    if e33 == 'r7':  # cached-reference and map arms serve their model's uncached score
        for g,t,n in cells(scaling,budget,floor,e33):
            for m,u in R7_SURFACES.items():
                assert rows[(g,t,n,m)]['score_bits'] == rows[(g,t,n,u)]['score_bits'], f'{m} score differs from {u}'
    # The revision-7 grid has no production by_v2fy arm; its parity is the surface check above.
    groups = [] if e33 == 'r7' else [arms] if budget else ([['by_v2fy_r5','by_v2fy_r5_before'], ['by_v2fy_r4']] if scaling else [[ARMS[0]]])
    for g in geometries:
        for group in groups:
            production = [rows[(g,t,n,a)] for t in tiers for n in threads for a in group]
            assert len({r['input_sha256'] for r in production}) == 1
            assert len({r['score_bits'] for r in production}) == 1, f'strict {group} score parity differs'
            bits = []
            for rec in production:
                f = rec['feature_values']
                assert len(f) == 420 and all(math.isfinite(v) for v in f)
                bits.append(b''.join(struct.pack('>d',v) for v in f))
            assert len(set(bits)) == 1, f'strict {group} consumed-feature parity differs'
    for (g,t,n,a),rec in rows.items():
        assert math.isfinite(rec['score'])
        assert rec['input_sha256'] == rows[(g,t,n,arms[0])]['input_sha256']
        validate_ready(rows,a,g,t,n,rec)
    return rows

def production_rows(rows):
    return {(g,t,n,owner.worker_revision(a)):rec for (g,t,n,a),rec in rows.items() if a.startswith('by_v2fy_r')}

def preflight(binary, dest, scaling=False, before_binary=None, budget=False, worker_binaries=None, floor=False, e33=False):
    root=Path(dest);root.mkdir(parents=True,exist_ok=False);records=[]
    _, _, _, arms, schema = configuration(scaling,budget,floor,e33)
    for g,t,n in cells(scaling,budget,floor,e33):
        for name in arms:
            arm='by_v2fy' if name.startswith('by_v2fy_r') else name
            revision = owner.worker_revision(name)
            executable = (worker_binaries or {}).get(name, before_binary if name=='by_v2fy_r5_before' else binary)
            rec=owner.worker_run(executable,arm,revision,g,t,n,root/f'{t}-t{n}-{g}-{name}.log')
            rec['name']=name;records.append(rec)
        print(f'preflight {t}-t{n}-{g}',flush=True)
    path=root/'PREFLIGHT_PASS.json'
    owner.write(path,dict(schema=schema,status='PASS',records=records,strict_rev5_cells=len(cells(scaling,budget,floor,e33)),consumed_features=420))
    try:
        receipt(path,scaling,budget,floor,e33)
    except Exception:
        path.rename(root/'PREFLIGHT_FAILED.json')
        raise

def r7_summary(root):
    """The scorecard's runtime rows for the two curated revision-7 models (QUAL-A's bars, p95 owner and layout:
    root/runtime/timing, root/runtime/rss) plus the size/thread sweep and peak RSS of every arm.
    Writes R7_RUNTIME.json and R7_RUNTIME.md in root."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
    from qual_a_report import runtime
    root = Path(root)
    rt = runtime(root)
    assert rt is not None, 'every timing cell must be complete'
    assert set(rt['cells']) == {f'{t}-t{n}-{g}' for g, t, n in cells(e33='r7')}, 'timing cells differ from the grid'
    ok = lambda b: 'PASS' if b else 'FAIL'
    rows = {}
    lines = ['## Scorecard rows (one pinned worker, v4x)', '',
             '| Model | Geometry | Uncached p95 | vs frozen D | vs fast-ssim2 0.8.2 | Cached-reference p95 | '
             'Score + map p95 (× uncached) | Peak incremental RSS (score / map) | Scalar | Spatial | Memory |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---|---|']
    for cell, bar_ms, px in (('v4x-t1-1024x1024', 50, 1024 * 1024), ('v4x-t1-2048x2048', 200, 2048 * 2048)):
        c = rt['cells'][cell]
        timing = root / 'runtime' / 'timing' / cell
        header = json.loads((timing / 'header.json').read_text())
        interference = json.loads((timing / 'interference.json').read_text())
        peers = {a: c[a]['p95_ms'] for a in ('zensim_D', 'fast_ssim2', 'fast_ssim2_main', 'zensim_B_rev5', 'e34_butter_scalar')}
        bar_kib = (128 * px + 64 * 2**20) / 1024
        for m in ('general', 'codec'):
            arm = f'r7_{m}'
            u, cached, mapped = (c[a]['p95_ms'] for a in (arm, arm + '_cached', arm + '_map'))
            base = rt['rss_kib'][f'{cell}-r7_baseline_{m}']
            inc = {s: rt['rss_kib'][f'{cell}-{a}'] - base for s, a in (('score', arm), ('cached', arm + '_cached'), ('map', arm + '_map'))}
            r = dict(cell=cell, model=m, rounds=c[arm]['n'], uncached_p95_ms=u, uncached_median_ms=c[arm]['median_ms'],
                     cached_p95_ms=cached, map_p95_ms=mapped, peers_p95_ms=peers,
                     scalar=dict(bar_ms=bar_ms, ratio_D=u / peers['zensim_D'], ratio_fast_ssim2=u / peers['fast_ssim2'],
                                 passes=u <= bar_ms and u <= 1.25 * peers['zensim_D'] and u <= peers['fast_ssim2']),
                     spatial=dict(ratio=mapped / u, passes=mapped <= 3 * u),
                     memory=dict(bar_kib=bar_kib, baseline_kib=base, incremental_kib=inc, passes=max(inc.values()) <= bar_kib),
                     quiet_gate=header['quiet_gate'], interference_admitted=interference['admitted'])
            rows[f'{m} {cell}'] = r
            lines.append(f"| {m} | {cell.split('-')[-1]} | {u:.2f} ms | {r['scalar']['ratio_D']:.2f}× | "
                         f"{r['scalar']['ratio_fast_ssim2']:.2f}× | {cached:.2f} ms | {mapped:.2f} ms ({r['spatial']['ratio']:.2f}×) | "
                         f"{inc['score']:,} / {inc['map']:,} KiB (bar {bar_kib:,.0f}) | {ok(r['scalar']['passes'])} | "
                         f"{ok(r['spatial']['passes'])} | {ok(r['memory']['passes'])} |")
    order = [f'v4x-t{n}-{g}' for n in R7_THREADS for g in R7_GEOMETRIES]
    lines += ['', '## Median / p95 per call (ms), 32 retained paired rounds', '',
              '| Arm | ' + ' | '.join(c.removeprefix('v4x-') for c in order) + ' |', '|---|' + '---:|' * len(order)]
    for a in R7_ARMS:
        lines.append(f'| {a} | ' + ' | '.join(f"{rt['cells'][c][a]['median_ms']:.2f} / {rt['cells'][c][a]['p95_ms']:.2f}"
                                               for c in order) + ' |')
    blocks = [f'v4x-t{n}-{g}' for gs, ns in R7_RSS for n in ns for g in gs]
    lines += ['', '## Fresh-process peak RSS (MiB)', '',
              '| Arm | ' + ' | '.join(c.removeprefix('v4x-') for c in blocks) + ' |', '|---|' + '---:|' * len(blocks)]
    for a in R7_BASELINES + R7_ARMS:
        lines.append(f'| {a} | ' + ' | '.join(f"{rt['rss_kib'][f'{c}-{a}'] / 1024:.1f}" for c in blocks) + ' |')
    owner.write(root / 'R7_RUNTIME.json', dict(schema='costcmp-r7-runtime-v1', rows=rows, runtime=rt))
    with (root / 'R7_RUNTIME.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['parity','timing','rss','status','summary'])
    ap.add_argument('--binary',type=Path);ap.add_argument('--dest',type=Path,required=True)
    ap.add_argument('--parity',type=Path);ap.add_argument('--analyzer',type=Path)
    ap.add_argument('--lock',type=Path);ap.add_argument('--only')
    ap.add_argument('--scaling',action='store_true',help='Rev5/Rev4/A/B plus frozen before Rev5; 24 scaling cells and 24 low-thread controls')
    ap.add_argument('--before-binary',type=Path)
    ap.add_argument('--budget-grid',action='store_true',help='64/128/256 MiB plus uncapped Rev5 at 8/16/32 threads and three geometries')
    ap.add_argument('--budget-binaries',type=Path,help='frozen executable/source inventory for the byte-budget grid')
    ap.add_argument('--rss-under-load',action='store_true',help='coordinator-instructed fresh-process RSS without a quiet wait; timing gates stay unchanged')
    ap.add_argument('--floor-arm',action='store_true',help='floor grid: add the floor-3 and floor-17 (original route) builds and the 3/4-slot widths (v3 inventory); RSS measures only the floor arms')
    ap.add_argument('--e33-grid',type=Path,metavar='CANDIDATES_JSON',help='E33 9.3 grid; JSON {"a":{"path","sha256"},"c":{...}} pins the candidate bakes')
    ap.add_argument('--qual-grid',type=Path,metavar='CANDIDATES_JSON',help='QUAL-A scorecard runtime grid; JSON pins a, c and seed0')
    ap.add_argument('--r7-grid',type=Path,metavar='CANDIDATES_JSON',help='the same scorecard runtime grid for the two curated revision-7 models; JSON pins general and codec')
    args=ap.parse_args()
    if sum(g is not None for g in (args.e33_grid,args.qual_grid,args.r7_grid))>1:ap.error('select one COSTCMP grid')
    e33='qual' if args.qual_grid is not None else 'r7' if args.r7_grid is not None else args.e33_grid is not None
    pins=next((g for g in (args.qual_grid,args.r7_grid,args.e33_grid) if g is not None),None)
    if args.mode=='summary' and e33!='r7':ap.error('summary applies to --r7-grid')
    if e33:
        if args.scaling or args.budget_grid or args.floor_arm or args.rss_under_load:ap.error('--e33-grid/--qual-grid/--r7-grid is its own grid')
        labels=json.loads(pins.read_text())
        assert set(labels)==({'a','c','seed0'} if e33=='qual' else {'general','codec'} if e33=='r7' else {'a','c'})
        prefix='ZEN_S2_R7' if e33=='r7' else 'ZEN_S2_E33'
        for label,rec in labels.items():
            assert hashlib.sha256(Path(rec['path']).read_bytes()).hexdigest()==rec['sha256']
            os.environ[f'{prefix}_BAKE_{label.upper()}']=rec['path'];os.environ[f'{prefix}_SHA_{label.upper()}']=rec['sha256']
    if args.floor_arm and not args.budget_grid:ap.error('--floor-arm applies only to the budget grid')
    if args.floor_arm and args.rss_under_load:ap.error('floor-arm RSS keeps the quiet gate')
    if args.rss_under_load and (args.mode!='rss' or not args.budget_grid):ap.error('--rss-under-load applies only to budget-grid RSS')
    if args.scaling and args.budget_grid:ap.error('select one COSTCMP grid')
    if args.budget_grid and args.budget_binaries is None:ap.error('--budget-grid requires --budget-binaries')
    if args.budget_binaries is not None and not args.budget_grid:ap.error('--budget-binaries requires --budget-grid')
    if args.scaling and args.before_binary is None:ap.error('--scaling requires --before-binary')
    if args.mode=='status':
        expected_timing = 2 if e33=='qual' else len(cells(e33='r7')) if e33=='r7' else 6 if e33 else (15 if args.floor_arm else 9) if args.budget_grid else (48 if args.scaling else 64)
        expected_rss = 2*len(QUAL_ARMS) if e33=='qual' else sum(len(g)*len(n) for g,n in R7_RSS)*(len(R7_ARMS)+len(R7_BASELINES)) if e33=='r7' else 3 if e33 else (36+30 if args.floor_arm else 36) if args.budget_grid else (48 if args.scaling else 80)
        owner.collection_status(args.dest,expected_timing=expected_timing,expected_rss=expected_rss);return
    if args.mode=='summary':
        r7_summary(args.dest);return
    if args.binary is None:ap.error('--binary is required')
    binaries = budget_inventory(args.budget_binaries,args.floor_arm) if args.budget_grid else ({'by_v2fy_r5_before':args.before_binary} if args.scaling else None)
    if args.budget_grid:assert args.binary.resolve() in {p.resolve() for p in binaries.values()}
    if args.mode=='parity':preflight(args.binary,args.dest,args.scaling,args.before_binary,args.budget_grid,binaries,args.floor_arm,e33);return
    if args.parity is None:ap.error('--parity is required')
    rows=receipt(args.parity,args.scaling,args.budget_grid,args.floor_arm,e33)
    check=lambda name,g,t,n,rec:validate_ready(rows,name,g,t,n,rec)
    loader=lambda path:production_rows(receipt(path,args.scaling,args.budget_grid,args.floor_arm,e33))
    _, _, _, arms, _ = configuration(args.scaling,args.budget_grid,args.floor_arm,e33)
    if args.mode=='timing':
        if args.analyzer is None or args.lock is None:ap.error('--analyzer and --lock are required')
        chosen=[c for c in cells(args.scaling,args.budget_grid,args.floor_arm,e33) if args.only is None or f'{c[1]}-t{c[2]}-{c[0]}'==args.only]
        assert chosen,'unknown COSTCMP grid tag'
        owner.timing(args.binary,args.dest,32,args.parity,args.analyzer,arms=arms,lock=args.lock,
                     cells=chosen,parity_loader=loader,baseline=arms[0],ready_check=check,
                     worker_binaries=binaries)
    else:
        if args.lock is None:ap.error('--lock is required')
        with owner.segment_lock(args.lock):
            if e33=='r7':
                for geometries,threads in R7_RSS:  # baseline arms (no scoring) first, for incremental RSS
                    owner.rss(args.binary,args.dest,args.parity,arms=R7_BASELINES,parity_loader=loader,ready_check=None,
                              geometries=geometries,thread_counts=threads)
                    owner.rss(args.binary,args.dest,args.parity,arms=R7_ARMS,parity_loader=loader,ready_check=check,
                              geometries=geometries,thread_counts=threads)
                return
            if e33=='qual':  # baseline arms (inputs + model + scorer, no scoring) for incremental RSS
                owner.rss(args.binary,args.dest,args.parity,arms=['e33_baseline_a','e33_baseline_c','e33_baseline_seed0'],
                          parity_loader=loader,ready_check=None,geometries=QUAL_GEOMETRIES,thread_counts=[1])
            owner.rss(args.binary,args.dest,args.parity,
                      arms=QUAL_ARMS if e33=='qual' else E33_ARMS if e33 else (list(BUDGET_FLOOR_SLOTS) if args.floor_arm else arms) if args.budget_grid else (['by_v2fy_r5','by_v2fy_r5_before'] if args.scaling else ARMS),
                      parity_loader=loader,ready_check=check,
                      geometries=QUAL_GEOMETRIES if e33=='qual' else ['1024x1024'] if e33 else (BUDGET_FLOOR_GEOMETRIES if args.floor_arm else BUDGET_GEOMETRIES) if args.budget_grid else (SCALING_GEOMETRIES if args.scaling else None),
                      thread_counts=[1] if e33 else BUDGET_THREADS if args.budget_grid else (SCALING_THREADS if args.scaling else None),
                      worker_binaries=binaries,require_quiet=not args.rss_under_load)

if __name__=='__main__':main()
