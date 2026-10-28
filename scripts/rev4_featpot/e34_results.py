"""Reduce immutable E34 measurements and render the registered readout.

All inferential results are read from the bound Rust owners. Engineering
thresholds and the sequential verdict are exactly the frozen registration.
"""
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

from e34_verdict import ARMS, GATE_KEYS, mean_guard_sensitivity, steer_rule, verdict
from e33_gates import _verdict_numbers, identity_rows
from qual_a_report import p95

ROOT = Path('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demos'))
import speedq_run


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def validate_statistical_reports(stats):
    """A complete decision never substitutes for an undefined required report."""
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    for test in stats['tests']:
        name = f'test {test["test"]}'
        for label, record in [('seed-unit', test['seed_unit']),
                              ('W2 seed-unit', test['w2']['seed_unit'])]:
            assert all(finite(record.get(k)) for k in ['delta', 'se']), (
                f'{name}: registered {label} report undefined; stop without substitution')
            assert record['n'] == 10
        intervals = [test['theta'], test['reference_only'], test['w2']['interval']]
        intervals += [r['interval'] for r in test['dataset_guards']]
        for row in test['pair_classes'] or []:
            assert finite(row['seed_unit_se']), f'{name}: pair-class seed SE undefined'
            intervals.append(row['interval'])
        for interval in intervals:
            assert all(finite(interval.get(k)) for k in ['estimate', 'lo', 'hi', 'se']), (
                f'{name}: registered report interval undefined; stop without substitution')
            assert interval['n_finite'] > 0
            student = interval['student_interval']
            assert student is not None and len(student) == 2 and all(map(finite, student)), (
                f'{name}: registered Student interval undefined; stop without substitution')


def penalties():
    inventory = read(ROOT / 'preread.json')
    cells = [c for c in inventory['cells'] if '-k' in c['jobset']]
    assert len(cells) == 90
    records = []
    pattern = re.compile(r'^E34 K epoch=(\d+) images=(\d+) pairs=(\d+) '
        r'skipped_denominator_lt_1e-10=(\d+) loss=(\S+) blocks_8_16_32_64=(\[.*\])$')
    for c in cells:
        cell = Path(c['cell'])
        assert sha(cell / 'result.json') == c['result_sha256']
        assert sha(cell / 'fleet_receipt.json') == c['fleet_receipt_sha256']
        path = cell / 'train.log'
        assert sha(path) == read(cell / 'fleet_receipt.json')['files']['train.log']
        epochs = []
        for line in path.read_text().splitlines():
            if not line.startswith('E34 K '):
                continue
            m = pattern.fullmatch(line)
            assert m, f'unrecognized penalty record: {path}'
            epoch, images, pairs, skipped = map(int, m.group(1, 2, 3, 4))
            blocks = json.loads(m[6])
            assert images == 512 and pairs == 4096 and len(blocks) == 4 and sum(blocks) == pairs
            epochs.append(dict(epoch=epoch, images=images, pairs=pairs,
                               skipped_denominator_lt_1e_10=skipped, loss=float(m[5]), blocks=blocks))
        # log_line writes stderr and the trainer's accumulated text log, which
        # the CLI prints at completion. Admit the two byte-identical copies;
        # retain that multiplicity as provenance rather than count two fits.
        assert len(epochs) == 240 and epochs[:120] == epochs[120:], f'K log copies disagree: {path}'
        epochs = epochs[:120]
        assert [r['epoch'] for r in epochs] == list(range(120)), f'incomplete K log: {path}'
        records.append(dict(jobset=c['jobset'], fold=c['fold'], seed=c['seed'],
            path=str(path), sha256=sha(path), identical_log_copies=2, epochs=epochs))
    value = dict(schema='e34-penalty-logs-v1', report_only=True,
          cells=len(records), epoch_records=sum(len(c['epochs']) for c in records), records=records)
    path = ROOT / 'penalty-logs.json'
    if path.exists():
        assert read(path) == value, 'preserved penalty report differs'
    else:
        write(path, value)
    print(json.dumps(dict(cells=len(records), epoch_records=sum(len(c['epochs']) for c in records))))


def runtime(arm):
    root = ROOT / 'runtime' / arm
    expected = {f'{t}-t1-{g}' for t in ['v4x','v3'] for g in ['64x64','256x256','1024x1024']}
    spatial = ['v4x-t1-1024x1024','v4x-t1-2048x2048']
    cells, spatial_cells, memory = {}, {}, {}
    for name in sorted(expected | set(spatial)):
        path = root / 'timing' / name
        done = read(path / 'COMPLETE.json')
        assert done['status'] == 'PASS' and done['rounds'] == 32
        raw = read(path / 'zenbench.inner.json')
        values, selection = speedq_run.select_clean_rounds(raw)
        assert done['retained_indices'] == selection['retained_indices']
        result = read(path / 'paired_analysis.json')
        assert result['baseline_arm'] == 'e34_control'
        c = result['comparisons']['e34_candidate']
        slower = c['ci_lower'] > .02*c['baseline']['mean'] and not c['resolution_limited']
        if name in expected:
            cells[name] = dict(comparison=c, plus2pct_ns=.02*c['baseline']['mean'], pass_=not slower)
        if name in spatial:
            uncached = p95(values['e34_candidate'])
            mapped = p95(values['e34_candidate_map'])
            spatial_cells[name] = dict(uncached_p95_ns=uncached, map_p95_ns=mapped,
                                       ratio=mapped/uncached, pass_=mapped<=3*uncached)
    for geometry in ['1024x1024','2048x2048']:
        prefix = root / 'rss' / f'v4x-t1-{geometry}'
        base = read(str(prefix)+'-e34_candidate_baseline.json')
        mapped = read(str(prefix)+'-e34_candidate_map.json')
        scored = read(str(prefix)+'-e34_candidate.json')
        assert all(r['quiet_gate']['admitted'] and r['quiet_required'] for r in [base,mapped,scored])
        width,height=map(int,geometry.split('x'))
        incremental=mapped['max_rss_kib']-base['max_rss_kib']
        bar=(128*width*height+64*2**20)//1024
        memory[geometry]=dict(baseline_kib=base['max_rss_kib'],map_peak_kib=mapped['max_rss_kib'],
                              score_peak_kib=scored['max_rss_kib'],map_incremental_kib=incremental,
                              bar_kib=bar,pass_=incremental<=bar)
    return dict(runtime=dict(pass_=all(r['pass_'] for r in cells.values()),cells=cells),
                spatial=dict(pass_=all(r['pass_'] for r in spatial_cells.values()),cells=spatial_cells),
                memory=dict(pass_=all(r['pass_'] for r in memory.values()),cells=memory))


def build():
    integrity = read(ROOT / 'installed-integrity-final.json')
    assert integrity['unchanged'] and integrity['cells']==305 and integrity['jobsets']==18
    assert integrity['preread_sha256']==sha(ROOT / 'preread.json')
    stats = read(ROOT / 'statistics.json')
    binding = read(ROOT / 'statistics-binding.json')
    assert stats['input_sha256'] == binding['input_sha256'] == sha(ROOT / 'panel-input.json')
    assert sha(binding['panel']) == binding['panel_sha256']
    assert all(t['state']=='COMPLETE' for t in stats['tests']), 'registered hypothesis incomplete'
    validate_statistical_reports(stats)
    report = read(ROOT / 'reports.json')
    report_execution = read(ROOT / 'reports-execution.json')
    original_report = read(ROOT / 'report-input.json')
    executed_report = read(ROOT / 'report-input-r7900x.json')
    assert report_execution['original_input_sha256'] == sha(ROOT / 'report-input.json')
    assert report['input_sha256'] == report_execution['executed_input_sha256'] == sha(ROOT / 'report-input-r7900x.json')
    original_report.pop('paired_input')
    executed_report.pop('paired_input')
    assert original_report == executed_report, 'only the paired-file locator may change during relocation'
    assert original_report['paired_sha256'] == sha(ROOT / 'panel-input.json')
    assert report_execution['binary_sha256'] == sha(ROOT / 'bin/panel-reports')
    models = read(ROOT / 'full-models.json')
    for r in models.values():
        assert sha(r['path']) == r['sha256']
    proof, candidates = identity_rows(ROOT / 'gates/identity-s0.jsonl')
    assert len(proof)==620
    candidate_shas=[r['sha256'] for r in candidates]
    hundred=struct.unpack('<q',struct.pack('<d',100.))[0]
    packs=read(ROOT / 'pack-checks.json')['models']
    k4=read(ROOT / 'gates/K4_FULL.json')['arms']
    steering={}
    for name in models:
        doc=read(ROOT / 'steer135' / name / 'result.json')
        assert doc['packet_sha256']==sha(ROOT / 'steer135' / name / 'job.json')
        rows=doc['rows']
        assert len(rows)==135
        assert all(r['model']==models[name]['sha256'] and r['sensitivity_method']=='exact-gradient-d12'
                   and r['sensitivity_shared_owner_exact'] and r['engine_feature_disagreements']==0 for r in rows)
        assert all(r['pass']==(r['m2']>=.99 and r['m3f']>=.70) for r in rows)
        steering[name]=dict(passed=sum(r['pass'] for r in rows),cases=135,
            m2_passed=sum(r['m2']>=.99 for r in rows),m3f_passed=sum(r['m3f']>=.70 for r in rows),
            failing=[dict(key=r['key'],block=r['blocks'][0]['bounds'][2]-r['blocks'][0]['bounds'][0],m2=r['m2'],m3f=r['m3f'],
                          reasons=[k for k,b in [('M2',r['m2']<.99),('M3f',r['m3f']<.70)] if b])
                     for r in rows if not r['pass']])
    gates={}
    for family,names in ARMS.items():
        control=[steering[f'base-{family}-s{s}']['passed'] for s in range(3)]
        for arm in names:
            model=models[f'{arm}-s0']
            std=_verdict_numbers(ROOT / f'gates/verdict-s0-{arm}/standard')
            lad=_verdict_numbers(ROOT / f'gates/verdict-s0-{arm}/ladder')
            for grid in [std,lad]:
                assert all(c['state'] in ['pass','fail'] for c in grid['checks'].values())
            g={k:dict(pass_=all(v['checks'][k]['state']=='pass' for v in [std,lad]),
                      standard=std['checks'][k],ladder=lad['checks'][k]) for k in ['C1','C3','C4','C6']}
            g['C2']=dict(pass_=all(v['checks']['C2']['measured']<=.05 for v in [std,lad]),
                         standard=std['checks']['C2'],ladder=lad['checks']['C2'],bar=.05)
            index=candidate_shas.index(model['sha256'])
            g['C5']=dict(pass_=all(r['candidate_score_bits'][index]==hundred for r in proof),
                         rows=620,rule='exact f64 100.0 at every identity/tier row')
            g['G-DIAL']=dict(pass_=std['g_dial']['pass'] and lad['g_dial']['pass'],
                             standard=std['g_dial'],ladder=lad['g_dial'])
            near=read(ROOT / f'gates/nearid-s0/candidate-{arm}.GATES.json')
            for k in ['N1','N2','N3']:
                g[k]={**near[k],'pass_':near[k]['pass']}
            pack=packs[f'{arm}-s0']
            assert pack['model_sha256']==model['sha256'] and pack['K1_K2_K3'], 'canonical packed checks failed'
            for k in ['K1','K2','K3']:
                g[k]=dict(pass_=True,measurement=pack)
            assert k4[arm]['model_sha256']==model['sha256']
            g['K4']={**k4[arm],'pass_':k4[arm]['pass_']}
            g['K5']=dict(pass_=all(g[k]['pass_'] for k in ['C1','C3','C4','C6','G-DIAL']))
            g['G-STEER']=steer_rule([steering[f'{arm}-s{s}']['passed'] for s in range(3)],control)
            g.update(runtime(arm))
            assert set(g)==set(GATE_KEYS)
            gates[arm]=g
    result=verdict(stats,gates)
    assert result['state']=='COMPLETE'
    clarification = read(ROOT / 'mean-guard-clarification.json')
    assert clarification['state'] == 'resolved before any p-value read'
    assert clarification['owner_answer'] == 'Follow §4.2: mean is report-only'
    nonlinear_misses=all(steering[f'{arm}-s0']['passed']<135 for arm in ['w-a','k-a','w-c','k-c'])
    none_eligible=not any(a['eligible'] for f in result['families'].values() for a in f['arms'].values())
    l_test = stats['tests'][2]
    l_section42_fails = (not stats['holm_reject'][2] or not l_test['w2']['point_guard']
                        or any(not g['point_guard'] for g in l_test['dataset_guards']))
    conjunction=(none_eligible and nonlinear_misses and steering['l-a-s0']['passed']==135
                 and l_section42_fails)
    output=dict(schema='e34-results-v1',registration_commit='bc9be24a77a3bda0f710274d36ca03b3b94d5076',
                statistics=stats,reports=report,models=models,steering=steering,gates=gates,verdict=result,
                registered_falsification=dict(holds=conjunction,none_eligible=none_eligible,
                    w_k_both_families_seed0_miss=nonlinear_misses,l_a_seed0_pass=steering['l-a-s0']['passed']==135,
                    l_a_eligibility_holm_fails=not stats['holm_reject'][2],
                    l_a_eligibility_section42_fails=l_section42_fails),
                per_case=dict(path=str(ROOT / 'per-case.json'),sha256=sha(ROOT / 'per-case.json')))
    output['mean_guard_sensitivity'] = mean_guard_sensitivity(stats, gates)
    output['mean_guard_interpretation'] = clarification
    output['installed_integrity'] = dict(path=str(ROOT / 'installed-integrity-final.json'),
                                        sha256=sha(ROOT / 'installed-integrity-final.json'),
                                        cells=integrity['cells'], files=integrity['files'], unchanged=True)
    ci = read(ROOT / 'source-ci.json')
    assert ci['conclusion'] == 'success' and len(ci['jobs']) == 19
    assert all(job['conclusion'] == 'success' for job in ci['jobs'])
    output['source_ci'] = dict(path=str(ROOT / 'source-ci.json'),sha256=sha(ROOT / 'source-ci.json'),
                              commit=ci['headSha'],url=ci['url'],jobs=19,conclusion='success')
    runtime_host = read(ROOT / 'runtime-host.json')
    assert runtime_host['host'] == 'r7900x' and runtime_host['transfer_verified']
    assert runtime_host['payload_sha256']['evaluation/bin/costcmp-instrument'] == sha(ROOT / 'bin/costcmp-instrument')
    output['provenance'] = dict(statistics_binding=binding,
        runtime_host=runtime_host,
        report_binary_sha256=sha(ROOT / 'bin/panel-reports'),
        descriptive_execution=report_execution,
        runtime_build=read(ROOT / 'runtime-build.json'),
        p3_instrument=read(ROOT / 'bin/p3-instrument.artifact.json'),
        p3_build_sources=read(ROOT / 'p3-build-sources.json'),
        report_order_sha256=sha(ROOT / 'report-order.json'),
        mean_interpretation_sha256=sha(ROOT / 'mean-guard-clarification.json'),
        nearid_label_mapping=dict(path=str(ROOT / 'gates/nearid-labels.json'),
                                  sha256=sha(ROOT / 'gates/nearid-labels.json')),
        full_models_sha256=sha(ROOT / 'full-models.json'),
        capture_receipt=read(ROOT / 'capture-complete.json'),
        development_binding=read(ROOT.parent / 'packet/DEVELOPMENT_BINDING.json'),
        pilot_selection_sha256=sha(ROOT.parent / 'launch/LAMBDA_SELECTION.json'))
    output['penalty_logs'] = dict(path=str(ROOT / 'penalty-logs.json'),
                                  sha256=sha(ROOT / 'penalty-logs.json'))
    assert read(ROOT / 'penalty-logs.json')['cells'] == 90
    output['p3'] = dict(path=str(ROOT / 'p3-switches.json'),sha256=sha(ROOT / 'p3-switches.json'))
    log_root = Path.home() / 'tmp/e34-eval'
    resource_logs = ['statistics', 'steering', 'development', 'gates', 'p3', 'reports-r7900x']
    resource_logs += [f'r7900x-{arm}-{mode}' for arm in ['w-a','k-a','l-a','w-c','k-c']
                      for mode in ['rss', 'timing']]
    output['execution_resources'] = {}
    for name in resource_logs:
        path = log_root / f'{name}.log'
        summaries = [line for line in path.read_text().splitlines()
                     if line.startswith('run-heavy: done rc=0 ')]
        assert len(summaries) == 1, f'completed resource receipt required: {path}'
        output['execution_resources'][name] = dict(path=str(path),sha256=sha(path),summary=summaries[0])
    p3 = read(ROOT / 'p3-switches.json')
    assert len(p3['cases']) == 14 and p3['job_sha256'] == sha(ROOT / 'p3-job.json')
    output['linearity_interpretation'] = {s['name']: dict(
        geometric_ratio=s['interval']['estimate'],percentile_upper=s['interval']['hi'],
        student_upper=s['interval']['student_interval'][1],
        criterion=s['interval']['estimate']<=.70 and s['interval']['hi']<1
                  and s['interval']['student_interval'][1]<1)
        for s in report['series'] if s['name'].startswith('rel-l2-ratio/')}
    per_case = read(ROOT / 'per-case.json')
    output['l_coefficients'] = {}
    for seed in range(3):
        name = f'l-a-s{seed}'
        rows = [r for r in per_case if r['model']==name]
        assert len(rows)==135
        output['l_coefficients'][name] = dict(cases=len(rows),
            served_max_abs_s_plus_c=max(r['linear_probe']['max_abs_s_plus_c'] for r in rows),
            raw_max_abs_s_plus_c=max(r['raw_linear_probe']['max_abs_s_plus_c'] for r in rows),
            negative_input_cases=[dict(key=r['key'],block=r['block'],rows=r['linear_probe']['negative_input_rows'])
                                  for r in rows if r['linear_probe']['negative_input_rows']])
    write(ROOT / 'result.json',output)
    print(json.dumps(dict(verdict=result,falsification=output['registered_falsification']),indent=2))


def interval(iv, shift=0.):
    """Format the Rust owner's intervals; never calculate an interval here."""
    f = lambda x: format(x-shift, '.6g')
    student = iv['student_interval']
    assert student is not None, 'registered Student interval absent'
    return (f'{f(iv["estimate"])} [{f(iv["lo"])}, {f(iv["hi"])}]; '
            f'Student [{f(student[0])}, {f(student[1])}]')


def render():
    doc = read(ROOT / 'result.json')
    stats, reports = doc['statistics'], doc['reports']
    out = Path(__file__).resolve().parents[2] / 'benchmarks/e34_results_2026-10-15'
    lines = ['# E34 registered evaluation — 2026-10-15', '',
        'Frozen registration `bc9be24a`, revision 5. D13 margin **0.0065** in both families; '
        'D12 exact loaded-network gradient in the served map and G-STEER. '
        'Origins 1444 and 1458 remain excluded. No production model, profile or default changed.', '',
        'The owner directed runtime/RSS measurements to r7900x before collection, because dev '
        'had no planned quiet window. All arms and family controls use the same transferred '
        'binary and the unchanged quiet gate, paired-round collector and thresholds on r7900x. '
        'These runtime measurements apply to r7900x; dev/Zen5 runtime was not measured.', '',
        '[Registration](e34_registration_freeze_draft_2026-10-15.md); '
        '[launch bindings](e34_frozen_launch_2026-10-15.json); '
        '[machine-readable readout](e34_results_2026-10-15.json).', '']
    for family, row in doc['verdict']['families'].items():
        lines += [f'**Family {family.upper()}: {row["incumbent"]}.**', '']
        for arm, result in row['arms'].items():
            lines.append(f'- {arm}: '+('eligible' if result['eligible'] else 'ineligible — '+', '.join(result['failed'])))
        lines.append('')
    lines += ['Eligibility permits entry into full qualification; production adoption remains the owner’s decision. '
        'The sequential order is W → K → L. An eligible later arm replaces the current incumbent only with '
        'its registered Holm rejection and point improvement > +0.002.', '',
        '## Inferential readout', '',
        'All nine hypotheses were computed unconditionally by the preserved D13 `panel e34` owner. '
        'Two-way reference × seed bootstrap; B = 10,000; 95% Hyndman–Fan type-7 percentile intervals. '
        'The Student interval uses corrected SE and df 9 for pooled tests. Decision p is '
        'max(bootstrap one-sided p, Student one-sided p); Holm covers exactly nine tests at α = .05. '
        'Table intervals are on Δ (the margin is subtracted from the eligibility owner’s θ intervals).', '',
        '| # | Arm − control | Δ [percentile 95%]; Student 95% | p bootstrap | p Student | p decision | p Holm | Reject | Mean ≥ −0.002 (report-only) |',
        '|---|---|---|---:|---:|---:|---:|---|---|']
    compact_tests = []
    for i,t in enumerate(stats['tests']):
        lines.append(f'| {t["test"]} | {t["arm"]} − {t["control"]} | {interval(t["theta"], t["margin"])} '
            f'| {t["p"]["bootstrap"]:.6g} | {t["p"]["student"]:.6g} | {t["p"]["decision"]:.6g} '
            f'| {stats["holm_adjusted"][i]:.6g} | {stats["holm_reject"][i]} '
            f'| {t["e33_mean_point_guard"] if i < 5 else "—"} |')
        compact_tests.append({k:t[k] for k in ['test','arm','control','margin','state','theta','p',
            'seed','seed_unit','reference_only','reference_only_seed','dataset_guards','w2',
            'e33_mean_point_guard','improvement_point_guard']})
    lines += ['', 'The mean Δ ≥ −0.002 is the E33 point form reported beside E34’s formal §4.2 test. '
        'E34’s decisive point guards are each dataset Δ ≥ −0.005 and W2 Δ > −2 × its two-way SE. '
        'The owner confirmed this interpretation before any p-values were read; '
        'the preface’s “unchanged” refers to the rev-4 dataset/W2 guard set. '
        'The mean column retains the E33 point form without adding it as a decisive gate.', '',
        '| Arm | Dataset / W2 | Δ [percentile 95%]; Student 95% | Two-way SE | Guard |',
        '|---|---|---|---:|---|']
    for t in stats['tests'][:5]:
        for g in t['dataset_guards']:
            lines.append(f'| {t["arm"]} | {g["dataset"]} | {interval(g["interval"])} '
                         f'| {g["interval"]["se"]:.6g} | {g["point_guard"]} |')
        w=t['w2']
        lines.append(f'| {t["arm"]} | W2 | {interval(w["interval"])} | {w["interval"]["se"]:.6g} '
                     f'| {w["point_guard"]} |')
    sensitivity = doc['mean_guard_sensitivity']
    changed = ', '.join(sensitivity['changed_arm_eligibility']) or 'none'
    section42 = ', '.join(sensitivity['changed_section42_eligibility']) or 'none'
    family_changes = '; '.join(
        f'{f.upper()}: {r["registered"]} → {r["counterfactual"]}'
        for f,r in sensitivity['families'].items() if r['changed']) or 'none'
    lines += ['', '**NON-decisive sensitivity — mean Δ ≥ −0.002 decisive:** '
        f'arms changing §4.2 statistical eligibility: {section42}; '
        f'arms changing full qualification eligibility: {changed}; '
        f'families changing sequential verdict: {family_changes}. '
        'This counterfactual replays W → K → L with unchanged p-values, Holm decisions and other gates; '
        'it does not alter the registered verdict. Its interpretation and reporting rule were fixed '
        'before reading p-values.', '']
    lines += ['', 'Seed-unit and reference-only SEs, finite-draw counts, exact RNG seeds and the '
        'E33 point guards are retained in the JSON and complete evidence. No forced-choice or sealed labels were read.', '',
        '## Fixed steering and development panels', '',
        'Counts decide G-STEER: seed 0 requires 135/135; seeds 1 and 2 each require '
        'max(paired control count, 133). Every case requires M2 ≥ .99 and M3f ≥ .70. '
        'Rate intervals below are descriptive, clustered by source picture (15 for the fixed steering panel, '
        '8 for dev96); no interval substitutes for a fixed count. The same 135-case packet and hash-bound '
        'native repair captures were read once per served model. Raw no-spline forwards are separate registered '
        'curvature and floor diagnostics.', '',
        '| Series | Rate / ratio [percentile 95%]; Student 95% |', '|---|---|']
    for s in reports['series']:
        lines.append(f'| {s["name"]} | {interval(s["interval"])} |')
    lines += ['', '| Arm | Joint passes seeds 0 / 1 / 2 | Required | G-STEER |', '|---|---|---|---|']
    for family,names in ARMS.items():
        for arm in names:
            g=doc['gates'][arm]['G-STEER']
            lines.append(f'| {arm} | {g["counts"]} / 135 | {g["required"]} | {g["pass_"]} |')
    lines += ['', 'L’s coefficient comparison separates the raw linear head from the served spline’s '
        'chain-rule scaling. The table reports fixed-panel maxima, not population estimates.', '',
        '| L model | max abs(s+c), served | max abs(s+c), raw | Cases with negative input rows |',
        '|---|---:|---:|---:|']
    for name,row in doc['l_coefficients'].items():
        lines.append(f'| {name} | {row["served_max_abs_s_plus_c"]:.17g} '
                     f'| {row["raw_max_abs_s_plus_c"]:.17g} | {len(row["negative_input_cases"])} |')
    lines += ['', 'W/K relative-L2 ratios are geometric means of within-case arm/control ratios, '
        'with source-clustered percentile and Student intervals. No epsilon or dropped zero is allowed. '
        'The registered interpretation requires ratio ≤ .70 and both 95% upper bounds < 1. '
        'Per-case M2, M3f, relative L2, strict raw true > linear counts, L’s sensitivity/coefficient '
        'comparison and negative-input lists are in the hash-bound per-case artifact.', '',
        '## Registered engineering gates', '',
        'These are fixed engineering thresholds on registered panels, so they have measured values and '
        'point decisions rather than invented sampling intervals. Runtime alone uses its registered '
        'paired timing-round CI. Every gate is required even when another has failed.', '',
        '| Gate | W-A | K-A | L-A | W-C | K-C |', '|---|---|---|---|---|---|']
    arms=['w-a','k-a','l-a','w-c','k-c']
    for gate in GATE_KEYS:
        lines.append('| '+gate+' | '+' | '.join('PASS' if doc['gates'][a][gate]['pass_'] else 'FAIL' for a in arms)+' |')
    lines += ['', 'C1 mono ≥ .93; C2 ties ≤ .05; C3/C4 negative-tail probes and C6 upper bound use '
        'the existing bake-verdict owner on both standard and ladder grids. C5 is exactly 100.0 '
        'on all 620 identity/tier rows. G-DIAL requires p5 ≤ 25, p95 ≥ 85 and mono ≥ .93 on both grids. '
        'N1/N2 require ≥ 99.0 on all 24 references; N3 requires ≥ 122/144 ladders. '
        'K1–K3 use the canonical packed-spline validator (dense grid plus knot/ULP probes); '
        'K4 requires zero raw forwards at/below the exact stored lower floor over every inherited '
        'population, including calibration, both grids, negative/identity probes, NearID and all '
        'G-STEER base/repair forwards. K5 repeats C1/C3/C4/C6/G-DIAL.', '',
        '| Arm | Grid | C1 mono | C2 ties | C3 fraction negative | C4 deepest | C6 above-identity fraction | Dial p5 | Dial p95 | Dial mono |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        g=doc['gates'][arm]
        for grid in ['standard','ladder']:
            values=[g[c][grid]['measured'] for c in ['C1','C2','C3','C4','C6']]
            values += [g['G-DIAL'][grid][k] for k in ['p5','p95','mono']]
            lines.append(f'| {arm} | {grid} | '+' | '.join(format(v,'.9g') for v in values)+' |')
    lines += ['', '| Arm | N1 one-pixel score range | N2 highest-nonidentical range | N3 ladders / 144 | Exact identity rows / 620 |',
        '|---|---|---|---:|---:|']
    for arm in arms:
        g=doc['gates'][arm]
        lines.append(f'| {arm} | {g["N1"]["one_pixel_score_range"]} | {g["N2"]["highest_nonidentical_range"]} '
                     f'| {g["N3"]["monotone_ladders"]} | {620 if g["C5"]["pass_"] else "not exact"} |')
    lines += ['', '| Arm | Exact x_floor | Exact x0 | K2 probe points | K1–K3 |', '|---|---:|---:|---:|---|']
    for arm in arms:
        p=doc['gates'][arm]['K1']['measurement']
        lines.append(f'| {arm} | {p["x_floor"]:.17g} | {p["x0"]:.17g} | {p["K2_points"]} | {p["K1_K2_K3"]} |')
    lines += ['', '| Arm | K4 population | Rows | At/below floor | Below x0 | Min raw − x_floor |', '|---|---|---:|---:|---:|---:|']
    for arm in arms:
        for name,p in doc['gates'][arm]['K4']['populations'].items():
            lines.append(f'| {arm} | {name} | {p["rows"]} | {p["at_or_below_floor"]} | {p["below_x0"]} '
                         f'| {p["min_raw_minus_x_floor"]:.9g} |')
    lines += ['',
        'Runtime uses six v4x/v3 × 64²/256²/1024² cells at one thread, each with the first 32 '
        'clean paired rounds out of at most 64; the quiet gate and frozen analyzer are unchanged. '
        'A cell fails if its paired delta CI lies wholly above +2% of its family-control mean '
        'and is not resolution-limited. Spatial uses v4x 1024²/2048²: complete cached bin-1 '
        'score/map p95 ≤ 3× the same candidate’s uncached p95. Fresh-process incremental '
        'map peak RSS must be ≤ 128 B/pixel + 64 MiB; /usr/bin/time -v owns the peaks. '
        'Every measured grid value, per-cell CI, retained-round receipt and RSS observation '
        'is retained in the full result JSON.', '',
        '| Arm | Runtime cell | Change % | Paired delta ns [95% CI] | +2% bar ns | Pass |',
        '|---|---|---:|---|---:|---|']
    for arm in arms:
        for name,c in doc['gates'][arm]['runtime']['cells'].items():
            v=c['comparison']
            lines.append(f'| {arm} | {name} | {v["pct_change"]:.6g} | [{v["ci_lower"]:.9g}, {v["ci_upper"]:.9g}] '
                         f'| {c["plus2pct_ns"]:.9g} | {c["pass_"]} |')
    lines += ['', '| Arm | Spatial cell | Uncached p95 ns | Complete map p95 ns | Ratio | Pass ≤3 |',
        '|---|---|---:|---:|---:|---|']
    for arm in arms:
        for name,c in doc['gates'][arm]['spatial']['cells'].items():
            lines.append(f'| {arm} | {name} | {c["uncached_p95_ns"]} | {c["map_p95_ns"]} '
                         f'| {c["ratio"]:.9g} | {c["pass_"]} |')
    lines += ['', '| Arm | Geometry | Baseline peak KiB | Map peak KiB | Score peak KiB | Map increment KiB | Bar KiB | Pass |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for arm in arms:
        for name,c in doc['gates'][arm]['memory']['cells'].items():
            lines.append(f'| {arm} | {name} | {c["baseline_kib"]} | {c["map_peak_kib"]} | {c["score_peak_kib"]} '
                         f'| {c["map_incremental_kib"]} | {c["bar_kib"]} | {c["pass_"]} |')
    lines += ['',
        '## Report-only human agreement', '',
        'Pair classes use the pinned PairTable with OneWay(reference), fixed E21 orientation, '
        'target_tie = 0, no family labels, and seed 0–9 differences averaged inside each draw. '
        'Exact target ties are excluded and counted. Their seed-unit SEs are in the evidence; '
        'these comparisons do not decide eligibility.', '',
        '| Arm | Dataset | Pair class (1 ladder, 2 cross-codec, 3 cross-source) | Δ agreement [percentile 95%]; Student 95% |',
        '|---|---|---|---|']
    for t in stats['tests'][:5]:
        for row in t['pair_classes']:
            lines.append(f'| {t["arm"]} | {row["dataset"]} | {row["class"]} | {interval(row["interval"])} |')
    lines += ['', 'High-quality slice: unchanged E33 top-20% target threshold within each dataset, '
        'with all ties retained; all arms use the same mask. Intervals use the registered '
        'two-way SROCC and source-clustered pair-class owners, report-only.', '',
        '| Arm | Dataset | Δ SROCC [percentile 95%]; Student 95% |', '|---|---|---|']
    for h in reports['high_quality']:
        for row in h['srocc']:
            lines.append(f'| {h["arm"]} | {row["dataset"]} | {interval(row["interval"])} |')
    lines += ['', '| Arm | Dataset | HQ pair class | Δ agreement [percentile 95%]; Student 95% |', '|---|---|---|---|']
    for h in reports['high_quality']:
        for row in h['pair_classes']:
            lines.append(f'| {h["arm"]} | {row["dataset"]} | {row["class"]} | {interval(row["interval"])} |')
    full=ROOT / 'result.json'
    penalties=read(ROOT / 'penalty-logs.json')
    lines += ['', '## Execution resources', '',
        'Exact run-heavy summaries below describe the collectors, including their quiet waits '
        'and retries. Peak load is not a retained-round admission measurement; the unchanged '
        'quiet and clean-round receipts decide admission. Fresh-process scoring/map peaks '
        'come from /usr/bin/time -v as reported above.', '',
        '| Measurement | run-heavy resource receipt |', '|---|---|']
    for name, receipt in doc['execution_resources'].items():
        summary = receipt['summary'].replace('|', r'\|')
        lines.append(f'| {name} | `{summary}` |')
    lines += ['', '## Audit and limits', '',
        f'[Source CI]({doc["source_ci"]["url"]}) passed all {doc["source_ci"]["jobs"]} jobs '
        f'at `{doc["source_ci"]["commit"]}`. Later changes concern the Python result adapter '
        'and reporting; Rust sources were unchanged. Final adapter execution and local checks '
        'validate the generated readout.', '',
        f'All 305/305 installed cells across 18 jobsets were inventoried before the exposure entry '
        'was pushed (`02cba876`), before label access. The selected K pilot was reused as full-data '
        'seed 0; A chose λ = 1 and C chose λ = 10 from dev96, before this evaluation. '
        f'K’s report-only log covers {penalties["cells"]} fits and {penalties["epoch_records"]} epoch records. '
        'The trainer emits each record in stderr and again in its accumulated log; the reducer '
        'verified the two copies agree exactly and retains their multiplicity.', '',
        'P3 reports ReLU switches and dominant input contributions on the 14 E33 A/C '
        'seed-0 cases recorded as failures under the historical finite-difference sensitivity, '
        'using the existing f64 loaded-network reference and immutable '
        'feature captures. This diagnostic does not change the exact native G-STEER gate.', '',
        'The design is adaptive to the historical 135-case panel. Development is disjoint from '
        'that panel but remains TRAIN and descriptive. Historical §3.5 says 26 distinct development '
        'models; the executable 305-cell schedule contains 25 distinct full/pilot models '
        '(21 full plus four unchosen pilots). The operative §4.7 requires all 21 full models; '
        'all six pilot reads are retained. No extra fit was invented.', '',
        'The first primary statistical attempt was interrupted before any result file was written; '
        'no p-values were read from it. The complete nine-test calculation was recomputed from the same captured input '
        'with the same preserved D13 binary and draw seeds; only thread count/memory scheduling '
        'changed. Heavy jobs were serialized per host under run-heavy. '
        'Raw outputs and interrupted logs are preserved.', '',
        'After runtime collection, the same preserved report binary computed the independent '
        'descriptive intervals on r7900x. The paired-data file locator was relocated; all numerical '
        'inputs, model panels, thresholds, series order, helper pin and draw seeds were unchanged '
        'and checked for equality. The primary nine-test calculation remained on dev.', '',
        'The first NearID dispatch was refused by the canonical owner’s alphanumeric label '
        'grammar before any model/packet read or scored row. An injective artifact-label mapping '
        'then ran the same seven seed-0 models through the unchanged owner and gates; the '
        'failed log and mapping are retained.', '',
        'The archived augmentation encoder executable and decoded bytes remain pinned; its '
        'source closure is still unrecovered, as disclosed before launch. No new source-provenance '
        'claim is made. No sealed labels, production profiles or default bakes were changed.', '',
        f'Registered falsification conjunction: **{doc["registered_falsification"]["holds"]}**. '
        f'Components: no eligible arm = {doc["registered_falsification"]["none_eligible"]}; '
        f'W/K miss seed 0 in both families = {doc["registered_falsification"]["w_k_both_families_seed0_miss"]}; '
        f'L-A seed 0 passes = {doc["registered_falsification"]["l_a_seed0_pass"]}; '
        f'L-A fails formal §4.2 = {doc["registered_falsification"]["l_a_eligibility_section42_fails"]} '
        f'(Holm failure alone = {doc["registered_falsification"]["l_a_eligibility_holm_fails"]}). '
        'This conjunction concerns the tested W/K/L training choices and does not prove that '
        'every model in the class must fail.', '',
        f'Full machine-readable evidence on dev: `{full}`; SHA-256 `{sha(full)}`. '
        'Large per-case, penalty and timing evidence remains outside git with these hashes. '
        'The adjacent JSON contains the decisive statistical and gate summary plus exact artifact bindings.', '']
    small=dict(schema='e34-published-results-v1',registration_commit=doc['registration_commit'],
        verdict=doc['verdict'], registered_falsification=doc['registered_falsification'],
        mean_guard_sensitivity=doc['mean_guard_sensitivity'],
        mean_guard_interpretation=doc['mean_guard_interpretation'],
        statistics=dict(eligibility_margin=.0065,resamples=10000,level=.95,percentile=stats['percentile'],
            tests=compact_tests,holm_adjusted=stats['holm_adjusted'],holm_reject=stats['holm_reject']),
        steering={a:{k:v for k,v in r.items() if k!='failing'} for a,r in doc['steering'].items()},
        gates={a:{g:v['pass_'] for g,v in gates.items()} for a,gates in doc['gates'].items()},
        linearity_interpretation=doc['linearity_interpretation'],
        l_coefficients=doc['l_coefficients'],
        evidence=dict(path=str(full),sha256=sha(full),per_case=doc['per_case'],penalty_logs=doc['penalty_logs'],
                      p3=doc['p3'],installed_integrity=doc['installed_integrity'],source_ci=doc['source_ci']),
        provenance=doc['provenance'])
    # Keep raw per-seed values and operator receipts in the bound full evidence.
    # All nine primary tests and all five decisive guard sets remain here.
    small = copy.deepcopy(small)
    for test in small['statistics']['tests']:
        test['seed_unit'].pop('seed_deltas')
        if test['test'] > 5:
            test.pop('dataset_guards')
            test.pop('w2')
        else:
            test['w2']['seed_unit'].pop('seed_deltas')
    small['statistics']['summary_omissions_in_full_evidence'] = [
        'per-seed deltas', 'non-decisive improvement dataset/W2 detail']
    small['mean_guard_sensitivity'].pop('counterfactual_verdict')
    reference = lambda name: dict(path=str(ROOT / name),sha256=sha(ROOT / name))
    provenance = small['provenance']
    host = provenance['runtime_host']
    provenance['runtime_host'] = dict(**reference('runtime-host.json'),host=host['host'],
        cpu_model=host['machine']['cpu_model'],
        benchmark_sha256=host['payload_sha256']['evaluation/bin/costcmp-instrument'],
        analyzer_sha256=host['payload_sha256']['evaluation/bin/paired-rounds-analyzer'])
    provenance['runtime_build'] = reference('runtime-build.json')
    provenance['descriptive_execution'].pop('payload_sha256')
    development = ROOT.parent / 'packet/DEVELOPMENT_BINDING.json'
    provenance['development_binding'] = dict(path=str(development),sha256=sha(development))
    payload=json.dumps(small,separators=(',',':'),allow_nan=False)+'\n'
    assert len(payload.encode()) <= 30000, 'publish a smaller summary; keep full generated data outside git'

    complete = '\n'.join(lines)
    report_start = complete.index('\n## Report-only human agreement')
    audit_start = complete.index('\n## Audit and limits')
    appendix = ('# E34 report-only appendix — 2026-10-15\n\n'
        '[Registered verdict and methods](e34_results_2026-10-15.md). '
        'These human-agreement tables are descriptive; execution receipts document the collectors.\n'
        + complete[report_start:audit_start].rstrip() + '\n')
    main = (complete[:report_start] + '\n\n'
        '[Report-only human agreement and execution resources](e34_report_only_2026-10-15.md) '
        'contains the remaining pair-class/HQ intervals and exact collector resource receipts.\n'
        + complete[audit_start:])
    assert len(main.encode()) <= 30000 and len(appendix.encode()) <= 30000
    appendix_path = out.parent / 'e34_report_only_2026-10-15.md'
    full_readout = ROOT / 'full-readout.md'
    # Resolve the original benchmark-relative hyperlinks at the copied location.
    for name in ['e34_registration_freeze_draft_2026-10-15.md',
                 'e34_frozen_launch_2026-10-15.json', 'e34_results_2026-10-15.json']:
        complete = complete.replace(f']({name})',
            f'](https://github.com/imazen/zensimx/blob/main/benchmarks/{name})')
    assert all(not path.exists() for path in [out.with_suffix('.json'),out.with_suffix('.md'),
                                             appendix_path,full_readout]), 'fresh publication paths required'
    with full_readout.open('x') as f:
        f.write(complete)
    with out.with_suffix('.json').open('x') as f:
        f.write(payload)
    with out.with_suffix('.md').open('x') as f:
        f.write(main)
    with appendix_path.open('x') as f:
        f.write(appendix)
    print(json.dumps(dict(markdown=str(out.with_suffix('.md')),json_bytes=len(payload.encode()),
                         markdown_bytes=len(main.encode()),appendix_bytes=len(appendix.encode()),
                         full_readout=str(full_readout))))


def verify_publication():
    """Compare the public serialization with immutable, complete measurements."""
    full = read(ROOT / 'result.json')
    base = Path(__file__).resolve().parents[2] / 'benchmarks'
    paths = [base / 'e34_results_2026-10-15.json', base / 'e34_results_2026-10-15.md',
             base / 'e34_report_only_2026-10-15.md']
    public = read(paths[0])
    assert all(path.stat().st_size <= 30000 for path in paths)
    assert public['evidence']['sha256'] == sha(ROOT / 'result.json')
    for key in ['verdict', 'registered_falsification', 'mean_guard_interpretation',
                'linearity_interpretation', 'l_coefficients']:
        assert public[key] == full[key], f'published {key} changed'
    expected = {k:v for k,v in full['mean_guard_sensitivity'].items() if k!='counterfactual_verdict'}
    assert public['mean_guard_sensitivity'] == expected
    assert public['gates'] == {a:{g:v['pass_'] for g,v in gates.items()}
                               for a,gates in full['gates'].items()}
    assert public['steering'] == {a:{k:v for k,v in row.items() if k!='failing'}
                                  for a,row in full['steering'].items()}
    statistical = public['statistics']
    original = full['statistics']
    assert statistical['holm_adjusted'] == original['holm_adjusted']
    assert statistical['holm_reject'] == original['holm_reject']
    assert len(statistical['tests']) == len(original['tests']) == 9
    for published, measured in zip(statistical['tests'], original['tests'], strict=True):
        for key in ['test','arm','control','margin','state','theta','p','seed','reference_only',
                    'reference_only_seed','e33_mean_point_guard','improvement_point_guard']:
            assert published[key] == measured[key], f'published test {measured["test"]} {key} changed'
        assert published['seed_unit'] == {k:v for k,v in measured['seed_unit'].items() if k!='seed_deltas'}
        if measured['test'] <= 5:
            assert published['dataset_guards'] == measured['dataset_guards']
            w2 = copy.deepcopy(measured['w2'])
            w2['seed_unit'].pop('seed_deltas')
            assert published['w2'] == w2
        else:
            assert 'dataset_guards' not in published and 'w2' not in published
    table_rows = lambda text: Counter(line for line in text.splitlines() if line.startswith('|'))
    complete_rows = table_rows((ROOT / 'full-readout.md').read_text())
    published_rows = table_rows(paths[1].read_text()) + table_rows(paths[2].read_text())
    assert published_rows == complete_rows, 'a report table row changed or was omitted'
    validate_statistical_reports(original)
    expected = dict(schema='e34-publication-checks-v1',all_nine_tests_unchanged=True,
        all_five_decisive_guard_sets_unchanged=True,all_verdicts_and_gates_unchanged=True,
        all_table_rows_unchanged=True,table_rows=sum(complete_rows.values()),
        full_result_sha256=sha(ROOT / 'result.json'),full_readout_sha256=sha(ROOT / 'full-readout.md'),
        publications={path.name:dict(bytes=path.stat().st_size,sha256=sha(path)) for path in paths})
    receipt = ROOT / 'PUBLICATION_VERIFIED.json'
    if receipt.exists():
        previous = read(receipt)
        previous.pop('time')
        assert previous == expected, 'verified publication changed'
    else:
        write(receipt,dict(time=datetime.now(timezone.utc).isoformat(),**expected))
    print(json.dumps(expected,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['build','penalties','render','verify-publication'])
    args=p.parse_args()
    {'build':build,'penalties':penalties,'render':render,'verify-publication':verify_publication}[args.mode]()
