#!/usr/bin/env python3
"""Verify and preserve the explicitly admitted frozen-model adjudication."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess

ROOT = Path('/mnt/v/output/zensim/adjudicate-2026-10-07')
PRIOR = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')
MIRROR = Path('/mnt/tower/output/zensim/adjudicate-2026-10-07')
SCRATCH = Path('/home/lilith/tmp/adjudicate')
BASE = 'dcf7e3552b06e1bc0b36520f945d77d62d60295b'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify():
    report = {'scope': 'frozen seed0; engineering inputs only; no shipping verdict'}
    identities = []
    reference_ids = [422, 480, 509, 538, 567, 596, 625, 654, 683, 712]
    for model in read(ROOT / 'identity.json')['models']:
        rows = model['rows']
        assert len(rows) == 38 and len(model['consumed_ids']) == 420
        for row in rows:
            assert row['f32_roundtrip_mismatch_ids'] == []
            assert row['f64_mismatch_ids'] == reference_ids
            assert row['served_vs_canonical_mismatch_ids'] == []
            assert [item[0] for item in row['nonzero_consumed']] == reference_ids
            assert len(row['consumed_native_bits']) == 420
            assert row['cached_score'] == row['native_feature_score'] == row['native_f32_feature_score']
            assert row['pixel_score'] == row['identity_aware_score'] == 100.0
        scores = [r['cached_score'] for r in rows]
        identities.append({'model': Path(model['path']).name, 'rows': len(rows),
                           'consumed_bits_verified': len(rows) * 420,
                           'expected_f64_storage_rounding': len(rows) * 10,
                           'minimum': min(scores), 'maximum': max(scores),
                           'in_band': sum(97.5 <= x <= 100 for x in scores)})
    report['identity'] = identities
    b = read(ROOT / 'identity-B.json')['models'][0]
    assert len(b['rows']) == 38 and b['revision'] == '1'
    assert all(r['pixel_score'] == r['identity_aware_score'] == 100 for r in b['rows'])
    scores = [r['native_feature_score'] for r in b['rows']]
    placeholders = {r['served_feature_score'] for r in b['rows']}
    assert len(placeholders) == 1
    report['B_identity'] = {'rows': 38, 'revision': 1, 'minimum': min(scores),
                            'maximum': max(scores), 'in_band': sum(97.5 <= x <= 100 for x in scores),
                            'shortcut_placeholder_score': placeholders.pop(),
                            'canonical_vs_shortcut_mismatches': sum(len(r['served_vs_canonical_mismatch_ids']) for r in b['rows'])}
    assert read(ROOT / 'identity-negative-control.json')['models'][0]['rows'][0]['f32_roundtrip_mismatch_ids'] == [422]
    report['negative_control'] = 'f422 low-bit change detected'
    grids = {}
    for name in ('standard', 'ladder'):
        original = ROOT / name
        corrected = ROOT / 'corrected' / name
        assert (original / 'predictions.tsv').read_bytes() == (corrected / 'predictions.tsv').read_bytes()
        old = {c['id']: c for c in read(original / 'gaddr.json')['checks']}
        new = {c['id']: c for c in read(corrected / 'gaddr.json')['checks']}
        for key in old:
            if key != 'C2':
                assert old[key] == new[key], (name, key)
        mirror = read(ROOT / 'TIES.json')[name][0]
        assert new['C2']['measured'] == mirror['missing_aware_ties']
        assert new['C2']['bar'] == .05 and new['C2']['state'] == 'pass'
        findings = [r for r in read(ROOT / f'{name}-tied-cells.json') if r['model'] == 'seed0.bin']
        reclassified = [r for r in findings if r['missing_aware_bucket'] != r['original_bucket']]
        remaining = [r for r in findings if r['missing_aware_bucket'] == 'flat']
        assert all(r['consumed_bits_equal'] and r['finite_feature_max_abs_delta'] == 0 for r in reclassified)
        packets = read(ROOT / 'packets' / f'{name}.json')
        ids = read(ROOT / f'{name}-native.json')['models'][0]['consumed_ids']
        for rung in reclassified:
            a = packets[rung['row0']]['feature_bits']
            b = packets[rung['row1']]['feature_bits']
            assert all(a[i] == b[i] for i in ids)
        assert all(r['at_lower_floor'] and r['uncalibrated0'] != r['uncalibrated1'] for r in remaining)
        grids[name] = {'prediction_tsv_sha256': sha(original / 'predictions.tsv'),
                       'all_non_C2_checks_unchanged': True, 'exact_equal_reclassified': len(reclassified),
                       'remaining_floor_ties': len(remaining), 'varying_network_outputs': len(remaining),
                       'original_C2': old['C2'], 'corrected_C2': new['C2']}
    report['grids'] = grids
    steering = read(ROOT / 'STEERING.json')
    packed = [r for r in steering if r['model'] == 'seed0']
    dense = [r for r in steering if r['model'] == 'dense']
    assert len(packed) == len(dense) == 7
    assert all(r['original_numerical_parity'] and not r['pass'] for r in packed)
    jpeg = [r for r in packed if r['instrument'] == 'jpeg8']
    assert len(jpeg) == 4 and all(r['all_deltas_zero'] for r in jpeg)
    report['steering'] = {'packed_exact_original_failures': 7,
                          'dense_M2_failures': sum(r['m2'] < .99 for r in dense),
                          'dense_M3f_failures': sum(r['m3f'] < .70 for r in dense),
                          'dense_cases_passing_both': sum(r['pass'] for r in dense)}
    probes = read(ROOT / 'jpeg-floor-probes.json')['models'][0]
    floor = probes['output_spline']['lower_floor']
    assert len(probes['rows']) == 16
    assert all(r['pixel_score'] == r['native_feature_score'] == floor for r in probes['rows'])
    groups = {}
    for row in probes['rows']:
        groups.setdefault(row['key'].split(':')[0], []).append(row['native_uncalibrated_score'])
    assert len(groups) == 4 and all(len(set(values)) > 1 for values in groups.values())
    report['floor_probes'] = {'four_pairs': groups, 'complete_score': floor}
    coverage = read(ROOT / 'OWNER_SEEDS.json')
    assert len(coverage) == 12
    for seed in ('seed0', 'seed1', 'seed2'):
        rows = [r for r in coverage if r['model'] == seed]
        assert len(rows) == 4 and all(r['pass'] for r in rows)
    assert all(r['original_numerical_parity'] for r in coverage if r['model'] == 'seed0')
    report['owner_coverage'] = {'production_seed0': '4/4 pass, exact original parity',
                               'independent_D1_seed_controls': '12/12 pass across seeds0,1,2',
                               'minimum_M2': min(r['m2'] for r in coverage),
                               'minimum_M3f': min(r['m3f'] for r in coverage),
                               'shipping_composition': 'seed0 alone; control seeds do not expand it'}
    for pin in read(ROOT / 'MODEL_PREREAD.json').values():
        assert sha(Path(pin['path'])) == pin['sha256']
    assert sha(ROOT / 'models/dense.bin') == 'e6995b1073de6022c31e012a104a50a1710488d4e520ee7fd016c649d924c4c4'
    assert Path('zensim-validate/src/bin/bake_verdict.rs').read_bytes() == (ROOT / 'bake_verdict.base.rs').read_bytes()
    report['canonical_owner_restored_exactly'] = True
    save(ROOT / 'FINAL_VERIFICATION.json', report)
    print('PASS: identity bits/scores, exact prediction parity, C2-only change, seven failures, floor probes, 12 owner controls')


def snapshot(commit):
    """Copy named engineering inputs and source; never enumerate nearby data."""
    evidence = ROOT / 'replay'
    evidence.mkdir(exist_ok=True)
    source = ROOT / 'source'
    source.mkdir(exist_ok=True)
    files = ['scripts/adjudicate_evidence.py', 'scripts/prodqual_label_free.py',
             'benchmarks/adjudicate.just', 'benchmarks/adjudicate_registration_2026-10-07.md',
             'benchmarks/adjudicate_2026-10-07.md', 'benchmarks/adjudicate_bake_verdict.patch',
             'benchmarks/adjudicate_known_bugs.patch', 'zensim/examples/serve_custom_bake.rs',
             'zensim/examples/serve_custom_bake/adjudicate.rs',
             'zensim-validate/src/bin/bake_verdict/missing_features.rs', 'Cargo.toml', 'Cargo.lock']
    for name in files:
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, dest)
    for name in ('diffmap_block_coherence', 'bake_verdict'):
        shutil.copy2(PRIOR / 'bin' / name, ROOT / 'bin' / f'{name}.original')
    for name in ('STEERING_PREREAD.json', 'STEERING_RESULTS.json',
                 'STEERING_PREREAD_B.json', 'STEERING_RESULTS_B.json',
                 'prodqual_b_registration_2026-10-07.json'):
        shutil.copyfile(PRIOR / name, evidence / name)
    refusals = read(PRIOR / 'STEERING_RESULTS_B.json')
    assert len(refusals) == 135 and all(r['status'] == 'blocked' for r in refusals)
    for row in refusals:
        path = Path(row['report']).with_suffix('.log')
        assert 'steering session refuses masked/IW' in path.read_text()
        shutil.copyfile(path, evidence / f"B-refusal-{row['instrument']}-{row['row']}.log")
    registration = read(PRIOR / 'STEERING_PREREAD.json')
    results = read(PRIOR / 'STEERING_RESULTS.json')
    records = []
    for row in results:
        if row['status'] == 'PASS' and row['instrument'] != 'owner':
            continue
        case = next(c for c in registration['cases'] if c['instrument'] == row['instrument'] and c['row_id'] == row['row'])
        old = Path(row['report'])
        assert sha(old) == row['sha256']
        dest = evidence / f"original-{row['instrument']}-{row['row']}.json"
        shutil.copyfile(old, dest)
        for kind in ('reference', 'distorted'):
            path = Path(case['ref_path'] if kind == 'reference' else case['dist_path'])
            digest = sha(path)
            assert digest == case[f'{kind}_file_sha256']
            dest = evidence / 'pixels' / (digest + path.suffix)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, dest)
            records.append({'original': str(path), 'archived': str(dest.relative_to(ROOT)), 'sha256': digest})
    for case in read(ROOT / 'packets/identity.json'):
        path = Path(case['ref_path'])
        digest = sha(path)
        dest = evidence / 'pixels' / (digest + path.suffix)
        shutil.copyfile(path, dest)
        records.append({'original': str(path), 'archived': str(dest.relative_to(ROOT)), 'sha256': digest})
    for pin in read(ROOT / 'PACKET_PREREAD.json'):
        for key in ('path', 'declaration'):
            path = Path(pin[key])
            digest = sha(path)
            assert digest == pin['sha256' if key == 'path' else 'declaration_sha256']
            dest = evidence / f"{pin['name']}-{path.name}"
            shutil.copyfile(path, dest)
            records.append({'original': str(path), 'archived': str(dest.relative_to(ROOT)), 'sha256': digest})
    save(evidence / 'INPUT_PATH_MAP.json', records)
    release = Path('/home/lilith/tmp/zensim-paper/rev4/RELEASE_after_speedq.md')
    speedq = release.with_name('SPEEDQ_DONE.md')
    save(ROOT / 'FINAL_PROVENANCE.json', {
        'build_commit': commit, 'base_commit': BASE,
        'review_only': True, 'no_push': True, 'production_seed0_unchanged': True,
        'human_payload_reads': [], 'calibration_payload_reads': [],
        'release_instruction_sha256': sha(release), 'speedq_completion_sha256': sha(speedq),
        'heavy_prefix': 'flock ~/tmp/zensim-paper/rev4/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 --',
        'models': {p.name: sha(p) for p in sorted((ROOT / 'models').glob('*.bin'))},
        'programs': {p.name: sha(p) for p in sorted((ROOT / 'bin').iterdir()) if p.is_file()},
        'program_note': 'Historical helper versions are retained with their original reports; final helper source is in source/. Owner overlay is base source plus the archived patch.',
        'coverage_receipt': (ROOT / 'coverage.log').read_text(),
        'verification': 'FINAL_VERIFICATION.json; native owner source unchanged outside the reviewed overlay'})
    print('Copied explicit engineering inputs, pixel bytes and source; no adjacent dataset enumeration')


def mirror(commit):
    assert read(ROOT / 'FINAL_PROVENANCE.json')['build_commit'] == commit
    mounts = subprocess.check_output(['findmnt', '-T', str(MIRROR.parent), '-o', 'TARGET,SOURCE,FSTYPE'], text=True)
    assert 'tower:' in mounts and 'nfs' in mounts, mounts
    excluded = {'_MANIFEST.json', 'MIRROR_VERIFICATION.json', 'CARGO_CLEANUP.json'}
    inventory = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in excluded]
    save(ROOT / '_MANIFEST.json', {'build_commit': commit, 'files': inventory,
                                  'excluded_receipts': sorted(excluded)})
    MIRROR.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--no-owner', '--no-group', str(ROOT) + '/', str(MIRROR) + '/'], check=True)
    for item in inventory:
        assert sha(MIRROR / item['path']) == item['sha256'], item['path']
    assert sha(MIRROR / '_MANIFEST.json') == sha(ROOT / '_MANIFEST.json')
    samples = random.Random(20261008).sample(inventory, 3)
    receipt = {'build_commit': commit, 'mount_observation': mounts,
               'resolving_command': 'findmnt -T /mnt/tower/output/zensim -o TARGET,SOURCE,FSTYPE',
               'files_verified': len(inventory), 'bytes_verified': sum(i['bytes'] for i in inventory),
               'manifest_sha256': sha(ROOT / '_MANIFEST.json'), 'three_random_files': samples,
               'local': str(ROOT), 'mirror': str(MIRROR), 'deletions': False}
    save(ROOT / 'MIRROR_VERIFICATION.json', receipt)
    shutil.copyfile(ROOT / 'MIRROR_VERIFICATION.json', MIRROR / 'MIRROR_VERIFICATION.json')
    assert sha(ROOT / 'MIRROR_VERIFICATION.json') == sha(MIRROR / 'MIRROR_VERIFICATION.json')
    print(f"PASS: all {len(inventory)} files SHA256 verified; three random files recorded")


def cleanup():
    target = SCRATCH / 'target'
    assert target.is_dir() and (target / '.rustc_info.json').is_file()
    assert read(ROOT / 'MIRROR_VERIFICATION.json')['files_verified'] > 0
    # A holder of this lane's serialized command cannot overlap another heavy job.
    # This unique target was only used by the adjudication lane's completed builds.
    for name in ('serve_custom_bake', 'bake_dial_refit', 'bake_verdict'):
        assert (ROOT / 'bin' / name).is_file()
        assert sha(ROOT / 'bin' / name) == sha(MIRROR / 'bin' / name)
    size = int(subprocess.check_output(['du', '-s', '-B1', str(target)], text=True).split()[0])
    shutil.rmtree(target)
    receipt = {'removed': str(target), 'allocated_bytes_before': size,
               'retained_programs_sha256_verified_on_tower': True,
               'other_targets_touched': False, 'other_generated_artifacts_deleted': False}
    save(ROOT / 'CARGO_CLEANUP.json', receipt)
    shutil.copyfile(ROOT / 'CARGO_CLEANUP.json', MIRROR / 'CARGO_CLEANUP.json')
    assert sha(ROOT / 'CARGO_CLEANUP.json') == sha(MIRROR / 'CARGO_CLEANUP.json')
    print(f'Removed own inactive Cargo target: {size} allocated bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['verify', 'snapshot', 'mirror', 'cleanup'])
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.mode == 'verify':
        verify()
    elif args.mode == 'cleanup':
        cleanup()
    else:
        assert args.commit, 'exact committed review source required'
        (snapshot if args.mode == 'snapshot' else mirror)(args.commit)
