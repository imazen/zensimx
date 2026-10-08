#!/usr/bin/env python3
"""Summarize serve_custom_bake --prodqual evidence, without reading labels."""
import argparse
import json
from pathlib import Path
import sys


def adjudicate_prepare(output):
    """Project only three admitted engineering packets, never opinion labels."""
    import hashlib
    import numpy as np
    import pyarrow.parquet as pq
    prior = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')
    registration = json.loads((prior / 'prodqual_b_registration_2026-10-07.json').read_text())
    inputs = registration['inputs'] if 'inputs' in registration else registration['instruments']
    if isinstance(inputs, dict):
        inputs = list(inputs.values())
    references = json.loads((prior / 'pixel-identity/PREREAD.json').read_text())['rows']
    pins = []
    for name in ('identity', 'standard', 'ladder'):
        item = next(x for x in inputs if x['name'] == name)
        path = Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        declaration = Path(item['declaration'])
        assert hashlib.sha256(declaration.read_bytes()).hexdigest() == item['declaration_sha256']
        fields = [f'f{i}' for i in range(720)]
        metadata = ['entry'] if name == 'identity' else ['image_id', 'codec', 'q']
        table = pq.read_table(path, columns=fields + metadata)
        values = np.column_stack([table[f].to_numpy() for f in fields]).astype('<f8')
        assert len(values) == item['rows']
        cases = []
        keys = []
        for i, bits in enumerate(values.view('<u8')):
            case = {'key': str(i), 'feature_bits': bits.tolist()}
            if name == 'identity':
                reference = references[i]
                assert table['entry'][i].as_py() == reference['entry']
                assert reference['ref_path'] == reference['dist_path']
                source = Path(reference['ref_path'])
                assert hashlib.sha256(source.read_bytes()).hexdigest() == reference['reference_file_sha256']
                case['ref_path'] = str(source)
                keys.append({'row': i, 'entry': reference['entry']})
            else:
                keys.append({'row': i, **{k: table[k][i].as_py() for k in metadata}})
            cases.append(case)
        (output / 'packets' / f'{name}.json').write_text(json.dumps(cases, separators=(',', ':')))
        (output / 'packets' / f'{name}.keys.json').write_text(json.dumps(keys, indent=2))
        if name == 'identity':
            b_cases = [{'key': x['key'], 'ref_path': x['ref_path']} for x in cases]
            (output / 'packets' / 'identity-B.json').write_text(json.dumps(b_cases))
        pins.append({**item, 'feature_dtype': str(table['f13'].type),
                     'projected_payload': str(output / 'packets' / f'{name}.json')})
        print(f'prepared {name}: {len(values)} label-free rows', flush=True)
    (output / 'PACKET_PREREAD.json').write_text(json.dumps(pins, indent=2))


def adjudicate_verdict(output, corrected=False):
    """Replay existing, corpus-disabled owner commands with per-cell dumps."""
    import os
    import subprocess
    import shutil
    prior = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')
    for name in ('standard', 'ladder'):
        folder = output / 'corrected' / name if corrected else output / name
        folder.mkdir(parents=True, exist_ok=True)
        (output / 'no-human-corpus').mkdir(exist_ok=True)
        shutil.copyfile(prior / 'no-human-corpus/_MANIFEST.json',
                        output / 'no-human-corpus/_MANIFEST.json')
        command = json.loads((prior / name / 'COMMAND.json').read_text())
        command = [s.replace(str(prior), str(output)) for s in command]
        command[0] = str((output if corrected else prior) / 'bin/bake_verdict')
        for flag in ('--gaddr-json', '--fulleval', '--output'):
            index = command.index(flag) + 1
            command[index] = str(folder / Path(command[index]).name)
        environment = dict(os.environ, ZENSIM_FORMULA_REV='5', RAYON_NUM_THREADS='1',
                           ZENSIM_DIAL_PRED_OUT=str(folder / 'predictions.tsv'))
        (folder / 'COMMAND.json').write_text(json.dumps(command, indent=2))
        with (folder / 'verdict.log').open('w') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, (name, result.returncode)
        print(f'owner replay {name} completed', flush=True)


def adjudicate_ties(output):
    """Independent classification mirror, bound to the native owner scores."""
    import csv
    import collections
    import numpy as np
    import pyarrow.parquet as pq
    summary = {}
    for name in ('standard', 'ladder'):
        keys = json.loads((output / 'packets' / f'{name}.keys.json').read_text())
        data = json.loads((output / f'{name}-native.json').read_text())
        native = data['models']
        features = pq.read_table(f'/var/tmp/shippath7/verified/instruments/{name}.parquet',
                                 columns=[f'f{i}' for i in range(720)])
        values = np.column_stack([features[f'f{i}'].to_numpy() for i in range(720)]).astype('f8')
        with (output / name / 'predictions.tsv').open() as stream:
            owner = list(csv.DictReader(stream, delimiter='\t'))
        assert len(owner) == len(keys)
        scores = [r['cached_score'] for r in native[0]['rows']]
        assert all(float(row['pred']) == score for row, score in zip(owner, scores, strict=True))
        assert all(row['image_id'] == key['image_id'] and row['codec'] == key['codec']
                   and float(row['q']) == key['q'] for row, key in zip(owner, keys, strict=True))
        curves = collections.defaultdict(list)
        for key in keys:
            curves[(key['image_id'], key['codec'])].append(key)
        findings = []
        model_summaries = []
        for model in native:
            score = [r['cached_score'] for r in model['rows']]
            uncal = [r['uncalibrated_score'] for r in model['rows']]
            counts = collections.Counter()
            by_codec = collections.defaultdict(collections.Counter)
            floor = model['output_spline']['lower_floor'] if model['output_spline'] else None
            for (image, codec), curve in sorted(curves.items()):
                curve.sort(key=lambda x: x['q'])
                for a, b in zip(curve, curve[1:]):
                    i, j = a['row'], b['row']
                    delta = score[j] - score[i]
                    x, y = values[i], values[j]
                    finite = np.isfinite(x) & np.isfinite(y)
                    assert np.all(np.isfinite(x[model['consumed_ids']]))
                    assert np.all(np.isfinite(y[model['consumed_ids']]))
                    original_eq = bool(np.all(np.abs(x-y) <= 1e-5))
                    corrected_eq = bool(np.all(((np.isnan(x) & np.isnan(y)) |
                                               (finite & (np.abs(x-y) <= 1e-5)))))
                    # Exact transcription of the owner's five buckets; bars unchanged.
                    def bucket(equal):
                        return ('forward' if delta > .5 else 'inversion' if delta < -.5
                                else 'codec_saturated' if equal else 'flat' if abs(delta) <= 1e-9
                                else 'subresolution')
                    old, corrected = bucket(original_eq), bucket(corrected_eq)
                    counts['pairs'] += 1
                    counts[f'original_{old}'] += 1
                    counts[f'corrected_{corrected}'] += 1
                    by_codec[codec]['pairs'] += 1
                    by_codec[codec][f'original_{old}'] += 1
                    by_codec[codec][f'corrected_{corrected}'] += 1
                    if old == 'flat' or corrected != old:
                        findings.append({'model':Path(model['path']).name,'image_id':image,'codec':codec,
                            'q0':a['q'],'q1':b['q'],'row0':i,'row1':j,'score0':score[i],'score1':score[j],
                            'uncalibrated0':uncal[i],'uncalibrated1':uncal[j],
                            'at_lower_floor':score[i] == floor and score[j] == floor,
                            'original_bucket':old,'missing_aware_bucket':corrected,
                            'consumed_bits_equal':bool(np.array_equal(x[model['consumed_ids']],y[model['consumed_ids']])),
                            'finite_feature_max_abs_delta':float(np.max(np.abs(x[finite]-y[finite]))),
                            'matching_nan_slots':int(np.sum(np.isnan(x)&np.isnan(y)))})
            model_summaries.append({'model':Path(model['path']).name,'counts':dict(counts),
                'original_ties':counts['original_flat']/counts['pairs'],
                'missing_aware_ties':counts['corrected_flat']/counts['pairs'],
                'by_codec':{k:dict(v) for k,v in by_codec.items()},
                'floor_cells':sum(s == floor for s in score),'lower_floor':floor})
        checks = json.loads((output / name / 'gaddr.json').read_text())['checks']
        c2 = next(c for c in checks if c['id']=='C2')
        assert c2['measured'] == model_summaries[0]['original_ties']
        summary[name] = model_summaries
        (output / f'{name}-tied-cells.json').write_text(json.dumps(findings,indent=2))
    (output / 'TIES.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def adjudicate_steering(output, coverage=False):
    """Bounded replay of seven registered failures through the existing map owner."""
    import hashlib
    import os
    import subprocess
    prior = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')
    registration = json.loads((prior / 'STEERING_PREREAD.json').read_text())
    original = json.loads((prior / 'STEERING_RESULTS.json').read_text())
    failures = [r for r in original if r['instrument']=='owner'] if coverage else [r for r in original if r['status'] != 'PASS']
    assert len(failures) == (4 if coverage else 7)
    binary = prior / 'bin/diffmap_block_coherence'
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == registration['binary']
    results = []
    for label in (('seed0', 'seed1', 'seed2') if coverage else ('seed0', 'dense')):
        model = output / 'models' / f'{label}.bin'
        folder = output / ('owner-seeds' if coverage else 'steering') / label
        folder.mkdir(parents=True, exist_ok=True)
        for failure in failures:
            case = next(c for c in registration['cases'] if
                        c['instrument'] == failure['instrument'] and c['row_id'] == failure['row'])
            for kind in ('reference', 'distorted'):
                path = Path(case['ref_path'] if kind=='reference' else case['dist_path'])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == case[f'{kind}_file_sha256']
            report = folder / f"{failure['instrument']}-{failure['row']}.json"
            command = [str(binary), case['ref_path'], case['dist_path'], '--block', str(case['block']),
                       '--bake', str(model), '--json', str(report)]
            env = dict(os.environ, ZENSIM_FORMULA_REV='5', ZENSIM_PREPARED_STEERING='1', RAYON_NUM_THREADS='1')
            for key in ('ZENSIM_NEIGHBOUR_EXACT','ZENSIM_FINITE_MOMENTS','ZENSIM_REPAIR_ALPHA',
                        'ZENSIM_REPAIR_SOURCE','ZENSIM_STEERING_BIN','ZENSIM_V2_DIAG'):
                env.pop(key,None)
            with report.with_suffix('.log').open('w') as log:
                log.write('COMMAND '+json.dumps(command)+'\n');log.flush()
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            assert result.returncode == 0
            data = json.loads(report.read_text())
            for kind in ('reference','distorted'):
                assert data[f'{kind}_pixels_sha256'] == case[f'{kind}_pixels_sha256']
            old = json.loads(Path(failure['report']).read_text())
            numerical_fields = ('base_score','m2','m3a','m3f','blocks','density_unsupported_ids',
                                'refinement_unsupported_ids','refinement_available','pixel_interventions')
            parity = all(data[k] == old[k] for k in numerical_fields) if label=='seed0' else None
            if label=='seed0':
                assert parity, failure['case']
            deltas = [b['score_delta'] for b in data['blocks']]
            results.append({'model':label,'case':failure['case'],'instrument':failure['instrument'],
                'row':failure['row'],'m2':data['m2'],'m3f':data['m3f'],'base_score':data['base_score'],
                'pass':data['m2']>=.99 and data['m3f']>=.70,'original_numerical_parity':parity,
                'report':str(report),'all_deltas_zero':all(x==0 for x in deltas) if deltas else None})
            print(f"{label} {failure['case']}: M2 {data['m2']} M3f {data['m3f']}",flush=True)
            (output / ('OWNER_SEEDS_PARTIAL.json' if coverage else 'STEERING_PARTIAL.json')).write_text(json.dumps(results,indent=2))
    (output / ('OWNER_SEEDS.json' if coverage else 'STEERING.json')).write_text(json.dumps(results,indent=2))


def adjudicate_owner_build(output):
    """Build a reviewable owner overlay; restore its exact baseline afterward."""
    import shutil
    import subprocess
    owner = Path('zensim-validate/src/bin/bake_verdict.rs')
    baseline = owner.read_bytes()
    assert baseline == (output / 'bake_verdict.base.rs').read_bytes()
    try:
        owner.write_bytes((output / 'bake_verdict.corrected.rs').read_bytes())
        commands = [
            ['cargo','test','-p','zensim-validate','--release','--bin','bake_verdict',
             'missing_features::tests::','--','--nocapture'],
            ['cargo','build','-p','zensim-validate','--release','--bin','bake_verdict'],
        ]
        (output / 'OWNER_BUILD_COMMANDS.json').write_text(json.dumps(commands,indent=2))
        for command in commands:
            subprocess.run(command,check=True)
        import os
        program = Path(os.environ['CARGO_TARGET_DIR']) / 'release/bake_verdict'
        shutil.copyfile(program, output / 'bin/bake_verdict')
        shutil.copymode(program, output / 'bin/bake_verdict')
    finally:
        owner.write_bytes(baseline)


def summarize(native, wasm):
    assert native["schema"] == 2, "cache-feature audit requires schema 2"
    if wasm is not None:
        assert wasm["schema"] == 2
    results = []
    for seed in native["seeds"]:
        permutations = seed["permutations"]
        baseline = permutations[0]["rows"]
        rows = [row for permutation in permutations for row in permutation["rows"]]
        identity = [row for row in baseline if row["dropped_bits"] == 0]
        ladders = []
        for width, height in native["geometries"]:
            ladder = [r for r in baseline if (r["width"], r["height"]) == (width, height)]
            ladders.append({"width": width, "height": height,
                            "scores": [r["pixel_score"] for r in ladder],
                            "nonincreasing": all(a["pixel_score"] >= b["pixel_score"]
                                                 for a, b in zip(ladder, ladder[1:]))})
        keys = ("pixel_bits", "cached_bits", "feature_bits", "identity_aware_bits", "read_bits",
                "feature_mismatches", "cached_feature_mismatches", "finite", "density_cells")
        tier_differences = sum(any(row[k] != base[k] for k in keys)
                               for permutation in permutations[1:]
                               for row, base in zip(permutation["rows"], baseline, strict=True))
        wasm_differences = None
        wasm_rows = []
        if wasm is not None:
            other = wasm["seeds"][seed["seed"]]
            assert other["declared_reads"] == seed["declared_reads"]
            wasm_rows = [r for p in other["permutations"] for r in p["rows"]]
            wasm_differences = sum(any(row[k] != base[k] for k in keys)
                                   for permutation in other["permutations"]
                                   for row, base in zip(permutation["rows"], baseline, strict=True))
        results.append({
            "seed": seed["seed"], "caller_width": seed["caller_width"],
            "declared_reads": seed["declared_reads"], "pairs_per_permutation": len(baseline),
            "permutations": [p["label"] for p in permutations],
            "finite": all(r["finite"] for r in rows),
            "pixel_cache_mismatches": sum(r["pixel_bits"] != r["cached_bits"] for r in rows),
            "pixel_identity_aware_feature_mismatches": sum(r["pixel_bits"] != r["identity_aware_bits"] for r in rows),
            "consumed_feature_mismatches": sum(len(r["feature_mismatches"]) for r in rows),
            "cached_feature_mismatches": sum(len(r["cached_feature_mismatches"]) for r in rows),
            "native_tier_row_mismatches": tier_differences,
            "wasm_row_mismatches": wasm_differences,
            "wasm_finite": all(r["finite"] for r in wasm_rows) if wasm is not None else None,
            "wasm_consumed_feature_mismatches": sum(len(r["feature_mismatches"]) for r in wasm_rows) if wasm is not None else None,
            "wasm_cached_feature_mismatches": sum(len(r["cached_feature_mismatches"]) for r in wasm_rows) if wasm is not None else None,
            "pixel_identity_exact_100": all(r["pixel_score"] == 100.0 for r in identity),
            "feature_identity_scores": [r["feature_score"] for r in identity],
            "feature_identity_band_97_5_to_100": all(97.5 <= r["feature_score"] <= 100.0 for r in identity),
            "no_distortion_above_identity": all(r["pixel_score"] <= 100.0 for r in baseline),
            "ladders_report_only": ladders,
        })
    return {"schema": 2, "scope": "three independent packed seeds; label-free synthetic SDR only",
            "geometries": native["geometries"], "dropped_bits": native["dropped_bits"], "seeds": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--wasm", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    wasm = json.loads(args.wasm.read_text()) if args.wasm else None
    if wasm is not None:
        assert native["geometries"] == wasm["geometries"]
        assert native["dropped_bits"] == wasm["dropped_bits"]
        assert len(native["seeds"]) == len(wasm["seeds"])
    summary = summarize(native, wasm)
    with args.output.open("x") as out:
        json.dump(summary, out, indent=2)
        out.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == '--adjudicate-prepare':
        adjudicate_prepare(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--adjudicate-verdict':
        adjudicate_verdict(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--adjudicate-corrected':
        adjudicate_verdict(Path(sys.argv[2]), corrected=True)
    elif len(sys.argv) == 3 and sys.argv[1] == '--adjudicate-ties':
        adjudicate_ties(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--adjudicate-steering':
        adjudicate_steering(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--adjudicate-coverage':
        adjudicate_steering(Path(sys.argv[2]), coverage=True)
    elif len(sys.argv) == 3 and sys.argv[1] == '--adjudicate-build-owner':
        adjudicate_owner_build(Path(sys.argv[2]))
    else:
        main()
