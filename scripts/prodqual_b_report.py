#!/usr/bin/env python3
"""Build qualification receipts from owned canonical outputs, no new eval reads."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    freeze = read('prodqual_b_registration_2026-10-07.json')
    standard = read('standard/gaddr.json'); ladder = read('ladder/gaddr.json')
    steering = read('STEERING_RESULTS.json'); incumbent = read('STEERING_RESULTS_B.json')
    assert len(steering) == len(incumbent) == 135
    groups = {}
    for name in ['broad', 'owner', 'jpeg8', 'm3']:
        rows = [r for r in steering if r['instrument'] == name]
        groups[name] = {'n': len(rows), 'pass': sum(r['status']=='PASS' for r in rows),
            'min_m2': min(r['m2'] for r in rows), 'min_m3f': min(r['m3f'] for r in rows)}
    codec = read('steercodec/summary.json'); bj = read('steercodec/summary-B-judge.json')
    assert len(codec['rows']) == 48 and len(codec['summary']) == len(bj['summary']) == 6
    chroma = read('chromaq/summary.json')
    results = {'schema': 'prodqual-b-verdict-v1', 'base': freeze['base'],
        'model_sha256': freeze['composition']['packed_model_sha256'],
        'qualification': 'FAIL', 'frozen_bytes_and_thresholds_unchanged': True,
        'protected_payloads_read': False, 'human_labels_read': False,
        'gates': {
            'G-RANK': {'state':'blocked', 'human_panels_evaluated':0,
                'reason':'no explicitly admitted independent human Rev5 roots; registered T0 and AIC panels excluded',
                'additional_rule':'supported aggregate/content and band rule pending'},
            'G-DIAL': {'state':'PASS', 'bars':freeze['thresholds']['G-DIAL'],
                'standard':standard['measured']['grid'], 'ladder':ladder['measured']['grid']},
            'G-ADDR': {'state':'FAIL', 'standard_contract':standard['contract'],
                'standard_checks':[{k:v for k,v in c.items() if k in ['id','bar','measured','state','tier']} for c in standard['checks']], 'ladder_contract':ladder['contract'],
                'ladder_checks':[{k:v for k,v in c.items() if k in ['id','bar','measured','state','tier']} for c in ladder['checks']], 'codec_floors': {'state':'blocked','qualified':0,
                    'required':5,'names':freeze['thresholds']['G-ADDR']['codec_floors'],
                    'reason':'new-grid registry/incumbent receipts and per-cell mentor truth missing; resolvable unchanged'}},
            'negative_tail': {'state':'PASS', **standard['measured']['negtail']},
            'native_pixel_identity': {'state':'PASS', **read('pixel-identity/RESULT.json')},
            'G-STEER': {'state':'FAIL','groups':groups,'failures':[r for r in steering if r['status']=='FAIL'],
                'B_comparison':{'state':'blocked','prepare_refusals':len(incumbent),
                    'reason':'masked/IW IDs f228-f371 unsupported by spatial refinement'}},
            'STEERCODEC': {'state':'rule pending','swaps':48,'composites':192,'pairs':264,
                'min_m2':min(r['m2'] for r in codec['rows']), 'min_m3f':min(r['m3f'] for r in codec['rows']),
                'summary':codec['summary'], 'B_judges_seed0_composites':bj['summary'],
                'B_meaning':'scores same seed0-selected pixels; not a B-map baseline'},
            'CHROMAQ': {'state':'rule pending','cells':chroma['n_cells'],'images':chroma['images'],
                'per_image_chroma_srocc':chroma['per_image_srocc']['chroma_only'],
                'seed0_monotonicity':chroma['monotone']['seed0']},
            'G-RD': {'state':'blocked','native_codecs_qualified':0,'required':4,
                'reason':'registered frozen-model native intervention packet missing; guard adapter/tests ready'},
            'G-TARGET': {'state':'blocked','requests_qualified':0,
                'reason':'matching production composition/configuration/TRAIN calibration/bounds and witnesses missing'},
            'ZCTH_v4_compatibility': {'state':'PASS','head_sha256':
                '568380f1bd2fe6202ebcc5a194078d7885035f466abe93ada3a4b3ea8f4a26f2',
                'meaning':'attachment accepted only; companion not attached to primary evaluation'},
            'integrity': {'state':'blocked','additional_rule':'reviewed class/source strata pending',
                'numerical_corruption_populations':0},
            'HDR': {'state':'blocked','meaning':'no independent human HDR accuracy qualification'},
            'speed': {'state':'blocked','research_parity_checks':384,'paired_segments':0,'required_segments':192,
                'RSS_cells':0,'required_RSS_cells':112,'reason':'available SPEEDQ weights differ from frozen seed0'}},
        'evidence_root':str(OUT),'tower_root':str(OUT).replace('/mnt/v/','/mnt/tower/')}
    text = json.dumps(results,indent=2,allow_nan=False)+'\n'
    assert len(text.encode()) < 30000
    (REPO/'benchmarks/prodqual_b_2026-10-07.json').write_text(text)
    (OUT/'RESULT.json').write_text(text)
    exposure = {'schema':'prodqual-b-exposure-final-v1','model':results['model_sha256'],
        'protected_payloads_read':False,'human_labels_read':False,'table_reads':[], 'pixel_reads':[]}
    for r in freeze['inputs']:
        exposure['table_reads'].append({**r,
            'read_mode':'whole-file hash verification; canonical numerical evaluation' if r['name'] in ['standard','ladder','negtail','identity'] else 'whole-file hash verification only',
            'keys_read_mode':'hash plus geometry/admission payload' if r['name'] in ['identity','broad','owner','jpeg8','m3'] else 'whole-file hash verification only',
            'numerical_invocations':2 if r['name'] in ['negtail','identity'] else (1 if r['name'] in ['standard','ladder'] else 0)})
    exposure['pixel_reads'].append({'population':'135 steering cases, 59 pairs, seed0 and B',
        'rows':read('STEERING_PREREAD.json')['cases'],'repeated_cases':'same 135 in STEERING_PREREAD_B.json'})
    exposure['pixel_reads'].append({'population':'38 native identities, seed0 and B',
        'rows':read('pixel-identity/PREREAD.json')['rows']})
    exposure['pixel_reads'].append({'population':'32 CHROMAQ references /1760 cells',
        'hashes':read('chromaq/PIXEL_INPUT_HASHES.json'),
        'registration':read('prodqual_b_chromaq_registration_2026-10-07.json')})
    exposure['pixel_reads'].append({'population':'8 STEERCODEC sources /72 native decodes',
        'hashes':read('steercodec/INPUT_HASHES.json'),
        'registration':read('prodqual_b_steercodec_registration_2026-10-07.json'),
        'generated_composites':192,'scored_pairs':264,'judge_reads':['seed0','serving B','SSIM2','butteraugli']})
    exposure['model_and_metadata_reads']={'pins':read('PROGRAM_MODEL_PINS.json'),
        'companion':'v4 SHIPPATH6 header/hash and canonical attachment probe only',
        'RD':'synthetic guard fixtures only; no real measurements',
        'speed':'existing base report metadata and DONE marker; no timing/stress payload'}
    (OUT/'EXPOSURE_FINAL.json').write_text(json.dumps(exposure,indent=2)+'\n')
    print('qualification FAIL; reports generated without new evaluation payload reads')


if __name__ == '__main__':
    main()
