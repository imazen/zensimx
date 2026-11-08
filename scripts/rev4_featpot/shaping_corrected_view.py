"""Explicit corrected-condition view; historical registrations stay immutable."""
import argparse,json,re
from pathlib import Path
from shaping_completion import REPO,encoded,sha
from shaping_corrected_registration import BUILD_SHA256,MATRIX_SHA256,FAMILIES,validate_matrix
from shaping_control_compatibility import FAMILIES as CONTROL_CLASSES


def corrected_cells(original,matrix):
    entries=validate_matrix(matrix,original)
    replacements={e['original_condition']['id']:dict(e['original_condition'],id=e['id']) for e in entries}
    result=[replacements.get(c['id'],c) for c in original]
    if len({c['id'] for c in result})!=len(original):raise ValueError('corrected cell collision')
    return result


def trajectory(record,family,case):
    """Require every finalized epoch and the prespecified original control."""
    mono=record.get('schema')=='native-control-trajectory-parity-v1'
    if mono:
        if family!='silu_monotone' or record.get('selection_candidate') is not False:
            raise ValueError('unexpected separate trajectory qualification')
        comparison=record
    else:
        if (record.get('stage')!='qualified' or record.get('scientific_selection') is not False
            or record.get('epochs')!=240 or record.get('pairs_per_epoch')!=50000
            or record.get('all_240_checkpoints_equal') is not True):
            raise ValueError('terminal full native compatibility qualification required')
        comparison=record['trajectory']
    if (record.get('cell')!=case['cell'] or record.get('outer_results_read') is not False
        or record.get('build_sha256')!=BUILD_SHA256):raise ValueError('control qualification identity differs')
    expected_keys=['zentrain.repro']
    if family in ('softplus','silu_monotone','silu_diagnostic'):expected_keys+=['zentrain.monotone_research']
    rows=comparison['compared']
    if (comparison.get('required')!=240 or comparison.get('all_equal') is not True
        or comparison.get('stripped_metadata_keys')!=expected_keys
        or set(rows)!={f'ckpt_epoch{e:03}.bin' for e in range(240)}):
        raise ValueError('all 240 exact control epochs and approved metadata scope required')
    for row in rows.values():
        canonical=row['canonical_sha256'];inputs=row['input_sha256']
        if (row.get('equal') is not True or set(canonical)!={'original','neutral'}
            or set(inputs)!={'original','neutral'} or canonical['original']!=canonical['neutral']
            or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in [*canonical.values(),*inputs.values()])):
            raise ValueError('control trajectory mismatch or invalid identity')
    if rows['ckpt_epoch239.bin']['input_sha256']['original']!=case['original_final_sha256']:
        raise ValueError('trajectory does not bind the registered original final model')


def registration(root,original):
    path=root/'corrected_thresholds/REGISTRATION.json';record=json.loads(path.read_text())
    if (record['schema']!='corrected-shaped-threshold-registration-v1' or record['core_fit_target']!=1461
        or record['corrected_condition_count']!=288 or record['source_build_sha256']!=BUILD_SHA256
        or record['plan_sha256']!=sha(root/'PLAN.json') or record['outer_results_read'] is not False
        or record['execution_activated'] is not False):raise ValueError('frozen corrected registration differs')
    for name,pin in record['evidence_sha256'].items():
        evidence=Path(name)
        if evidence.name=='SHAPING_THRESHOLD_AMENDMENT.md' and evidence.parent.name=='docs':
            evidence=REPO/'docs'/evidence.name
        if sha(evidence)!=pin:raise ValueError('corrected registration evidence drift: '+name)
    for host,pin in record['observation_sha256'].items():
        if sha(path.parent/'observations'/(host+'.json'))!=pin:raise ValueError('frozen omission observation drift')
    matrix_path=Path(record['matrix_path'])
    if sha(matrix_path)!=MATRIX_SHA256:raise ValueError('corrected matrix drift')
    matrix=json.loads(matrix_path.read_text());effective=corrected_cells(original,matrix)
    return record,matrix,effective


def certify(root,proofs):
    """Build a certificate, never activate or launch scientific work implicitly."""
    from shaping_execution import load_registration
    _,original,_=load_registration(root,original=True);record,matrix,effective=registration(root,original)
    compatibility=root.parent/'corrected-control-qualification-20261108/REGISTRATION.json'
    controls=json.loads(compatibility.read_text())
    if (controls['build_sha256']!=BUILD_SHA256 or controls['plan_sha256']!=record['plan_sha256']
        or set(controls['cases'])!=set(CONTROL_CLASSES) or set(proofs)!=set(CONTROL_CLASSES)):
        raise ValueError('all six registered control classes required')
    pins={str(root/'corrected_thresholds/REGISTRATION.json'):sha(root/'corrected_thresholds/REGISTRATION.json'),
          str(compatibility):sha(compatibility)}
    for family,path in proofs.items():
        proof=json.loads(path.read_text());case=controls['cases'][family]
        trajectory(proof,family,case)
        source=Path(case['run'])
        if proof.get('original_result_sha256',proof.get('source_result_sha256'))!=sha(source/'RESULT.json'):
            raise ValueError('qualified original result differs')
        if family!='silu_monotone' and proof.get('registration_sha256')!=sha(compatibility):
            raise ValueError('qualification registration differs')
        pins[str(path)]=sha(path)
    return dict(schema='corrected-shaped-threshold-compatibility-certificate-v1',
        registration_sha256=sha(root/'corrected_thresholds/REGISTRATION.json'),
        matrix_sha256=MATRIX_SHA256,build_sha256=BUILD_SHA256,control_classes=list(CONTROL_CLASSES),
        evidence_sha256=pins,effective_matrix_sha256=__import__('hashlib').sha256(encoded(effective)).hexdigest(),
        corrected_cells=288,reused_primary_controls=528,matched_fresh_controls=72,
        outer_results_read=False,scientific_fits_started=False,driver_sha256=sha(__file__))


def activated_view(root,original):
    path=root/'corrected_thresholds/ACTIVATION.json'
    if not path.exists():return original
    activation=json.loads(path.read_text());certificate=path.with_name('COMPATIBILITY.json')
    if (activation.get('schema')!='corrected-shaped-threshold-activation-v1'
        or activation.get('certificate_sha256')!=sha(certificate)
        or activation.get('outer_results_read') is not False):raise ValueError('corrected activation differs')
    saved=json.loads(certificate.read_text());record,matrix,effective=registration(root,original)
    if (saved['schema']!='corrected-shaped-threshold-compatibility-certificate-v1'
        or saved['registration_sha256']!=sha(path.with_name('REGISTRATION.json'))
        or saved['matrix_sha256']!=MATRIX_SHA256 or saved['build_sha256']!=BUILD_SHA256
        or saved['control_classes']!=list(CONTROL_CLASSES) or saved['outer_results_read'] is not False
        or saved['effective_matrix_sha256']!=__import__('hashlib').sha256(encoded(effective)).hexdigest()):
        raise ValueError('corrected compatibility certificate differs')
    for name,pin in saved['evidence_sha256'].items():
        if sha(name)!=pin:raise ValueError('corrected compatibility evidence drift: '+name)
    return effective


def activate(root,proofs):
    path=root/'corrected_thresholds/ACTIVATION.json';certificate=path.with_name('COMPATIBILITY.json')
    if path.exists() or certificate.exists():raise ValueError('preserve prior activation or partial certification')
    if list((root/'selection').glob('outer-*/OUTER_ADMISSION.json')):
        raise ValueError('corrected selection must precede outer admission')
    result=certify(root,proofs)
    with certificate.open('xb') as stream:stream.write(encoded(result))
    activation=dict(schema='corrected-shaped-threshold-activation-v1',certificate_sha256=sha(certificate),
        outer_results_read=False,driver_sha256=sha(__file__))
    with path.open('xb') as stream:stream.write(encoded(activation))
    return activation


def corrected_entry(root,cell):
    if not (root/'corrected_thresholds/ACTIVATION.json').exists():return None
    from shaping_execution import load_registration
    _,original,_=load_registration(root,original=True)
    effective=activated_view(root,original)
    _,matrix,_=registration(root,original)
    entries=[e for e in matrix['corrected_pretrained_conditions'] if e['id']==cell['id']]
    if not entries:return None
    if cell not in effective:raise ValueError('corrected cell parameters differ')
    return entries[0]


def admit_parent(root,cell,parent,original_build,new_build):
    entry=corrected_entry(root,cell)
    if entry is None or entry['parent']!=parent or new_build!=BUILD_SHA256:
        raise ValueError('qualified corrected parent binding required')
    controls=json.loads((root.parent/'corrected-control-qualification-20261108/REGISTRATION.json').read_text())
    if original_build!=controls['cases'][parent['family']]['original_build_sha256']:
        raise ValueError('parent build is outside qualified control class')


def admit_fit(root,cell,build,epochs):
    if cell['stage']!='inner' or cell['family'] not in FAMILIES:return None
    if not (root/'corrected_thresholds/REGISTRATION.json').exists():return None
    entry=corrected_entry(root,cell)
    if entry is None:raise ValueError('historical pretrained condition is not a corrected scientific fit')
    if build!=BUILD_SHA256 or epochs!=120:raise ValueError('corrected fit binary or exposure differs')
    return dict(registration_sha256=sha(root/'corrected_thresholds/REGISTRATION.json'),
        activation_sha256=sha(root/'corrected_thresholds/ACTIVATION.json'),condition=entry)


def base_path(root,family,outer):
    # Only an explicitly activated study changes selection paths. Originals remain
    # readable for historical diagnostics and unchanged families retain their path.
    suffix='-corrected-threshold' if (root/'corrected_thresholds/ACTIVATION.json').exists() and family in FAMILIES else ''
    return root/'selection'/f'outer-{outer}'/(family+suffix+'-base.json')


def pending(root):
    return (root/'corrected_thresholds/REGISTRATION.json').exists() and not (root/'corrected_thresholds/ACTIVATION.json').exists()


def require_activation(root):
    if pending(root):raise ValueError('corrected protocol awaits native compatibility activation')


def require_downstream_build(root,family,build_sha256):
    """Retain corrected initialization in new pretrained rescue/refit phases.

    Callers first reproduce the full registration/selection or outer admission.
    Unchanged control families keep their independently qualified runtimes.
    """
    if family not in FAMILIES or not (root/'corrected_thresholds/REGISTRATION.json').exists():return
    require_activation(root)
    if build_sha256!=BUILD_SHA256:
        raise ValueError('corrected pretrained downstream fit requires the registered corrected native build')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--proof',action='append',required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--activate',action='store_true')
    args=parser.parse_args();proofs={}
    if args.activate==bool(args.output):parser.error('choose --output for certification or --activate')
    for value in args.proof:
        family,path=value.split('=',1)
        if family in proofs:parser.error('duplicate control class')
        proofs[family]=Path(path)
    if args.activate:result=activate(args.study,proofs)
    else:
        result=certify(args.study,proofs)
        with args.output.open('xb') as stream:stream.write(encoded(result))
    print(json.dumps(result),flush=True)
