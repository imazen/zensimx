"""Admit one archived, exact-trajectory runtime for unstarted joint-shaping fits."""
import argparse,json,re
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha

BASE_BUILD='56a05880e05dcfa78d28207a88df6f605888814e47707e216e5941d743ac0cff'
CLASSES={'product_joint_identity':('caa39cae7cd461126a97',240),
    'separate_joint_identity':('1d9acf972b7722929b41',240),
    'product_pretrained_joint':('5e2cce20a78639f2cca8',120),
    'separate_pretrained_joint':('a868b99338b668c63467',120)}


def check_trajectory(proof,epochs,references):
    value=proof['trajectory'];expected={f'ckpt_epoch{e:03}.bin' for e in range(epochs)}
    if (proof['stage']!='qualified' or proof.get(f'all_{epochs}_checkpoints_equal') is not True
        or value['required']!=epochs or value['all_equal'] is not True
        or value['stripped_metadata_keys']!=['zentrain.repro'] or set(value['compared'])!=expected
        or set(references)!=expected):raise ValueError('complete same-order trajectory proof required')
    for name,row in value['compared'].items():
        if (row['equal'] is not True or row['canonical_sha256']['original']!=row['canonical_sha256']['neutral']
            or row['input_sha256']['original']!=references[name]
            or any(not re.fullmatch('[0-9a-f]{64}',pin) for pin in (*row['canonical_sha256'].values(),*row['input_sha256'].values()))):
            raise ValueError('runtime trajectory checkpoint differs')


def prepare(study,baseline,candidate,identity,pretrained):
    from shaping_artifact_retention import archive_inventory
    from shaping_completion import REPO
    if sha(baseline/'BUILD.json')!=BASE_BUILD:raise ValueError('original corrected runtime required')
    base=json.loads((baseline/'BUILD.json').read_text());new=json.loads((candidate/'BUILD.json').read_text())
    if set(base['binary_sha256'])!=set(new['binary_sha256']):raise ValueError('runtime component set differs')
    for name,pin in new['binary_sha256'].items():
        if sha(candidate/name)!=pin:raise ValueError('candidate runtime component differs')
        if name!='zensim_mlp_train' and pin!=base['binary_sha256'][name]:
            raise ValueError('inference/evaluation runtime must remain byte-identical')
    build=sha(candidate/'BUILD.json');root=study/'runtime_equivalence'/build
    if root.exists():raise ValueError('preserve existing runtime certificate')
    artifacts={str(baseline/'BUILD.json'):BASE_BUILD,str(candidate/'BUILD.json'):build}
    cases={}
    for directory in (identity,pretrained):
        _,manifest=archive_inventory(directory)
        for name in ('MIRROR_VERIFIED.json','SHA256SUMS','REGISTRATION.json'):
            artifacts[str(directory/name)]=sha(directory/name)
        registration=json.loads((directory/'REGISTRATION.json').read_text())
        epochs=240 if directory==identity else 120
        schema='paired-gradient-native-trajectory-v1' if epochs==240 else 'paired-gradient-corrected-pretrained-v1'
        if (registration['schema']!=schema or registration['epochs']!=epochs or registration['pairs_per_epoch']!=50000
            or registration['outer_results_read'] is not False
            or registration['plan_sha256']!=sha(study/'PLAN.json') or registration['build_sha256']!=build):
            raise ValueError('qualified runtime registration differs')
        if epochs==120 and (registration['imported_original_shaping_banks'] is not True or registration['pretraining_replayed'] is not False):
            raise ValueError('original pretrained banks required')
        for family,case in registration['cases'].items():
            if family not in CLASSES:raise ValueError('unsupported runtime control class')
            key,epochs=CLASSES[family]
            expected=dict(id=key,family=family,stage='inner',partition='inner-0-0',setting=0,seed=0)
            if case['cell']!=expected:raise ValueError('prespecified runtime control differs')
            path=directory/family/('RESULT.json' if epochs==240 else 'QUALIFICATION.json')
            proof=json.loads(path.read_text());check_trajectory(proof,epochs,case['reference_sha256'])
            if proof['build_sha256']!=build:raise ValueError('trajectory runtime differs')
            for item in (path,directory/family/'RUN.json',directory/family/'RESULT.json'):
                relative=item.relative_to(directory).as_posix()
                if manifest.get(relative)!=sha(item):raise ValueError('trajectory artifact differs from verified archive')
                artifacts[str(item)]=sha(item)
            run=json.loads((directory/family/'RUN.json').read_text())
            if (run['cell']!=expected or run['epochs']!=epochs or run['pairs_per_epoch']!=50000
                or run['build_sha256']!=build):
                raise ValueError('qualified native exposure differs')
            cases[family]=dict(cell=expected,epochs=epochs,proof=str(path),proof_sha256=sha(path),
                registration=str(directory/'REGISTRATION.json'),registration_sha256=sha(directory/'REGISTRATION.json'))
    if set(cases)!=set(CLASSES):raise ValueError('all four joint control classes required')
    original=study/'corrected_thresholds/COMPATIBILITY.json';compat=json.loads(original.read_text())
    if compat['build_sha256']!=BASE_BUILD:raise ValueError('corrected control certificate differs')
    artifacts[str(original)]=sha(original)
    policy=REPO/'docs/SHAPING_RUNTIME_EQUIVALENCE.md';root.mkdir(parents=True)
    (root/'POLICY.md').write_bytes(policy.read_bytes());artifacts[str(root/'POLICY.md')]=sha(root/'POLICY.md')
    record=dict(schema='paired-gradient-runtime-equivalence-v1',plan_sha256=sha(study/'PLAN.json'),
        baseline_build_sha256=BASE_BUILD,candidate_build_sha256=build,baseline_binary_dir=str(baseline),
        candidate_binary_dir=str(candidate),cases=cases,artifact_sha256=artifacts,
        source_driver_sha256=sha(__file__),checkpoint_comparisons=720,scientific_inputs_changed=False,
        scope='only new outer, variant and extension registrations; preserve existing fits and all selection gates')
    (root/'CERTIFICATE.json').write_bytes(encoded(record));return root/'CERTIFICATE.json'


def validate(study,path):
    record=json.loads(path.read_text())
    if (record['schema']!='paired-gradient-runtime-equivalence-v1' or record['plan_sha256']!=sha(study/'PLAN.json')
        or record['baseline_build_sha256']!=BASE_BUILD or set(record['cases'])!=set(CLASSES)
        or record['scientific_inputs_changed'] is not False or record['checkpoint_comparisons']!=720
        or path.parent.name!=record['candidate_build_sha256']):raise ValueError('runtime certificate identity differs')
    for name,pin in record['artifact_sha256'].items():
        if sha(name)!=pin:raise ValueError('runtime certificate evidence drift')
    candidate=Path(record['candidate_binary_dir']);baseline=Path(record['baseline_binary_dir'])
    if sha(candidate/'BUILD.json')!=record['candidate_build_sha256'] or sha(baseline/'BUILD.json')!=BASE_BUILD:
        raise ValueError('runtime build identity differs')
    current=json.loads((candidate/'BUILD.json').read_text());original=json.loads((baseline/'BUILD.json').read_text())
    if set(current['binary_sha256'])!=set(original['binary_sha256']):raise ValueError('runtime component set differs')
    for name,pin in current['binary_sha256'].items():
        if sha(candidate/name)!=pin or (name!='zensim_mlp_train' and pin!=original['binary_sha256'][name]):
            raise ValueError('qualified runtime component drift')
    for family,item in record['cases'].items():
        key,epochs=CLASSES[family];expected=dict(id=key,family=family,stage='inner',partition='inner-0-0',setting=0,seed=0)
        if item['cell']!=expected or item['epochs']!=epochs or sha(item['proof'])!=item['proof_sha256']:
            raise ValueError('runtime control class differs')
        registration=json.loads(Path(item['registration']).read_text())
        if sha(item['registration'])!=item['registration_sha256'] or registration['build_sha256']!=record['candidate_build_sha256']:
            raise ValueError('runtime control registration differs')
        check_trajectory(json.loads(Path(item['proof']).read_text()),epochs,registration['cases'][family]['reference_sha256'])
    return record


def active(study,*,build=None,family=None):
    found=[]
    for marker in sorted((study/'runtime_equivalence').glob('*/ACTIVATION.json')):
        binding=json.loads(marker.read_text());path=marker.with_name('CERTIFICATE.json')
        if sha(path)!=binding['certificate_sha256']:raise ValueError('active runtime certificate differs')
        info=json.loads(path.read_text())
        if build is not None and info['candidate_build_sha256']!=build:continue
        if family is not None and family not in info['cases']:continue
        found.append((binding['activated_utc'],path,validate(study,path)))
    return max(found,key=lambda row:row[0]) if found else None


def canonical_build(study,family,build):
    if build==BASE_BUILD:return build
    matched=active(study,build=build,family=family)
    if matched is None:raise ValueError('no qualified runtime for this family/build')
    return matched[2]['baseline_build_sha256']


def runtime_family(study,key,kind):
    if kind=='variant':
        from shaping_variant_execution import resolve
        return resolve(study,key)['resolved_family']
    if kind=='extension':
        from shaping_extension import resolve
        return resolve(study,key)['cell']['family']
    from shaping_execution import load_registration
    _,cells,_=load_registration(study);found=[c for c in cells if c['id']==key]
    if not found:
        from shaping_outer_admission import resolve
        for outer in range(3):
            if (study/'selection'/f'outer-{outer}'/'OUTER_ADMISSION.json').exists():
                found.extend(c for c in resolve(study,outer)[0]['additional_cells'] if c['id']==key)
    if len(found)!=1:raise ValueError('one registered runtime family required')
    return found[0]['family']


def select(study,key,kind,default):
    if kind not in ('outer','variant','extension'):return default,None
    if not (study/'runtime_equivalence').exists():return default,None
    family=runtime_family(study,key,kind);matched=active(study,family=family)
    if matched is None:return default,None
    _,path,record=matched
    if sha(default/'BUILD.json') not in (BASE_BUILD,record['candidate_build_sha256']):
        raise ValueError('preserve explicit unrelated runtime selection')
    evidence=dict(certificate=str(path),certificate_sha256=sha(path),family=family,
        candidate_build_sha256=record['candidate_build_sha256'],baseline_build_sha256=BASE_BUILD)
    return Path(record['candidate_binary_dir']),evidence


def verify_binding(study,record):
    item=record.get('runtime_compatibility')
    if item is None:return
    path=Path(item['certificate']);value=validate(study,path)
    if (sha(path)!=item['certificate_sha256'] or item['candidate_build_sha256']!=record['build_sha256']
        or value['candidate_binary_dir']!=record['binary_dir']
        or runtime_family(study,record['cell']['id'],record['execution_kind'])!=item['family']
        or item['family'] not in value['cases']):raise ValueError('worker runtime compatibility binding differs')


def activate(study,path):
    record=validate(study,path);marker=path.with_name('ACTIVATION.json')
    if marker.exists():raise ValueError('preserve existing runtime activation')
    with marker.open('xb') as stream:stream.write(encoded(dict(certificate_sha256=sha(path),
        candidate_build_sha256=record['candidate_build_sha256'],activated_utc=datetime.now(timezone.utc).isoformat())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','activate'))
    for name in ('study','baseline','candidate','identity','pretrained','certificate'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.mode=='prepare':print(prepare(a.study,a.baseline,a.candidate,a.identity,a.pretrained))
    else:activate(a.study,a.certificate)
