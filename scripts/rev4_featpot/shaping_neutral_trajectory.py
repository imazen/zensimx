"""Compare every completed control checkpoint; never launch or resume training."""
import argparse,json,subprocess
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha

STRIPPED_KEYS=('zentrain.repro','zentrain.monotone_research')


def canonical(source,binary,output,keys=STRIPPED_KEYS):
    path=source
    for index,key in enumerate(keys):
        target=output.with_name(output.name+f'-{index}.bin')
        subprocess.run([str(binary),'strip','--in',str(path),'--out',str(target),'--key',key],
                       check=True,capture_output=True,text=True)
        path=target
    return path


def compare_checkpoints(original,neutral,binary,output,epochs,metadata_keys=STRIPPED_KEYS):
    """Internal comparator; the public command separately requires all 240 epochs."""
    names=[f'ckpt_epoch{epoch:03}.bin' for epoch in range(epochs)]
    expected=set(names)
    if any({p.name for p in root.glob('ckpt_epoch*.bin')}!=expected for root in (original,neutral)):
        raise ValueError('both complete checkpoint inventories are required')
    output.mkdir(parents=True,exist_ok=False);rows={}
    for name in names:
        inputs={label:root/name for label,root in [('original',original),('neutral',neutral)]}
        pins={label:sha(path) for label,path in inputs.items()}
        directory=output/Path(name).stem;directory.mkdir()
        canonical_pins={}
        for label,path in inputs.items():
            result=canonical(path,binary,directory/label,metadata_keys)
            canonical_pins[label]=sha(result)
            if sha(path)!=pins[label]:raise ValueError('checkpoint changed during comparison')
        rows[name]=dict(input_sha256=pins,canonical_sha256=canonical_pins,
                        equal=canonical_pins['original']==canonical_pins['neutral'])
        (output/'PROGRESS.json').write_bytes(encoded(dict(required=epochs,compared=rows)))
    return dict(required=epochs,compared=rows,all_equal=all(row['equal'] for row in rows.values()),
                stripped_metadata_keys=list(metadata_keys))


def qualify(args):
    from shaping_reduced_scope import admit_compute
    from shaping_execution import load_registration
    admit_compute(args.study)
    _,cells,_=load_registration(args.study)
    original=json.loads((args.run/'RUN.json').read_text());terminal=json.loads((args.run/'RESULT.json').read_text())
    qualified=json.loads((args.qualification/'RESULT.json').read_text())
    if (original['cell'] not in cells or original['cell']['stage']!='inner' or original['cell']['family'] not in NEURAL
        or terminal['stage']!='trained' or terminal['exit_code']!=0
        or original['plan_sha256']!=sha(args.study/'PLAN.json') or terminal['cell']!=original['cell']
        or original['epochs']!=240 or qualified['epochs']!=240 or qualified['original_run']!=str(args.run)
        or qualified['original_result_sha256']!=sha(args.run/'RESULT.json')
        or qualified['build_sha256']!=sha(args.binary_dir/'BUILD.json')):
        raise ValueError('complete original and qualified control identities required')
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('trajectory comparison binary drift')
    if sha(args.run/'checkpoints/ckpt_epoch239.bin')!=terminal['bake_sha256']:
        raise ValueError('original final model differs')
    # The native qualifier writes RESULT only after successful CLI finalization
    # and its final comparison. An in-flight raw dump is not a finalized bake.
    if type(qualified.get('match')) is not bool:raise ValueError('terminal native qualification required')
    result=compare_checkpoints(args.run/'checkpoints',args.qualification/'checkpoints',
                               args.binary_dir/'bake_dial_refit',args.output,240)
    result.update(schema='native-control-trajectory-parity-v1',cell=original['cell'],
                  original_result_sha256=sha(args.run/'RESULT.json'),
                  qualification_result_sha256=sha(args.qualification/'RESULT.json'),
                  build_sha256=sha(args.binary_dir/'BUILD.json'),driver_sha256=sha(__file__),
                  selection_candidate=False,outer_results_read=False)
    (args.output/'RESULT.json').write_bytes(encoded(result))
    if not result['all_equal']:raise ValueError('native control trajectory differs; preserve every mismatch')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','run','qualification','binary-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    result=qualify(p.parse_args())
    print(json.dumps({k:v for k,v in result.items() if k!='compared'}),flush=True)
