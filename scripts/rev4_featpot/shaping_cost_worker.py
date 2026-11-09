"""Prepare all 48 admitted outer models on a worker without timing or fitting."""
import argparse,hashlib,json,socket
from pathlib import Path
from shaping_completion import REPO,NEURAL,encoded,sha
from shaping_bootstrap_worker import output_scope


def context(study,binaries,host):
    from shaping_reduced_scope import admit_compute
    from shaping_cost_finalists import outer_training_rows
    from shaping_outer_evidence import collect
    from tree_study import RAW_IDS
    admit_compute(study,host=host)
    _,_,pins,chosen=outer_training_rows(study)
    _,_,rows,evidence,qualifications=collect(study);pins.update(evidence)
    if len(rows)!=48 or not all('role' in row for row in rows):
        raise ValueError('all 48 admitted outer finalist models required')
    for row in rows:
        run=Path(row['run_directory']);report=row['report'];family=row['cell']['family']
        if family in NEURAL:
            if 'reexport_receipt' in report:
                model=Path(json.loads(Path(report['reexport_receipt']).read_text())['bake'])
            else:
                epochs=json.loads((run/'RUN.json').read_text())['epochs']
                model=run/'fit/checkpoints'/f'ckpt_epoch{epochs-1:03}.bin'
            pin=report['checkpoint_sha256'];dense=Path(row['evaluation_directory'])/'dense'/(pin[:16]+'.bin')
            pins[str(dense)]=sha(dense)
        else:
            model=run/('model.opaque-vector.zcth' if family=='extra_trees' else 'native/model.bin')
            q=report['qualification'];pin=q['native_tree_gate']['model_sha256'] if family=='extra_trees' else q['model_sha256']
        if sha(model)!=pin:raise ValueError('outer cost source model differs')
        pins[str(model)]=pin
    from shaping_cost_complexity import inputs as complexity_inputs
    complexity_pins,_,_=complexity_inputs(study,rows);pins.update(complexity_pins)
    build=binaries/'BUILD.json';converter=binaries/'bake_dial_refit'
    if sha(converter)!=json.loads(build.read_text())['binary_sha256']['bake_dial_refit']:
        raise ValueError('outer cost converter differs')
    pins.update({str(build):sha(build),str(converter):sha(converter)})
    from shaping_rescue_worker import source_identity
    normalized={source_identity(name):pin for name,pin in pins.items()}
    if len(normalized)!=len(pins):raise ValueError('ambiguous outer cost evidence paths')
    return dict(input_sha256=normalized,cells=[dict(cell=r['cell'],role=r['role'],
        schedule=json.loads((Path(r['run_directory'])/'RUN.json').read_text()).get('schedule')) for r in rows],
        chosen_rows={str(o):chosen[o] for o in range(3)},qualifications=qualifications,
        raw_ids=RAW_IDS,scientific_fit_count=0,final_holdout_read=False)


def register(study,binaries,result,host,output):
    from shaping_rescue_worker import HOSTS,reads,seal_inputs
    # Initialize third-party imports before capturing actual recipe reads.
    import shaping_cost_finalists,shaped_linear_study,tree_study,shaping_outer_evidence
    # Native preparation calls these lazily; bind their source in its capsule.
    import shaping_parameter_count,shaping_training_resources
    if host not in HOSTS or not output_scope(study,result):
        raise ValueError('approved worker and fresh study-local cost output required')
    result.parent.mkdir(parents=True,exist_ok=True)
    with reads() as paths:resolution=context(study,binaries,host)
    cell=dict(stage='cost_prepare',family='native_cost',partition='outer-all',
        id=hashlib.sha256(encoded(resolution)).hexdigest()[:20])
    record=seal_inputs(paths,study,cell,host,binaries,None,resolution,execution_kind='cost_prepare')
    record.update(result_output=str(result),cost_driver_sha256=sha(__file__),final_holdout_read=False)
    output.mkdir(parents=True,exist_ok=False);(output/'REQUEST.json').write_bytes(encoded(record))
    (output/'ASSIGNMENT.json').write_bytes(encoded(dict(host=host,cells=[cell['id']],
        plan_sha256=sha(study/'PLAN.json'),build_sha256=sha(binaries/'BUILD.json'))))
    return record


def reproduce(record,*,host=None):
    study=Path(record['study']);result=Path(record['result_output'])
    if (record['cost_driver_sha256']!=sha(__file__) or record['final_holdout_read'] is not False
        or not output_scope(study,result)):
        raise ValueError('cost preparation source or fresh output scope differs')
    result.parent.mkdir(parents=True,exist_ok=True)
    resolution=context(study,Path(record['binary_dir']),host or socket.gethostname())
    command=['python3','-B',str(REPO/'scripts/rev4_featpot/shaping_cost_finalists.py'),
        '--study',str(study),'--binary-dir',record['binary_dir'],'--output',str(result)]
    return resolution,command


def validate_result(record,directory):
    from shaping_rescue_worker import source_identity
    marker=json.loads((directory/'WORKLOADS.json').read_text())
    if marker['models']!=48 or marker['timing_measured'] is not False or len(marker['workloads'])!=3:
        raise ValueError('complete untimed outer cost preparation required')
    all_pins={}
    for outer,path in enumerate(marker['workloads']):
        if Path(path)!=Path(record['result_output'])/f'outer-{outer}'/'WORKLOAD.json':
            raise ValueError('cost workload location differs')
        root=directory/f'outer-{outer}';w=json.loads((root/'WORKLOAD.json').read_text())
        p=json.loads((root/'PROVENANCE.json').read_text())
        if (p['outer']!=outer or p['workload_sha256']!=sha(root/'WORKLOAD.json')
            or p['original_training_indices']!=record['resolution']['chosen_rows'][str(outer)]
            or p['timing_measured'] is not False
            or p['convergence_qualifications']!=record['resolution']['qualifications'][str(outer)]
            or any(r['resolved_family']!=r['cell']['family'] for r in p['cells'])):
            raise ValueError('prepared cost fold or TRAIN rows differ')
        cells=[dict(cell=r['cell'],role=r['role'],schedule=r['schedule']) for r in p['cells']]
        expected=[r for r in record['resolution']['cells'] if r['cell']['partition']==f'outer-{outer}']
        if sorted(cells,key=lambda r:r['cell']['id'])!=sorted(expected,key=lambda r:r['cell']['id']):
            raise ValueError('prepared cost model coverage differs')
        for name,pin in p['source_pins'].items():
            name=source_identity(name,record['source_root'])
            if all_pins.get(name,pin)!=pin:raise ValueError('inconsistent prepared cost input pins')
            all_pins[name]=pin
        if (w['schema']!='shaping-cost-v1' or w['raw_ids']!=record['resolution']['raw_ids']
            or len(w['arms'])!=16 or len(w['feature_bits'])!=40):
            raise ValueError('all sixteen arms and forty timing rows required')
        for arm in w['arms']:
            path=root/arm['model']
            if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
                or sha(path)!=arm['sha256']):raise ValueError('prepared cost model hash differs')
        from shaping_cost_complexity import validate_fold
        # Proof paths use the worker's original absolute data identities.
        validate_fold(root/'complexity',root/'WORKLOAD.json',cells,
            {name:pin for name,pin in p['source_pins'].items()})
    if any(all_pins.get(name)!=pin for name,pin in record['resolution']['input_sha256'].items()):
        raise ValueError('prepared cost input pins differ')


def pack(request,bundle):
    from shaping_terminal_transport import pack as owner
    return owner(request,bundle,validate_result)


def collect(request,bundle,pin):
    from shaping_terminal_transport import collect as owner
    return owner(request,bundle,pin,reproduce,validate_result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('pack','collect'))
    p.add_argument('--request',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--bundle-sha256');a=p.parse_args()
    print(json.dumps(pack(a.request,a.bundle) if a.mode=='pack' else collect(a.request,a.bundle,a.bundle_sha256)))
