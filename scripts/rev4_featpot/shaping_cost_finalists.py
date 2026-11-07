"""Admit all sixteen completed full-FIT models to the native cost instrument."""
from __future__ import annotations
import argparse,json,shutil,subprocess
from pathlib import Path
import numpy as np
from shaping_completion import ROLES,NEURAL,encoded,sha
from shaping_execution import load_registration
from shaping_full_fit_execution import refit_cell
from shaping_full_fit_selection import candidate
from shaping_cost_fixture import choose_rows,float_bits


def coverage(cells):
    selected=[c for c in cells if c['stage']=='full_fit']
    expected={(role,seed) for role in ROLES for seed in ((None,) if role=='shaped_additive' else (0,1,2))}
    if len(selected)!=16 or {(c['family'],c['seed']) for c in selected}!=expected:
        raise ValueError('all sixteen registered finalist roles and seeds required')
    return sorted(selected,key=lambda c:(c['family'],-1 if c['seed'] is None else c['seed']))


def prepare(root,binaries,output):
    from shaped_linear_study import OLD
    from tree_study import RAW_IDS
    if output.exists():raise ValueError('preserve existing finalist workload')
    plan,cells,partitions=load_registration(root);selected=coverage(cells)
    # Fix TRAIN row identities before opening any model prediction table.
    membership=partitions['full-fit'];train=membership['train']
    if membership['test'] or set(train)&set(membership['descriptive_development']):
        raise ValueError('finalist workload requires disjoint full-FIT training membership')
    for name,pin in plan['input_pins'].items():
        if sha(name)!=pin:raise ValueError('frozen finalist input drift')
    pop=np.load(OLD/'data/pop.npy',mmap_mode='r',allow_pickle=False)
    chosen=choose_rows(train,pop)
    raw=np.load(OLD/'data/raw.npy',mmap_mode='r',allow_pickle=False)
    features=np.asarray(raw[chosen],dtype='<f4')
    if not np.isfinite(features).all():raise ValueError('nonfinite finalist feature row')
    build_path=binaries/'BUILD.json';build=json.loads(build_path.read_text());converter=binaries/'bake_dial_refit'
    if sha(converter)!=build['binary_sha256']['bake_dial_refit']:raise ValueError('converter binary drift')
    pins={str(root/'PLAN.json'):sha(root/'PLAN.json'),str(build_path):sha(build_path),str(converter):sha(converter)}
    expected_ids=sorted(set(train)|set(membership['descriptive_development']))
    arms=[];models={};dense={};identities=[];admission_pin=None
    for cell in selected:
        _,_,_,admission,path,registered,family,schedule=refit_cell(root,cell['id'])
        if registered!=cell:raise ValueError('finalist cell differs')
        if admission_pin is not None and admission_pin!=sha(path):raise ValueError('mixed finalist admissions')
        admission_pin=sha(path);pins[str(path)]=admission_pin
        row,classification,evidence=candidate(root,cell,family,admission,path);pins.update(evidence)
        report_paths=[Path(p) for p in evidence if Path(p).name=='REPORT.json']
        # Diagnostic inputs also include reports; choose the terminal descriptive
        # report by complete identity, never a convenient training checkpoint.
        report_paths=[p for p in report_paths if json.loads(p.read_text()).get('selection_scope')=='descriptive_only'
                      and json.loads(p.read_text()).get('cell')==cell]
        if len(report_paths)!=1:raise ValueError('one descriptive final report required')
        panel=report_paths[0].parent;predictions=panel/'predictions.npz'
        if row.get('predictions_sha256')!=sha(predictions):raise ValueError('final predictions are not pinned')
        with np.load(predictions,allow_pickle=False) as table:
            if set(table.files)!={'original_indices','native'}:raise ValueError('prediction fields differ')
            ids=table['original_indices'];values=table['native']
            if ids.dtype.kind not in 'iu' or ids.ndim!=1 or not np.array_equal(ids,expected_ids) or values.shape!=ids.shape or not np.isfinite(values).all():
                raise ValueError('final prediction membership or values differ')
            lookup=dict(zip(ids.tolist(),values.tolist()))
        pins[str(predictions)]=sha(predictions)
        run=root/'runs'/cell['id']/panel.parent.name
        if family in NEURAL:
            original=run/'fit/checkpoints'/f"ckpt_epoch{schedule['phase_epochs'][-1]-1:03}.bin"
            model=Path(json.loads(Path(row['reexport_receipt']).read_text())['bake']) if 'reexport_receipt' in row else original
            model_pin=row['checkpoint_sha256']
        else:
            model=run/('model.opaque-vector.zcth' if family=='extra_trees' else 'native/model.bin')
            q=row['qualification'];model_pin=q['native_tree_gate']['model_sha256'] if family=='extra_trees' else q['model_sha256']
        if sha(model)!=model_pin:raise ValueError('final native model differs')
        name=cell['family']+'-'+('deterministic' if cell['seed'] is None else str(cell['seed']))
        models[name]=model;pins[str(model)]=model_pin
        if family in NEURAL:
            dense[name]=panel/'dense'/(model_pin[:16]+'.bin');pins[str(dense[name])]=sha(dense[name])
        arms.append(dict(id=name,kind='tree' if family=='extra_trees' else 'shaped_additive' if family=='shaped_additive' else 'bake',
            model=str(Path('models')/(name+'.bin')),sha256=model_pin,expected_score_bits=[float_bits(lookup[i]) for i in chosen]))
        identities.append(dict(cell=cell,resolved_family=family,schedule=schedule,training_classification=classification))
    output.mkdir(parents=True);(output/'models').mkdir()
    for arm in arms:
        dest=output/arm['model'];source=models[arm['id']]
        if arm['id'] in dense:
            log=output/('densify-'+arm['id']+'.log')
            result=subprocess.run([str(converter),'densify','--in',str(source),'--out',str(dest)],capture_output=True,text=True)
            log.write_text(result.stdout+'\n'+result.stderr)
            if result.returncode or 'BIT-IDENTICAL' not in log.read_text():raise ValueError('final dense-input gate failed')
            expected=sha(dense[arm['id']])
            if sha(dest)!=expected:raise ValueError('evaluated final dense representation differs')
            arm['sha256']=expected
        else:
            shutil.copyfile(source,dest)
            if sha(dest)!=arm['sha256']:raise ValueError('copied finalist model differs')
    for name,pin in pins.items():
        if sha(name)!=pin:raise ValueError('finalist evidence changed during preparation')
    workload=dict(schema='shaping-cost-v1',admission_sha256=admission_pin,raw_ids=RAW_IDS,feature_bits=features.view('<u4').tolist(),arms=arms)
    (output/'WORKLOAD.json').write_bytes(encoded(workload))
    (output/'PROVENANCE.json').write_bytes(encoded(dict(scope='all sixteen registered full-FIT finalists; TRAIN-only timing rows',
        cells=identities,original_training_indices=chosen,source_pins=pins,workload_sha256=sha(output/'WORKLOAD.json'),
        native_cost_instrument_qualification_required=True,timing_measured=False)))
    return dict(output=str(output),models=len(arms),rows=len(chosen))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','binary-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.study,a.binary_dir,a.output)))
