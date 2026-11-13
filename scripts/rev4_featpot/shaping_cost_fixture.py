"""Prepare fixed inner models for instrument qualification, never finalist timing."""
from __future__ import annotations
import argparse,csv,json,struct,shutil,subprocess
from pathlib import Path
import numpy as np
from shaping_completion import encoded,sha
from shaping_execution import load_registration

FAMILIES=('relu','softplus','silu_monotone','silu_diagnostic',
          'product_joint_identity','separate_joint_identity','shaped_additive','extra_trees')


def choose_rows(train,populations):
    groups=sorted({str(populations[i]) for i in train})
    if len(groups)!=5:raise ValueError('five admitted populations required')
    chosen=[]
    for group in groups:
        rows=sorted(i for i in train if str(populations[i])==group)
        if len(rows)<8:raise ValueError('eight training rows per population required')
        chosen.extend(rows[:8])
    return chosen


def float_bits(value):return struct.unpack('<Q',struct.pack('<d',float(value)))[0]


def native_training_predictions(panel, train, groups):
    """Read round-tripping native text without a dataframe decimal conversion."""
    training=set(train);values={};pins={};offset=0
    if len(training)!=len(train):raise ValueError('distinct native training rows required')
    for group in groups:
        ids=[i for i in range(offset,offset+group['rows']) if i in training]
        offset+=group['rows']
        if not ids:continue
        path=panel/(group['name']+'-train.tsv')
        with path.open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
        if len(rows)!=len(ids) or [int(row['row_idx']) for row in rows]!=list(range(len(ids))):
            raise ValueError('native training row order differs')
        predictions=[float(row['pred']) for row in rows]
        if not np.isfinite(predictions).all():raise ValueError('finite native predictions required')
        values.update(zip(ids,predictions));pins[str(path)]=sha(path)
    if set(values)!=training:raise ValueError('complete native training prediction coverage required')
    return values,pins


def archived_control(study,family,paths):
    """Original inner controls bind predictions through their verified batch archive."""
    archive=study.parent/('tree-inner-archive' if family=='extra_trees' else 'linear-inner-archive')
    manifest=archive/'SHA256SUMS';receipt=archive/'MIRROR_VERIFIED.json'
    saved=json.loads(receipt.read_text());destination=Path(saved['destination'])
    if not saved['all_artifact_sha256_verified'] or sha(manifest)!=sha(destination/'SHA256SUMS'):
        raise ValueError('control archive is not verified')
    hashes=dict((name,pin) for pin,name in (line.split('  ',1) for line in manifest.read_text().splitlines()))
    pins={str(manifest):sha(manifest),str(receipt):sha(receipt)}
    for path in paths:
        relative=str(path.relative_to(study));pin=hashes[relative]
        if sha(path)!=pin or sha(destination/relative)!=pin:raise ValueError('control archive content differs')
        pins[str(path)]=pin;pins[str(destination/relative)]=pin
    return pins


def prepare(study,output,binaries):
    from shaped_linear_study import OLD,PREP
    from tree_study import RAW_IDS
    if output.exists():raise ValueError('preserve existing qualification fixture')
    plan,cells,partitions=load_registration(study);plan_pin=sha(study/'PLAN.json')
    build=json.loads((binaries/'BUILD.json').read_text());converter=binaries/'bake_dial_refit'
    if sha(converter)!=build['binary_sha256']['bake_dial_refit']:raise ValueError('converter binary differs')
    for name,pin in plan['input_pins'].items():
        if sha(name)!=pin:raise ValueError('frozen input differs')
    membership=partitions['inner-0-0'];train=membership['train']
    pop=np.load(OLD/'data/pop.npy',mmap_mode='r',allow_pickle=False)
    chosen=choose_rows(train,pop)
    raw=np.load(OLD/'data/raw.npy',mmap_mode='r',allow_pickle=False)
    features=np.asarray(raw[chosen],dtype='<f4')
    if not np.isfinite(features).all():raise ValueError('nonfinite feature fixture')
    arms=[];pins={str(study/'PLAN.json'):plan_pin,str(binaries/'BUILD.json'):sha(binaries/'BUILD.json'),str(converter):sha(converter)};identities=[];dense={}
    for family in FAMILIES:
        matched=[c for c in cells if c['stage']=='inner' and c['partition']=='inner-0-0'
                 and c['setting']==0 and c['seed']==(None if family=='shaped_additive' else 0) and c['family']==family]
        if len(matched)!=1:raise ValueError('fixed qualification cell differs')
        cell=matched[0];run=study/'runs'/cell['id']/'attempt-001'
        identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
        panel=run/'evaluation';report_path=panel/'REPORT.json';report=json.loads(report_path.read_text())
        marker=json.loads((panel/'COMPLETE.json').read_text())
        if identity['cell']!=cell or result['cell']!=cell or result['exit_code']!=0:
            raise ValueError('qualification fit differs')
        if identity['plan_sha256']!=plan_pin or report['plan_sha256']!=plan_pin or report['cell']!=cell or report['outer_results_read']:
            raise ValueError('qualification is restricted to fixed inner evidence')
        if marker['report_sha256']!=sha(report_path) or (panel/'FAILED_GATE.json').exists():
            raise ValueError('qualification native report is not complete')
        for name in ('RUN.json','RESULT.json','evaluation/REPORT.json','evaluation/COMPLETE.json'):
            pins[str(run/name)]=sha(run/name)
        if family in ('extra_trees','shaped_additive'):
            qualification=report['qualification'];model=run/('model.opaque-vector.zcth' if family=='extra_trees' else 'native/model.bin')
            model_pin=qualification['native_tree_gate']['model_sha256'] if family=='extra_trees' else qualification['model_sha256']
            predictions=panel/'predictions.npz'
            pins.update(archived_control(study,family,[run/'RUN.json',run/'RESULT.json',report_path,
                                                       panel/'COMPLETE.json',predictions,model]))
            with np.load(predictions,allow_pickle=False) as table:
                if not np.array_equal(table['original_indices'],sorted(set(train)|set(membership['test']))):
                    raise ValueError('control prediction membership differs')
                values=dict(zip(table['original_indices'].tolist(),table['native'].tolist()))
            pins[str(predictions)]=sha(predictions)
        else:
            model=run/identity.get('fit_directory','.')/'checkpoints'/f"ckpt_epoch{result['epochs']-1:03}.bin"
            model_pin=result['bake_sha256']
            if report['checkpoint_sha256']!=model_pin:raise ValueError('native checkpoint differs')
            if identity['build_sha256']!=sha(binaries/'BUILD.json'):raise ValueError('original evaluation build differs')
            dense[family]=panel/'dense'/(model_pin[:16]+'.bin')
            pins[str(dense[family])]=sha(dense[family])
            # Rust debug-formatted f64 values round trip; avoid pandas' default
            # decimal parser when constructing a bit-exact instrument oracle.
            values={};offset=0;training=set(train)
            for group in json.loads(PREP.read_text())['groups']:
                ids=[i for i in range(offset,offset+group['rows']) if i in training];offset+=group['rows']
                if not ids:continue
                path=panel/(group['name']+'-train.tsv')
                with path.open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
                if len(rows)!=len(ids) or [int(row['row_idx']) for row in rows]!=list(range(len(ids))):
                    raise ValueError('native training row order differs')
                values.update({i:float(row['pred']) for i,row in zip(ids,rows)})
                pins[str(path)]=sha(path)
        if sha(model)!=model_pin:raise ValueError('qualification model hash differs')
        pins[str(model)]=model_pin
        arms.append(dict(id=family,kind='tree' if family=='extra_trees' else 'shaped_additive' if family=='shaped_additive' else 'bake',model=str(model),
                         sha256=model_pin,expected_score_bits=[float_bits(values[i]) for i in chosen]))
        identities.append(cell)
    output.mkdir(parents=True)
    (output/'models').mkdir()
    for arm in arms:
        source=Path(arm['model']);relative=Path('models')/(arm['id']+'.bin')
        if arm['id'] in dense:
            log=output/('densify-'+arm['id']+'.log')
            result=subprocess.run([str(converter),'densify','--in',str(source),'--out',str(output/relative)],capture_output=True,text=True)
            log.write_text(result.stdout+'\n'+result.stderr)
            if result.returncode or 'BIT-IDENTICAL' not in log.read_text():raise ValueError('native dense-input gate failed')
            expected=sha(dense[arm['id']])
            if sha(output/relative)!=expected:raise ValueError('evaluated dense representation differs')
            arm['sha256']=expected
        else:
            shutil.copyfile(source,output/relative)
            if sha(output/relative)!=arm['sha256']:raise ValueError('copied fixture model differs')
        arm['model']=str(relative)
    workload=dict(schema='shaping-cost-v1',admission_sha256=plan_pin,raw_ids=RAW_IDS,
                  feature_bits=features.view('<u4').tolist(),arms=arms)
    (output/'WORKLOAD.json').write_bytes(encoded(workload))
    (output/'PROVENANCE.json').write_bytes(encoded(dict(scope='instrument qualification only; no finalist or timing selection',
        cells=identities,original_training_indices=chosen,source_pins=pins,workload_sha256=sha(output/'WORKLOAD.json'))))
    return dict(output=str(output),models=len(arms),rows=len(chosen))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path,required=True)
    args=p.parse_args();print(json.dumps(prepare(args.study,args.output,args.binary_dir)))
