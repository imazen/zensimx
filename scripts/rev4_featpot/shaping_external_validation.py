"""Score the registered fixed external models; training and selection stay closed."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['TMPDIR']=str(Path.home()/'tmp')
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shaping_completion import REPO,sha,encoded
from shaping_reduced_scope import admit_compute
from shaping_storage import reservation
from tree_study import RAW_IDS


def native(command,log):
    result=subprocess.run(list(map(str,command)),capture_output=True,text=True,check=True)
    log.write_text(result.stdout+'\n'+result.stderr)
    return result.stdout


def mapped(original,copies):
    value=copies[str(original)];path=Path(value['path'])
    if sha(path)!=value['sha256']:raise ValueError('external copy identity differs: '+str(path))
    return path


def score(root,study,binaries,tree_binary,copies,model,dataset):
    admit_compute(study)
    registration=json.loads((root/'REGISTRATION.json').read_text())
    bindings=json.loads((root/'BINDINGS-initial.json').read_text())
    if bindings['registration_sha256']!=sha(root/'REGISTRATION.json'):raise ValueError('external registration drift')
    matches=[row for row in bindings['models'] if row['cell']['id']==model]
    if len(matches)!=1:raise ValueError('one bound external model required')
    bound=matches[0];cell=bound['cell'];spec=registration['datasets'][dataset]
    if cell not in registration['cells'] or cell['setting']!=0 or cell['partition']!='inner-0-0':raise ValueError('external model selection drift')
    bake=mapped(bound['bake'],copies)
    if sha(bake)!=bound['bake_sha256'] or sha(mapped(bound['report'],copies))!=bound['report_sha256']:
        raise ValueError('external model or qualified report drift')
    features=mapped(spec['features'],copies);keys=mapped(spec['keys'],copies);targets=mapped(spec['targets'],copies)
    if sha(features)!=spec['features_sha256'] or sha(keys)!=spec['keys_sha256'] or sha(targets)!=spec['targets_sha256']:
        raise ValueError('external data drift')
    output=root/'results'/model/(dataset+'-native-basis' if cell['family']=='shaped_additive' else dataset)
    if output.exists():raise ValueError('preserve every external scoring attempt')
    with reservation(study,output,30*1024**3,bytes_required=64*1024**2):
        output.mkdir(parents=True)
        # The exact model and all targets were frozen in the published registration.
        # This is the first target-value parse in this worker.
        rows=json.loads(targets.read_text());keyrows=json.loads(keys.read_text())['rows']
        if len(rows)!=spec['rows'] or len(keyrows)!=len(rows):raise ValueError('external row coverage differs')
        if [row['a'] for row in rows]!=[row['pair_key'] for row in keyrows]:raise ValueError('external target/key order differs')
        table=pq.read_table(features)
        if table.num_rows!=len(rows):raise ValueError('external feature row coverage differs')
        if cell['family']=='shaped_additive':
            raw=np.column_stack([table[f'f{i}'].to_numpy() for i in RAW_IDS]).astype(np.float32)
            if not np.isfinite(raw).all():raise ValueError('nonfinite external linear input')
            np.savez_compressed(output/'raw.npz',x=raw)
            linear_binary=tree_binary.with_name('bake_dial_refit')
            native([linear_binary,'linear-predict','--bake',bake,'--inputs',output/'raw.npz','--out',output/'native.tsv'],output/'predict.log')
            with (output/'native.tsv').open() as stream:records=list(csv.DictReader(stream,delimiter='\t'))
            prediction=np.array([float(row['pred']) for row in records])
        elif bound['prediction_kind']=='tree':
            raw=np.column_stack([table[f'f{i}'].to_numpy() for i in RAW_IDS]).astype(np.float32).astype(np.float64)
            if not np.isfinite(raw).all():raise ValueError('nonfinite external tree input')
            np.savez_compressed(output/'raw.npz',test_X=raw)
            native([tree_binary,'--head',bake,'--parity',output/'raw.npz','--set','test','--predict-raw',output/'native.tsv'],output/'predict.log')
            with (output/'native.tsv').open() as stream:
                records=list(csv.DictReader(stream,delimiter='\t'))
            prediction=np.array([float(row['raw'])*100 for row in records])
        else:
            for name,value in [('human_score',np.zeros(len(rows))),('ref_basename',[row['group'] for row in rows])]:
                if name in table.column_names:table=table.drop([name])
                table=table.append_column(name,pa.array(value))
            view=output/'features.parquet';pq.write_table(table,view,compression='zstd')
            Path(str(view)+'.manifest.json').write_bytes(encoded(dict(formula_revision=7,feature_set_id='basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af',source_sha256=sha(features))))
            import v2_common,v2_lodo_mlp
            v2_common.FITBIN=v2_lodo_mlp.FITBIN=binaries/'bake_dial_refit'
            prediction=v2_lodo_mlp.predict(bake,view,output/'native.tsv')
        if len(prediction)!=len(rows) or not np.isfinite(prediction).all():raise ValueError('external prediction coverage/finite gate failed')
        from external_sets import CONTENT_OVERLAP
        subsets={'all':np.arange(len(rows))}
        if dataset=='live':
            clean=np.array([i for i,row in enumerate(rows) if row['group']+'.bmp' not in CONTENT_OVERLAP['live']])
            if len(clean)!=267 or len({rows[i]['group'] for i in clean})!=10:raise ValueError('fixed LIVE clean10 differs')
            subsets['clean10']=clean
        result=dict(schema='shaping-fixed-external-result-v1',cell=cell,dataset=dataset,bake_sha256=sha(bake),
            report_sha256=bound['report_sha256'],registration_sha256=sha(root/'REGISTRATION.json'),
            inference_binary_sha256=sha(tree_binary.with_name('bake_dial_refit') if cell['family']=='shaped_additive' else tree_binary if bound['prediction_kind']=='tree' else binaries/'bake_dial_refit'),
            driver_sha256=sha(__file__),
            panel_sha256=sha(binaries/'panel'),prediction_sha256=sha(output/'native.tsv'),populations={},
            selection_permitted=False,training_cells_added=0)
        for name,ix in subsets.items():
            target=np.array([rows[i]['target'] for i in ix],dtype=float);pred=prediction[ix];groups=[rows[i]['group'] for i in ix]
            if not np.isfinite(target).all():raise ValueError('nonfinite external target')
            pairs=output/(name+'.tsv');pairs.write_text('predicted\ttarget\n'+''.join(f'{p!r}\t{y!r}\n' for p,y in zip(pred.tolist(),target.tolist(),strict=True)))
            full=native([binaries/'panel','--input',pairs,'--json','--signed-quality'],output/(name+'-panel.log'))
            statistics=json.loads(full)
            if statistics['n']!=len(ix) or statistics['n_dropped']!=0:raise ValueError('external panel dropped rows')
            group_ids={group:i for i,group in enumerate(sorted(set(groups)))}
            rank_input=output/(name+'-rank-input.json')
            rank_input.write_bytes(encoded(dict(predicted=pred.tolist(),target=target.tolist(),reference=[group_ids[g] for g in groups])))
            interval=json.loads(native([binaries/'panel','rank-interval','--input',rank_input,'--resamples','1000','--seed','20261108'],output/(name+'-interval.log')))
            scatter=dict(indices=ix.tolist(),predicted=pred.tolist(),target=target.tolist(),source=groups,
                         distortion=[rows[i].get('codec',rows[i].get('distortion','unknown')) for i in ix])
            (output/(name+'-scatter.json')).write_bytes(encoded(scatter))
            result['populations'][name]=dict(rows=len(ix),sources=len(set(groups)),statistics=statistics,srocc_source_interval=interval,scatter=str(output/(name+'-scatter.json')))
        (output/'RESULT.json').write_bytes(encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','study','binaries','tree-binary','copies'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cell',required=True);p.add_argument('--dataset',choices=('csiq','live'),required=True)
    a=p.parse_args();print(json.dumps(score(a.root,a.study,a.binaries,a.tree_binary,json.loads(a.copies.read_text()),a.cell,a.dataset)),flush=True)
