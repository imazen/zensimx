"""One fixed-final development evaluation per new fit, through native consumers."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TMPDIR']=str(Path.home()/'tmp')

import numpy as np
from scipy.stats import rankdata
from shaping_completion import encoded, sha


def write(path, value):
    path.write_bytes(encoded(value))


def predict(bake, table, output, binaries):
    import v2_common, v2_lodo_mlp
    v2_common.FITBIN=v2_lodo_mlp.FITBIN=binaries/'bake_dial_refit'
    v2_lodo_mlp.predict(bake,table,output)
    # Preserve native decimal parsing, including f64 values near a tie.
    with output.open() as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    if [int(r['row_idx']) for r in rows]!=list(range(len(rows))):
        raise ValueError('native prediction order differs')
    values=np.array([float(r['pred']) for r in rows])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('native predictions empty or nonfinite')
    return values


def export(original, command, output, binaries):
    from shaping_reexport import registered_export_pairs,convert_export
    inspection=subprocess.run([str(binaries/'zenpredict-inspect'),str(original),'--weights'],capture_output=True,text=True,check=True)
    (output/'INSPECTION.json').write_text(inspection.stdout)
    model=json.loads(inspection.stdout)
    pairs=registered_export_pairs(model,command)
    declaration=output/'pairs.txt';declaration.write_text('relu-max-pairs-v1\n'+''.join(f'{a} {b}\n' for a,b in pairs))
    bake=output/'consumer.bin'
    conversion=convert_export(original,bake,declaration,pairs,binaries/'bake_dial_refit',output/'export.log')
    return bake,dict(original_sha256=sha(original),consumer_sha256=sha(bake),pairs=pairs,conversion=conversion)


def tails(pred, target):
    n=len(pred);err=np.abs((rankdata(pred)-rankdata(target))/n)
    return dict(rank_mae=float(err.mean()),rank_p95=float(np.quantile(err,.95)),rank_p99=float(np.quantile(err,.99)))


def replay_batch(path,jobs):
    """Standalone input to the bundled Rust panel; no Python needed to rescore."""
    labels={}
    with path.open('x') as f:
        for i,(label,pred,target) in enumerate(jobs):
            token='panel-'+str(i);labels[token]=label
            f.write(token+'\t'+','.join(repr(float(v)) for v in pred)+'\t'+','.join(repr(float(v)) for v in target)+'\n')
    write(path.with_suffix('.labels.json'),labels)


def feature_probe(study,bake,output,binaries):
    probes=json.loads((study/'evaluation_views/outer-0/PROBES.json').read_text())
    if sha(probes['path'])!=probes['sha256']:raise ValueError('feature probe drift')
    p=predict(bake,Path(probes['path']),output/'probes.tsv',binaries);delta=p[128:256]-p[:128]
    gate=dict(finite=bool(np.isfinite(p).all()),identity=float(p[256]),
        damage_increase_violations=int(np.sum(delta>1e-5)),maximum_damage_increase=float(max(0.,delta.max())),
        ceiling_violations=int(np.sum(p>100.)))
    if not gate['finite'] or gate['identity']!=100. or gate['damage_increase_violations'] or gate['ceiling_violations']:
        raise ValueError('consumer feature contract failed')
    return gate


def human(study, run, output, binaries):
    from shaped_linear_study import OLD, metrics
    terminal=json.loads((run/'RESULT.json').read_text());job=terminal['job']
    if terminal['stage']!='trained' or job['epochs']!=240:
        raise ValueError('fixed complete human fit required')
    original=run/'checkpoints/ckpt_epoch239.bin'
    if sha(original)!=terminal['final_bake_sha256']:
        raise ValueError('final checkpoint drift')
    if output.exists():
        raise ValueError('preserve existing evaluation')
    output.mkdir(parents=True)
    bake,conversion=export(original,job['argv'],output,binaries)
    gate=feature_probe(study,bake,output,binaries)
    plan=json.loads((study/'PLAN.json').read_text());part=json.loads((study/'PARTITIONS.json').read_text())['outer-0']
    indices=np.array(part['test']);lookup={int(i):j for j,i in enumerate(indices)}
    validation=json.loads((study/'evaluation_views/outer-0/RECEIPT.json').read_text())
    pred=np.full(len(indices),np.nan)
    for g in validation['groups']:
        for p,pin in g['file_sha256'].items():
            if sha(p)!=pin:raise ValueError('frozen development input drift')
        values=predict(bake,Path(g['path']),output/(g['name']+'.tsv'),binaries)
        if len(values)!=len(g['indices']):raise ValueError('prediction coverage differs')
        pred[[lookup[i] for i in g['indices']]]=values
    if not np.isfinite(pred).all():raise ValueError('incomplete development predictions')
    baseline=study/'reexport_max_pairs'/job['baseline_cell']['id']/'attempt-001/final/REPORT.json'
    control=json.loads(baseline.read_text())
    original_root=Path(plan['original_plan']).parent
    d={n:np.load(OLD/'data'/f'{n}.npy',allow_pickle=False)[indices] for n in ('pop','y')}
    d.update({n:np.load(original_root/'data'/f'{n}.npy',allow_pickle=False)[indices] for n in ('component','family')})
    os.environ['ZEN_PANEL_BIN']=str(binaries/'panel')
    panels=metrics(pred,d['y'],np.arange(len(indices)),d,control['maps'])
    jobs=[]
    for pop in sorted(set(d['pop'])):
        ids=np.flatnonzero(d['pop']==pop);jobs.append((str(pop)+'|pooled',pred[ids],d['y'][ids]))
        for component in sorted(set(d['component'][ids])):
            q=ids[d['component'][ids]==component]
            jobs.append((str(pop)+'|source|'+str(component),pred[q],d['y'][q]))
    replay_batch(output/'replay-batch.tsv',jobs)
    rank_tails={str(pop):tails(pred[d['pop']==pop],d['y'][d['pop']==pop]) for pop in sorted(set(d['pop']))}
    diagnostics=[]
    for line in (run/'train.log').read_text().splitlines():
        if line.startswith('TRAINING_DIAGNOSTICS '):diagnostics.append(json.loads(line.split(' ',1)[1]))
    result=dict(stage='complete',id=job['id'],seed=job['seed'],arm=job['arm'],recipe='human_ssim2',
        scope='Single already-exposed development fold; no fresh confirmation.',final_holdout_read=False,
        checkpoint_epoch=239,conversion=conversion,consumer_gate=gate,metrics=panels,rank_tails=rank_tails,
        matched_control=control['outer_test'],baseline_report_sha256=sha(baseline),
        final_training_diagnostics=diagnostics[-1] if diagnostics else None,
        result_sha256=sha(run/'RESULT.json'),binary_build_sha256=sha(binaries/'BUILD.json'))
    np.savez_compressed(output/'predictions.npz',original_indices=indices,native=pred)
    result['predictions_sha256']=sha(output/'predictions.npz')
    write(output/'RESULT.json',result)
    print(json.dumps(dict(id=job['id'],stage='complete',human_composite=panels['selection_score'])),flush=True)


def a_recipe(run, inputs, output, binaries, study):
    from external_sets import CONTENT_OVERLAP
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from lib.zen_stats import panel_batch
    terminal=json.loads((run/'RUN.json').read_text())
    if terminal['stage']!='trained' or len(terminal['steps'])!=2 or any(s['exit_code'] for s in terminal['steps']):
        raise ValueError('two complete frozen A phases required')
    bake=run/'1-continuation/writes/refit/last.bin'
    if sha(bake)!=terminal['steps'][-1]['final_bake_sha256']:raise ValueError('A final model drift')
    if output.exists():raise ValueError('preserve existing evaluation')
    output.mkdir(parents=True)
    gate=feature_probe(study,bake,output,binaries)
    plan=json.loads((inputs/'EVALUATION_REGISTRATION.json').read_text());datasets={}
    os.environ['ZEN_PANEL_BIN']=str(binaries/'panel')
    for dataset,spec in plan['datasets'].items():
        dest=output/dataset;dest.mkdir()
        for name,item in spec['files'].items():
            if sha(inputs/'eval-inputs'/name)!=item['sha256']:raise ValueError('A exposed view drift')
        rows=json.loads((inputs/'eval-inputs'/(dataset+'-targets.json')).read_text())
        values=predict(bake,inputs/'eval-inputs'/(dataset+'.parquet'),dest/'native.tsv',binaries)
        if len(values)!=len(rows):raise ValueError('A prediction coverage differs')
        subsets={'all':np.arange(len(rows))}
        if dataset=='live':
            clean=np.array([i for i,r in enumerate(rows) if r['group']+'.bmp' not in CONTENT_OVERLAP['live']])
            if len(clean)!=267 or len({rows[i]['group'] for i in clean})!=10:raise ValueError('LIVE clean10 differs')
            subsets['clean10']=clean
        stats={}
        for name,ids in subsets.items():
            target=np.array([rows[i]['target'] for i in ids]);pred=values[ids]
            panel,=panel_batch([(name,pred,target)],stats='srocc',raw_errors=True)
            if panel['n_dropped']:raise ValueError('canonical A panel dropped rows')
            stats[name]=dict(rows=len(ids),statistics=panel,**tails(pred,target))
        replay_batch(dest/'replay-batch.tsv',[(name,values[ids],[rows[i]['target'] for i in ids]) for name,ids in subsets.items()])
        write(dest/'SCATTER.json',dict(predicted=values.tolist(),target=[r['target'] for r in rows],source=[r['group'] for r in rows]))
        datasets[dataset]=stats
    result=dict(stage='complete',id=run.name,arm=terminal['arm'],seed=terminal['seed'],recipe='A',
        scope='Previously exposed A evaluation views; descriptive only.',final_holdout_read=False,
        consumer_gate=gate,
        checkpoint='fixed continuation epoch 59 f32',bake_sha256=sha(bake),datasets=datasets,
        training_run_sha256=sha(run/'RUN.json'),binary_build_sha256=sha(binaries/'BUILD.json'))
    write(output/'RESULT.json',result)
    print(json.dumps(dict(id=run.name,stage='complete',datasets=datasets)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=('human','a','qualify-export'))
    for name in ('run','output','binaries'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--study',type=Path);p.add_argument('--a-inputs',type=Path);a=p.parse_args()
    if a.kind=='human':human(a.study,a.run,a.output,a.binaries)
    elif a.kind=='a':a_recipe(a.run,a.a_inputs,a.output,a.binaries,a.study)
    else:
        job=json.loads((a.run/'RUN.json').read_text())['job'];a.output.mkdir(parents=True)
        checkpoint=a.run/'checkpoints/ckpt_epoch000.bin'
        bake,conversion=export(checkpoint,job['argv'],a.output,a.binaries)
        gate=feature_probe(a.study,bake,a.output,a.binaries)
        write(a.output/'QUALIFIED.json',dict(stage='pass',scope='Early checkpoint consumer-format and feature probes only; no target or evaluation labels.',conversion=conversion,gate=gate))
