"""Source bootstrap, duplicate diagnostics and evidence reduction; no fit selection."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shaped_linear_study import (load,write,sha,clean,rms,weights_and_targets,FAMILIES,POSITIVE,REPO,OLD,META,PREP,BASE)
from lib.zen_stats import panel_batch_indexed,scatter


def audit(root):
    plan,d=load(root);out=root/'audit';out.mkdir(exist_ok=True)
    fit=np.flatnonzero(~d['dev']);w,target,maps=weights_and_targets(d['y'][fit],d['pop'][fit],d['component'][fit],'range',1.)
    x=np.asarray(d['raw']);scale=rms(x[fit],w);z=x.astype(np.float64)/scale
    hashes=[hashlib.sha256(row.tobytes()).hexdigest() for row in x]
    groups=defaultdict(list)
    for i,h in enumerate(hashes):groups[h].append(i)
    duplicates=[]
    for h,ids in groups.items():
        if len(ids)>1:
            duplicates.append(dict(feature_sha256=h,ids=ids,populations=d['pop'][ids].tolist(),components=d['component'][ids].tolist(),development=d['dev'][ids].tolist(),labels=d['y'][ids].tolist()))
    norm=np.einsum('ij,ij->i',z,z);nearest=[]
    for start in range(0,len(z),192):
        ids=np.arange(start,min(start+192,len(z)));distance=norm[ids,None]+norm[None,:]-2*z[ids]@z.T;distance=np.maximum(distance,0)
        for k,i in enumerate(ids):
            candidates=(d['component']!=d['component'][i]);crossrole=d['dev']!=d['dev'][i]
            for kind,mask in (('other_source',candidates),('other_role',candidates&crossrole)):
                if not mask.any():raise ValueError('nearest diagnostic has no candidates')
                j=int(np.argmin(np.where(mask,distance[k],np.inf)))
                p=str(d['pop'][i]);q=str(d['pop'][j]);mi,mj=maps[p],maps[q]
                ti=(d['y'][i]-mi['values'][0])/(mi['values'][-1]-mi['values'][0]);tj=(d['y'][j]-mj['values'][0])/(mj['values'][-1]-mj['values'][0])
                nearest.append(dict(row=int(i),neighbor=j,kind=kind,distance=float(np.sqrt(distance[k,j])),population=p,neighbor_population=q,component=str(d['component'][i]),neighbor_component=str(d['component'][j]),development=bool(d['dev'][i]),neighbor_development=bool(d['dev'][j]),mapped_label_difference=float(abs(ti-tj)),same_family=bool(d['family'][i]==d['family'][j])))
    pq.write_table(pa.Table.from_pylist(nearest),out/'nearest.parquet',compression='zstd')
    write(out/'DUPLICATES.json',dict(exact_feature_groups=duplicates,groups=len(duplicates),rows=sum(len(v['ids']) for v in duplicates),cross_role_groups=sum(len(set(v['development']))>1 for v in duplicates),cross_component_groups=sum(len(set(v['components']))>1 for v in duplicates),automatic_changes=0))
    ordered=sorted(nearest,key=lambda v:(v['distance'],-v['mapped_label_difference']))
    write(out/'NEAREST.json',dict(metric='Euclidean raw features / FIT-only source-balanced RMS, without centering',scale=scale.tolist(),nearest_sha256=sha(out/'nearest.parquet'),nearest_examples=ordered[:40],largest_label_disagreements=sorted(nearest,key=lambda v:-v['mapped_label_difference'])[:40],interpretation='candidate diagnostics; different mapped population scales are not calibrated perceptual units',automatic_changes=0))
    print(json.dumps(dict(stage='audit',duplicate_groups=len(duplicates),nearest_records=len(nearest))),flush=True)


def selected_path(root,family,outer):
    name=f's1-outer-{family}-o{outer}' if outer is not None else f's1-final-{family}'
    return Path(json.loads((root/'fits'/name/'COMPLETE.json').read_text())['results'][0]['path'])


def bootstrap(root):
    plan,d=load(root);os.environ['ZEN_PANEL_BIN']=str(root/'bin/panel');out=root/'bootstrap';out.mkdir(exist_ok=True)
    names=list(FAMILIES)+[f'{kind}-s{seed}' for kind in ('extra','neural') for seed in plan['seeds']]
    pops=sorted(set(d['pop'][~d['dev']]))
    draws=2000;samples=np.empty((len(names),len(pops),3,draws));pooled_samples=np.empty_like(samples);within_samples=np.empty_like(samples);pins={};points=np.empty((len(names),len(pops),3))
    for o in range(3):
        ids=np.flatnonzero((~d['dev'])&(d['fold']==o));predictions={};reports={}
        for family in FAMILIES:
            path=selected_path(root,family,o);predictions[family]=np.load(path/'predictions.npz')['native'][:len(d['y'])];reports[family]=json.loads((path/'REPORT.json').read_text())['test'];pins[str(path/'predictions.npz')]=sha(path/'predictions.npz')
        for kind in ('extra','neural'):
            for seed in plan['seeds']:
                name=f'{kind}-s{seed}';path=root/'comparators'/f'{kind}-o{o}-true-s{seed}'
                predictions[name]=np.load(path/'predictions.npz')['native'];reports[name]=json.loads((path/'REPORT.json').read_text())['test'];pins[str(path/'predictions.npz')]=sha(path/'predictions.npz')
        for pi,pop in enumerate(pops):
            q=ids[d['pop'][ids]==pop];refs=d['component'][q];sources=sorted(set(refs));groups=[np.flatnonzero(refs==s) for s in sources]
            bases={f'p{k}':predictions[name][q] for k,name in enumerate(names)};bases['y']=d['y'][q]
            native_sources=panel_batch_indexed(bases,[(f'p{k}|{i}',f'p{k}','y',g) for k in range(len(names)) for i,g in enumerate(groups)],stats='srocc')
            source_score=np.array([r['srocc_signed'] for r in native_sources]).reshape(len(names),len(groups))
            if not np.isfinite(source_score).all():raise ValueError('undefined source bootstrap correlation')
            rng=np.random.default_rng(20261106);resamples=rng.integers(len(groups),size=(draws,len(groups)))
            for start in range(0,draws,32):
                jobs=[]
                for b in range(start,min(start+32,draws)):
                    indices=np.concatenate([groups[i] for i in resamples[b]])
                    jobs.extend((f'{k}|{b}',f'p{k}','y',indices) for k in range(len(names)))
                rows=panel_batch_indexed(bases,jobs,stats='srocc')
                for r in rows:
                    if r['n_dropped'] or not np.isfinite(r['srocc_signed']):raise ValueError('bootstrap dropped or undefined')
                    k,b=map(int,r['label'].split('|'));pooled_samples[k,pi,o,b]=r['srocc_signed']
            for k,name in enumerate(names):
                within_samples[k,pi,o]=source_score[k,resamples].mean(axis=1)
                samples[k,pi,o]=(pooled_samples[k,pi,o]+within_samples[k,pi,o])/2
                v=reports[name]['summary'][pop];points[k,pi,o]=(v['pooled']+v['within'])/2
            print(json.dumps(dict(stage='bootstrap',outer=o,population=pop,draws=draws,models=len(names))),flush=True)
    np.savez_compressed(out/'samples.npz',composite=samples,pooled=pooled_samples,within=within_samples,point=points)
    baseline=names.index('positive_products');humans=[i for i,p in enumerate(pops) if p!='cid22_ssim2'];intervals={}
    for k,name in enumerate(names):
        by={}
        for pi,p in enumerate(pops):
            delta=samples[k,pi].mean(axis=0)-samples[baseline,pi].mean(axis=0)
            by[p]=dict(point=float(points[k,pi].mean()-points[baseline,pi].mean()),low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)))
        agg=samples[k,humans].mean(axis=(0,1));control=samples[baseline,humans].mean(axis=(0,1));delta=agg-control
        intervals[name]=dict(populations=by,equal_human_composite=dict(point=float(points[k,humans].mean()),low=float(np.quantile(agg,.025)),high=float(np.quantile(agg,.975)),difference_from_positive_products=float(points[k,humans].mean()-points[baseline,humans].mean()),difference_low=float(np.quantile(delta,.025)),difference_high=float(np.quantile(delta,.975))))
    # Mean seed performance, not averaged predictions, is the matched comparator.
    for kind in ('extra','neural'):
        selected=[k for k,name in enumerate(names) if name.startswith(kind+'-s')]
        sm=samples[selected].mean(axis=0);pt=points[selected].mean(axis=0);by={}
        for pi,p in enumerate(pops):
            delta=sm[pi].mean(axis=0)-samples[baseline,pi].mean(axis=0)
            by[p]=dict(point=float(pt[pi].mean()-points[baseline,pi].mean()),low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)))
        agg=sm[humans].mean(axis=(0,1));delta=agg-samples[baseline,humans].mean(axis=(0,1))
        intervals[kind+'-mean-seeds']=dict(populations=by,equal_human_composite=dict(point=float(pt[humans].mean()),low=float(np.quantile(agg,.025)),high=float(np.quantile(agg,.975)),difference_from_positive_products=float(pt[humans].mean()-points[baseline,humans].mean()),difference_low=float(np.quantile(delta,.025)),difference_high=float(np.quantile(delta,.975))))
    write(out/'SOURCE_BOOTSTRAP.json',dict(draws=draws,seed=20261106,names=names,populations=pops,baseline=names[baseline],intervals=intervals,predictions_sha256=pins,scope='paired whole-source resampling within each population and outer fold; average fold metrics; conditional on these fits and exposed sources, no multiplicity correction'))


def reduce(root):
    plan,d=load(root);os.environ['ZEN_PANEL_BIN']=str(root/'bin/panel');out=root/'report';out.mkdir(exist_ok=True)
    records={};rows=[]
    for complete in sorted((root/'fits').glob('*/COMPLETE.json')):
        task=json.loads((complete.parent/'TASK.json').read_text())
        for cell in json.loads(complete.read_text())['results']:
            path=Path(cell['path']);r=json.loads((path/'REPORT.json').read_text());records[str(path.relative_to(root))]=r
            for role in ('train','test'):
                for p,v in r[role]['summary'].items():rows.append(dict(task=task['id'],family=task['family'],lambda_=cell['lambda_'],role=role,population=p,**v))
    expected=155+8+60+60+48+3+len(json.loads((root/'S5_influence.json').read_text()))
    if len(records)!=expected:raise ValueError(f'incomplete quadratic inventory {len(records)}/{expected}')
    comparators={}
    for cell in json.loads((root/'S4-comparison.json').read_text()):
        path=root/'comparators'/cell['id'];complete=json.loads((path/'COMPLETE.json').read_text())
        if complete['report_sha256']!=sha(path/'REPORT.json'):raise ValueError('comparator report changed')
        comparators[cell['id']]=json.loads((path/'REPORT.json').read_text())
    pq.write_table(pa.Table.from_pylist(rows),out/'metrics.parquet',compression='zstd')
    certificates=[v['certificate'] for v in records.values()]
    main={};scatters={}
    for family in FAMILIES:
        main[family]={}
        for o in range(4):
            path=selected_path(root,family,o if o<3 else None);r=json.loads((path/'REPORT.json').read_text())
            main[family][str(o) if o<3 else 'development']=r
            pred=np.load(path/'predictions.npz')['native'][:len(d['y'])]
            ids=np.flatnonzero((~d['dev'])&(d['fold']==o)) if o<3 else np.flatnonzero(d['dev'])
            for p in sorted(set(d['pop'][ids])):
                q=ids[d['pop'][ids]==p];scatters[f'{family}/{o}/{p}']=scatter(pred[q],d['y'][q])
    write(out/'SCATTERS.json',clean(scatters))
    # Verify nested-class objective ordering on every matched selection cell.
    bounds=[]
    for name in ('inner','fullselect'):
        for positive in (root/'fits').glob(f's1-{name}-positive_products-*'):
            free=positive.with_name(positive.name.replace('positive_products','free_products'))
            if sha(positive/'gram.npz')!=sha(free/'gram.npz'):raise ValueError('matched convex objectives have different Grams')
            for lam in plan['lambdas']:
                a=json.loads((positive/f'lambda-{lam:g}/RESULT.json').read_text())['certificate'];b=json.loads((free/f'lambda-{lam:g}/RESULT.json').read_text())['certificate']
                delta=a['regularized_objective']-b['regularized_objective']
                if delta < -1e-7:raise ValueError('convex nested-class objective ordering violated')
                bounds.append(dict(cell=positive.name,lambda_=lam,positive_objective=a['regularized_objective'],free_objective=b['regularized_objective'],restriction_cost=delta))
    influence=[]
    for t in json.loads((root/'S5_influence.json').read_text()):
        path=root/'fits'/t['id']/f"lambda-{t['lambdas'][0]:g}";base=selected_path(root,t['family'],None)
        pred=np.load(path/'predictions.npz')['native'][:len(d['y'])];control=np.load(base/'predictions.npz')['native'][:len(d['y'])]
        influence.append(dict(task=t['id'],omitted_component=t['omitted_component'],baseline_selected_lambda=json.loads((base/'RESULT.json').read_text())['certificate']['lambda'],refit_lambda=t['lambdas'][0],development_mean_absolute_score_change=float(np.mean(np.abs(pred[d['dev']]-control[d['dev']]))),development_max_absolute_score_change=float(np.max(np.abs(pred[d['dev']]-control[d['dev']])))))
    summary=dict(registered_quadratic_fits=expected,quadratic_fits=len(records),comparator_fits=len(comparators),calibrations=Counter(v['calibration']['status'] for v in records.values() if v['calibration']['status']!='not_requested'),
        maximum_kkt=max(c['kkt_infinity'] for c in certificates),maximum_gap=max(c['objective_gap_upper_bound'] for c in certificates),maximum_scientific_f32_difference=max(v['maximum_f32_scientific_difference'] for v in records.values()),
        main=main,convex_bounds=bounds,influence=influence,comparators=comparators,all_reports_sha256={k:sha(root/k/'REPORT.json') for k in records},limitations=['regularized squared-loss bounds only; no universal SROCC or feature-information ceiling','opaque-vector qualification; no pixel/runtime promotion','source bootstrap conditional on these fits and exposed populations','neural and quadratic objectives and compute budgets differ'])
    write(out/'SUMMARY.json',clean(summary))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    labels=[f.replace('positive_','+ ').replace('free_','free ').replace('_',' ') for f in FAMILIES]
    for i,f in enumerate(FAMILIES):
        tr=[main[f][str(o)]['train']['selection_score'] for o in range(3)];te=[main[f][str(o)]['test']['selection_score'] for o in range(3)]
        axes[0].scatter([i-.12]*3,tr,label='training' if i==0 else None,color='#be5134');axes[0].scatter([i+.12]*3,te,label='unseen sources' if i==0 else None,color='#17659a')
    axes[0].set_xticks(range(5),labels,rotation=25,ha='right');axes[0].set_ylabel('Equal-human mean of pooled / within-source signed SROCC');axes[0].legend();axes[0].set_title('Nested source validation: each dot is a fold')
    for family,color in zip(POSITIVE,('#17659a','#38834f')):
        fractions=[.25,.5,.75,1.];values=[]
        for fraction in fractions:
            selected=[v['test']['selection_score'] for k,v in records.items() if k.startswith('fits/s3-source-'+family+'-') and (('-full/' in k) if fraction==1 else (f'-f{fraction}-' in k))]
            values.append(float(np.mean(selected)))
        axes[1].plot(fractions,values,'o-',label=family.replace('positive_','+ '),color=color)
    axes[1].set_xlabel('Fraction of training source components');axes[1].set_ylabel('Unseen-source composite');axes[1].set_title('Sources added; outer sources stay fixed');axes[1].legend()
    fig.savefig(out/'study.svg');fig.savefig(out/'study.pdf');plt.close(fig)
    print(json.dumps(dict(stage='report',quadratic_fits=len(records),comparators=len(comparators),max_kkt=summary['maximum_kkt'],max_gap=summary['maximum_gap'])),flush=True)


def archive(root):
    plan=json.loads((root/'PLAN.json').read_text());dest=root/'reproduction';dest.mkdir(exist_ok=False)
    inventory={}
    sources=[Path(p) for p in plan['inputs']]+[META.with_name('PEERS.json'),OLD/'PLAN.json',REPO/'Cargo.toml',REPO/'Cargo.lock',REPO/'docs/SHAPED_LINEAR_GENERALIZATION_STUDY.md',REPO/'scripts/rev4_featpot/linear_probe.py',REPO/'scripts/rev4_featpot/shaped_linear_study.py',REPO/'scripts/rev4_featpot/shaped_linear_compare.py',REPO/'scripts/rev4_featpot/shaped_linear_report.py',REPO/'scripts/lib/zen_stats.py',REPO/'zensim-validate/src/bin/bake_dial_refit/linear_study.rs',REPO/'scripts/byconstruction/dataset_mirror.py']
    for i,path in enumerate(dict.fromkeys(sources)):
        copied=dest/f'{i:03d}-{path.name}';shutil.copy2(path,copied);inventory[str(path)]=dict(artifact=str(copied.relative_to(root)),sha256=sha(copied))
    for name in ('shaped-linear-prepare.log','shaped-linear-prepare-r2.log','shaped-linear-oracle.log','shaped-linear-s1.log','shaped-linear-diagnostics.log','shaped-linear-comparators.log','shaped-linear-influence.log','shaped-linear-native-check.log','shaped-linear-native-build-r2.log','shaped-linear-native-build-r3.log'):
        path=Path.home()/'tmp'/name
        if not path.exists():raise FileNotFoundError(path)
        shutil.copy2(path,dest/name)
    write(dest/'INDEX.json',inventory)
    write(root/'CLOSURE.json',dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),report_sha256=sha(root/'report/SUMMARY.json'),bootstrap_sha256=sha(root/'bootstrap/SOURCE_BOOTSTRAP.json'),numeric_gate_sha256=sha(root/'checks/NUMERIC_GATE.json'),reproduction_inventory_sha256=sha(dest/'INDEX.json'),final_test_accessed=False,serving_promoted=False,finished=datetime.now(timezone.utc).isoformat()))


def linear_details(root):
    from shaped_linear_study import permuted,term_matrix
    plan,d=load(root);out=root/'linear-details';out.mkdir(exist_ok=True)
    n=len(d['y']);fit=np.flatnonzero(~d['dev']);objective=[];calibration=[]
    for complete in sorted((root/'fits').glob('*/COMPLETE.json')):
        task=json.loads((complete.parent/'TASK.json').read_text());ids=np.asarray(task['train'])
        y=permuted(d['y'],d,ids,task['permutation'],task['seed']) if task.get('permutation') else d['y']
        w,target,maps=weights_and_targets(y[ids],d['pop'][ids],d['component'][ids],'range',1.)
        if task.get('pairs'):
            pair=np.load(complete.parent/'pairs.npz');left=pair['left'];right=pair['right'];weight=pair['weight']
            all_target=np.full(n,np.nan);all_target[ids]=target
            truth=all_target[left]-all_target[right]
        else:weight=w;truth=target
        for cell in json.loads(complete.read_text())['results']:
            path=Path(cell['path']);r=json.loads((path/'RESULT.json').read_text());pred=np.load(path/'predictions.npz')
            values={}
            for kind in ('optimum','native'):
                q=pred[kind][:n]/100
                residual=q[left]-q[right]-truth if task.get('pairs') else q[ids]-truth
                values[kind]=float(np.dot(weight,residual*residual)/weight.sum()/2)
            cert=r['certificate'];error=abs(values['optimum']-cert['weighted_half_mse'])
            if error>1e-8:raise ValueError('independent row-space objective disagrees with Gram certificate')
            objective.append(dict(task=task['id'],lambda_=cell['lambda_'],independent_half_mse=values['optimum'],native_half_mse=values['native'],gram_rowspace_absolute_error=error,f32_half_mse_difference=values['native']-values['optimum'],regularized_objective_lower_bound=cert['regularized_objective']-cert['objective_gap_upper_bound']))
            if r['calibration']['status']!='not_requested':
                row=dict(task=task['id'],status=r['calibration'])
                if (path/'calibrated.npz').exists():
                    raw=pred['native'][:n];mapped=np.load(path/'calibrated.npz')['native'][:n]
                    order=np.argsort(raw,kind='stable');diff=np.diff(mapped[order]);minimum=float(diff.min())
                    if minimum < -1e-5:raise ValueError('calibration reversed raw projection order')
                    row.update(raw_distinct=int(len(np.unique(raw))),mapped_distinct=int(len(np.unique(mapped))),minimum_sorted_increment=minimum)
                calibration.append(row)
    duplicates=json.loads((root/'audit/DUPLICATES.json').read_text())['exact_feature_groups']
    weights,targets,maps=weights_and_targets(d['y'][fit],d['pop'][fit],d['component'][fit],'range',1.)
    local={v:i for i,v in enumerate(fit)};collision=0.;clusters=[]
    meta=pq.read_table(META,columns=['pair_key','ref_pixels_sha256','dist_pixels_sha256']).to_pydict()
    for group in duplicates:
        ids=[local[i] for i in group['ids'] if i in local]
        if len(ids)<2:continue
        yy=targets[ids];ww=weights[ids];best=float(np.average(yy,weights=ww));loss=float(np.dot(ww,(yy-best)**2)/weights.sum()/2);collision+=loss
        original=group['ids'];clusters.append(dict(rows=original,mapped_targets=yy.tolist(),optimal_common_prediction=best,half_mse_floor=loss,pair_keys=[meta['pair_key'][i] for i in original],distorted_pixel_hashes=[meta['dist_pixels_sha256'][i] for i in original]))
    stability=[]
    for family in POSITIVE:
        for outer in range(3):
            base=root/'fits'/f's3-source-{family}-o{outer}-full';bs=json.loads((base/'SPEC.json').read_text());bp=np.load(base/'lambda-0.0001/predictions.npz')
            train=np.array(json.loads((base/'TASK.json').read_text())['train']);test=np.array(json.loads((base/'TASK.json').read_text())['test'])
            grid=np.quantile(term_matrix(d['raw'][train],bs['terms']),[0,.25,.5,.75,1.],axis=0)
            def slopes(spec,coef):
                sc=np.array(spec['scales']);norm=np.array(spec['basis_scales'])
                if not spec['knots']:return np.broadcast_to(coef/sc/norm,grid.shape)
                k=np.array(spec['knots'])*sc[:,None];bin=(grid>k[:,0]).astype(int)+(grid>k[:,1]).astype(int)
                grad=coef.reshape(-1,3)/sc[:,None]/norm.reshape(-1,3)
                return np.take_along_axis(np.broadcast_to(grad,(len(grid),*grad.shape)),bin[:,:,None],axis=2)[:,:,0]
            reference=slopes(bs,bp['coefficients']).ravel();norm=float(np.linalg.norm(reference))
            for prefix in ('s3-source','s3-rows'):
                for path in sorted((root/'fits').glob(f'{prefix}-{family}-o{outer}-*')):
                    spec=json.loads((path/'SPEC.json').read_text());pred=np.load(path/'lambda-0.0001/predictions.npz');v=slopes(spec,pred['coefficients']).ravel();denom=float(np.linalg.norm(v)*norm)
                    stability.append(dict(task=path.name,baseline=base.name,diagnostic='continuous basis slopes at five TRAIN-only raw-term quantiles; not pixel gradients',slope_cosine=float(np.dot(v,reference)/denom) if denom else None,relative_slope_l2_change=float(np.linalg.norm(v-reference)/norm) if norm else None,outer_mean_absolute_score_change=float(np.mean(np.abs(pred['native'][test]-bp['native'][test]))),nonzero_coefficients=int(np.sum(pred['coefficients']!=0))))
    os.environ['ZEN_PANEL_BIN']=str(root/'bin/panel')
    transfer=[]
    for task in json.loads((root/'S5_family.json').read_text()):
        family=task['family'];outer=int(task['id'].split('-o')[1].split('-')[0]);excluded=task['excluded']
        codecs={'avif':['aom','cld_avif','vis_avif'],'jpeg':['mozjpeg'],'jxl':['libjxl'],'webp':['cld_webp']}.get(excluded,[excluded])
        ids=np.array(task['test']);path=root/'fits'/task['id']/'lambda-0.0001';baseline=root/'fits'/f's3-source-{family}-o{outer}-full'/'lambda-0.0001'
        pred=np.load(path/'predictions.npz')['native'][:n];control=np.load(baseline/'predictions.npz')['native'][:n]
        jobs=[]
        for pop in sorted(set(d['pop'][ids])):
            selected=ids[(d['pop'][ids]==pop)&np.isin(d['family'][ids],codecs)]
            if not len(selected):continue
            for label,q in [('excluded',selected),('retained',ids[(d['pop'][ids]==pop)&~np.isin(d['family'][ids],codecs)])]:
                if len(q)<2:raise ValueError('undefined transfer panel population')
                jobs.extend([(pop+'|'+label+'|refit','refit','y',q),(pop+'|'+label+'|control','control','y',q)])
        panels=panel_batch_indexed(dict(refit=pred,control=control,y=d['y']),jobs,stats='srocc')
        transfer.append(dict(task=task['id'],family=family,outer=outer,excluded=excluded,panels=panels))
    write(out/'TRANSFER.json',clean(transfer))
    amplitudes=[]
    for task in json.loads((root/'S4_linear.json').read_text()):
        path=root/'fits'/task['id']/'lambda-0.0001';pred=np.load(path/'predictions.npz')['native'][:n];ids=np.array(task['test']);train=np.array(task['train'])
        y=permuted(d['y'],d,train,task['permutation'],task['seed']);_,_,maps=weights_and_targets(y[train],d['pop'][train],d['component'][train],'range',1.)
        for pop in sorted(set(d['pop'][ids])):
            q=ids[d['pop'][ids]==pop];lo,hi=maps[pop]['values'][0],maps[pop]['values'][-1]
            value=scatter(pred[q],100*(d['y'][q]-lo)/(hi-lo))
            if value['status']!='MEASURED':raise ValueError('negative-control scatter incomplete')
            amplitudes.append(dict(task=task['id'],population=pop,raw=value['raw'],geometry=value['geo']))
    write(out/'PERMUTATION_SCATTER.json',clean(amplitudes))
    write(out/'OBJECTIVES.json',dict(fits=len(objective),maximum_rowspace_gram_difference=max(v['gram_rowspace_absolute_error'] for v in objective),maximum_absolute_f32_loss_change=max(abs(v['f32_half_mse_difference']) for v in objective),fits_detail=objective))
    write(out/'COLLISION_FLOOR.json',dict(full_fit_weighted_half_mse_floor=collision,clusters=clusters,scope='unregularized mapped-label squared-loss lower bound for any deterministic function of these 576 features on this frozen FIT population; only the exact collisions contribute; not a tight bound or SROCC ceiling'))
    write(out/'CALIBRATION.json',calibration);write(out/'STABILITY.json',stability)
    print(json.dumps(dict(stage='linear-details',fits=len(objective),calibrations=len(calibration),stability_cells=len(stability),collision_half_mse_floor=collision)),flush=True)


def label_structure(root):
    from shaped_linear_study import permuted,mapped_labels,metrics
    _,d=load(root);os.environ['ZEN_PANEL_BIN']=str(root/'bin/panel');seen=set();records=[]
    for task in json.loads((root/'S4_linear.json').read_text()):
        key=(tuple(task['train']),task['permutation'],task['seed'])
        if key in seen:continue
        seen.add(key);ids=np.array(task['train']);y=permuted(d['y'],d,ids,task['permutation'],task['seed'])
        _,_,maps=weights_and_targets(y[ids],d['pop'][ids],d['component'][ids],'range',1.)
        records.append(dict(representative_task=task['id'],mode=task['permutation'],seed=task['seed'],original_vs_permuted=metrics(mapped_labels(y,d['pop'],maps)*100,d['y'],ids,d,maps)))
    write(root/'linear-details/LABEL_STRUCTURE.json',records)
    print(json.dumps(dict(stage='label-structure',permutations=len(records))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','linear-details','label-structure','bootstrap','report','archive']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();{'audit':audit,'linear-details':linear_details,'label-structure':label_structure,'bootstrap':bootstrap,'report':reduce,'archive':archive}[a.stage](a.root)


if __name__=='__main__':main()
