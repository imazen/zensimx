"""Compare existing native tail diagnostics; never fit, score or select a model."""
import argparse,csv,hashlib,itertools,json,math,statistics
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha
from shaping_execution import load_registration

HUMAN=('kadid','tid2013','konfig','cid22_a25')
POPULATIONS=(*HUMAN,'cid22_ssim2')
METRICS={
    'outlier_ratio':('full','or'),
    'z_rmse':('full','z_rmse'),
    'mae_raw':('full','mae_raw'),
    'raw_residual_p99':('raw','residual_p99'),
    'raw_residual_mad':('raw','mad'),
    'shape_p99':('geo','p99d'),
    'shape_outside_envelope':('geo','out4'),
}


def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def paired_summary(pairs):
    good=[(a,b) for a,b in pairs if finite(a) and finite(b)]
    complete=len(good)==len(pairs) and bool(pairs)
    return dict(required=len(pairs),finite=len(good),
        baseline=statistics.fmean(a for a,b in good) if complete else None,
        shaped=statistics.fmean(b for a,b in good) if complete else None,
        delta=statistics.fmean(b-a for a,b in good) if complete else None,
        lower=sum(b<a for a,b in good),equal=sum(b==a for a,b in good),higher=sum(b>a for a,b in good))


def read_panel(dashboard,index,record,plan_pin):
    cell=record['cell'];entry=index['cells'][cell['id']]
    if entry['cell']!=cell or not record.get('details') or entry['report_sha256']!=record['details']['report_sha256']:
        raise ValueError('saved panel registration/report differs')
    path=dashboard/entry['native_url']
    if not path.resolve().is_relative_to((dashboard/'details/native').resolve()):raise ValueError('native panel outside dashboard')
    row=json.loads(path.read_text())
    if (row['schema']!='zensim-native-detailed-panel-v1' or row['cell']!=cell
        or row['report_sha256']!=entry['report_sha256'] or row['outer_results_read']):
        raise ValueError('native panel identity or exposure differs')
    if index['plan_sha256']!=plan_pin or index['outer_results_read']:raise ValueError('native index scope differs')
    values={}
    for role,pop in itertools.product(('train','validation'),POPULATIONS):
        label=pop+'|'+role;full=row['statistics']['full'][label];scatter=row['statistics']['scatter'][label]
        if full['n_dropped'] or full['n']!=scatter['n']:raise ValueError('native panel denominator differs')
        geometry=scatter if scatter['status']=='MEASURED' else {}
        tables=dict(full=full,raw=geometry.get('raw',{}),geo=geometry.get('geo',{}))
        values[(role,pop)]={k:tables[table].get(name) for k,(table,name) in METRICS.items()}
    pin=dict(path=str(path),sha256=sha(path),report_sha256=row['report_sha256'],
             input_sha256=row['input_sha256'],panel_sha256=row['panel_sha256'])
    return values,pin


def summarise(study,dashboard,output):
    plan,cells,_=load_registration(study);registered={c['id']:c for c in cells}
    snapshot_bytes=(dashboard/'stats.json').read_bytes();index_bytes=(dashboard/'details/INDEX.json').read_bytes()
    snapshot=json.loads(snapshot_bytes);index=json.loads(index_bytes)
    if index['schema']!='zensim-inner-scatter-index-v1':raise ValueError('native index schema differs')
    plan_pin=sha(study/'PLAN.json')
    if snapshot['plan_sha256']!=plan_pin:raise ValueError('snapshot plan differs')
    records={f:{} for f in NEURAL};pins={};values={}
    expected=set(itertools.product((f'inner-{o}-{i}' for o in range(3) for i in range(2)),range(4),range(3)))
    for row in snapshot['records']:
        c=row['cell'];f=c['family']
        if c['stage']!='inner' or f not in records:continue
        if registered.get(c['id'])!=c:raise ValueError('primary cell differs from registration')
        key=(c['partition'],c['setting'],c['seed'])
        if key in records[f]:raise ValueError('duplicate paired condition')
        records[f][key]=row
    for f,rows in records.items():
        if set(rows)!=expected:raise ValueError('complete 72-condition neural primary required: '+f)
        for key,row in rows.items():
            cid=row['cell']['id'];values[cid],pins[cid]=read_panel(dashboard,index,row,plan_pin)
    cases=[];summary=[]
    for f in NEURAL:
        if f=='relu':continue
        paired={}
        for key in sorted(expected):
            a=records['relu'][key]['cell']['id'];b=records[f][key]['cell']['id']
            if pins[a]['panel_sha256']!=pins[b]['panel_sha256']:raise ValueError('paired native statistic binaries differ')
            for role in ('train','validation'):
                for pop in (*POPULATIONS,'human_equal'):
                    def metric(cid,name):
                        populations=HUMAN if pop=='human_equal' else (pop,)
                        data=[values[cid][(role,p)][name] for p in populations]
                        return statistics.fmean(data) if all(map(finite,data)) else None
                    for name in METRICS:
                        baseline,shaped=metric(a,name),metric(b,name)
                        paired.setdefault((role,pop,name),[]).append((baseline,shaped))
                        cases.append(dict(family=f,partition=key[0],setting=key[1],seed=key[2],
                            baseline_cell=a,shaped_cell=b,role=role,population=pop,metric=name,
                            baseline=baseline,shaped=shaped,delta=shaped-baseline if finite(baseline) and finite(shaped) else None))
        for (role,pop,name),pairs in paired.items():
            summary.append(dict(family=f,role=role,population=pop,metric=name,**paired_summary(pairs)))
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='matched-primary-tail-summary-v1',plan_sha256=plan_pin,
        snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        input_panels=pins,summary=summary,primary_conditions_per_family=72,
        interpretation='Descriptive existing inner results across all four settings and paired seeds. No retuning or architecture selection. Lower tail/error metrics are better; corpus-standard-deviation outlier ratio, raw calibration error and quantile-normalized geometry are separate. Human composite averages four populations; teacher is separate. Counts across conditions are not independent confidence intervals.',
        outer_results_read=False,new_fitting=False,new_scoring=False,driver_sha256=sha(__file__))
    (output/'SUMMARY.json').write_bytes(encoded(result))
    with (output/'PAIRED_CASES.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','dashboard','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=summarise(a.study,a.dashboard,a.output)
    print(json.dumps(dict(panels=len(r['input_panels']),conditions_per_family=r['primary_conditions_per_family'],summary_rows=len(r['summary']))))
