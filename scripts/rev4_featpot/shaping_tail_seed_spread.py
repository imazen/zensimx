"""Within-condition seed spread from saved paired tail metrics; no new scoring."""
import argparse,csv,json,math,statistics
from collections import defaultdict
from pathlib import Path
from shaping_completion import encoded,sha


def summarise(path,output):
    groups=defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            if row['role']=='validation' and row['population']=='human_equal' and row['metric'] in ('outlier_ratio','shape_p99','z_rmse'):
                groups[(row['family'],row['metric'],row['partition'],int(row['setting']))].append(row)
    if len(groups)!=9*3*24:raise ValueError('complete nine-family, three-metric, 24-condition comparison required')
    by_family=defaultdict(list)
    for (family,metric,partition,setting),rows in groups.items():
        if len(rows)!=3 or {int(row['seed']) for row in rows}!={0,1,2}:raise ValueError('three distinct paired seeds required')
        a=[float(row['baseline']) for row in rows];b=[float(row['shaped']) for row in rows]
        if not all(map(math.isfinite,(*a,*b))):raise ValueError('finite paired seed metrics required')
        by_family[(family,metric)].append(dict(partition=partition,setting=setting,
            baseline_sd=statistics.stdev(a),shaped_sd=statistics.stdev(b),baseline_max=max(a),shaped_max=max(b)))
    result=[]
    for (family,metric),rows in sorted(by_family.items()):
        expected={(f'inner-{o}-{i}',s) for o in range(3) for i in range(2) for s in range(4)}
        if len(rows)!=24 or {(r['partition'],r['setting']) for r in rows}!=expected:raise ValueError('all partition/setting groups required')
        result.append(dict(family=family,metric=metric,condition_groups=24,seeds_per_group=3,
            mean_baseline_seed_sd=statistics.fmean(x['baseline_sd'] for x in rows),
            mean_shaped_seed_sd=statistics.fmean(x['shaped_sd'] for x in rows),
            groups_with_lower_sd=sum(x['shaped_sd']<x['baseline_sd'] for x in rows),
            worst_baseline=max(x['baseline_max'] for x in rows),worst_shaped=max(x['shaped_max'] for x in rows),groups=rows))
    output.mkdir(parents=True,exist_ok=False)
    value=dict(input=str(path),input_sha256=sha(path),driver_sha256=sha(__file__),
        scope='Descriptive within-partition/setting seed spread. Human-equal metrics; no new scoring, fitting, selection or confidence intervals.',results=result)
    (output/'SUMMARY.json').write_bytes(encoded(value));return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cases','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(dict(summaries=len(summarise(a.cases,a.output)['results']))))
