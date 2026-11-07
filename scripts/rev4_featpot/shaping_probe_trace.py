"""Independent f64 algebra diagnostic of stored dense weights, not a scorer."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import pyarrow.parquet as pq
from shaping_completion import encoded,sha


def trace(model_path,probes,predictions,out):
    model=json.loads(model_path.read_text());meta={x['key']:x.get('value_text') for x in model['metadata']}
    rows=list(csv.DictReader(predictions.open(),delimiter='\t'));native=np.array([float(r['pred']) for r in rows])
    failures=np.flatnonzero(native[128:256]-native[:128]>1e-5)
    if not len(failures):raise ValueError('no failing registered probe pairs')
    table=pq.read_table(probes);terms=meta['zensim.derived_inputs'].splitlines()
    if terms[0]!='zensim-derived-inputs v1':raise ValueError('unsupported derived layout')
    columns=[]
    for line in terms[1:]:
        tokens=line.split();values=[np.asarray(table[f'f{int(i)}'].to_numpy(),dtype=np.float32) for i in tokens[1:]]
        if tokens[0]=='in' and len(values)==1:value=values[0]
        elif tokens[0]=='product' and len(values)==2:value=np.multiply(*values,dtype=np.float32)
        else:raise ValueError('unsupported diagnostic derived term: '+line)
        columns.append(value)
    raw=np.stack(columns,axis=1).astype(np.float64)
    if raw.shape[1]!=model['n_inputs']:raise ValueError('input layout differs')
    x=(raw-np.array(model['scaler_mean']))/np.array(model['scaler_scale']);layers=[]
    for layer in model['layers']:
        w=np.array(layer['weights'],dtype=np.float64).reshape(layer['in_dim'],layer['out_dim']);b=np.array(layer['biases'],dtype=np.float64)
        z=x@w+b
        if layer['activation']=='relu':x=np.maximum(z,0.)
        elif layer['activation']=='identity':x=z
        else:raise ValueError('unsupported diagnostic activation')
        layers.append(x.copy())
    cases=[]
    for index in failures:
        i=int(index);j=i+128;changed=np.flatnonzero(raw[j]!=raw[i])
        cases.append(dict(pair=i,native_low=float(native[i]),native_high=float(native[j]),native_increase=float(native[j]-native[i]),
            f64_low=float(x[i,0]),f64_high=float(x[j,0]),f64_increase=float(x[j,0]-x[i,0]),
            changed_derived_inputs=changed.tolist(),layer_minimum_activation_changes=[float((v[j]-v[i]).min()) for v in layers],
            layer_maximum_activation_changes=[float((v[j]-v[i]).max()) for v in layers]))
    result=dict(pins={str(p):sha(p) for p in (model_path,probes,predictions)},cases=cases,
                scope='independent f64 dense algebra from stored f32 coefficients; not native replay or qualification',
                native_failure_preserved=True)
    out.write_bytes(encoded(result));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model','probes','predictions','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();trace(a.model,a.probes,a.predictions,a.output)
