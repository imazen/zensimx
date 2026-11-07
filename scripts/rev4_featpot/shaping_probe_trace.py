"""Independent f64 algebra diagnostic of stored dense weights, not a scorer."""
from __future__ import annotations
import argparse,csv,json,os,ctypes
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import pyarrow.parquet as pq
from shaping_completion import encoded,sha


def f32_layers(raw,model):
    # libc fmaf supplies one correctly rounded float32 fused multiply-add.
    # This is an independent scalar reference; equality is checked against native.
    fmaf=ctypes.CDLL('libm.so.6').fmaf
    fmaf.argtypes=[ctypes.c_float]*3;fmaf.restype=ctypes.c_float
    scale=np.array(model['scaler_scale'],np.float32);scale=np.where(scale==0,np.float32(1),scale)
    x=(raw.astype(np.float32)-np.array(model['scaler_mean'],np.float32))/scale
    outputs=[]
    for layer in model['layers']:
        w=np.array(layer['weights'],np.float32).reshape(layer['in_dim'],layer['out_dim'])
        out=np.tile(np.array(layer['biases'],np.float32),(len(x),1))
        for row in range(len(x)):
            for i in range(layer['in_dim']):
                value=float(x[row,i])
                if value==0:continue
                for j in range(layer['out_dim']):out[row,j]=fmaf(value,float(w[i,j]),float(out[row,j]))
        if layer['activation']=='relu':out=np.maximum(out,np.float32(0))
        elif layer['activation']!='identity':raise ValueError('unsupported scalar activation')
        outputs.append(out);x=out
    return outputs


def unique_signed_pairs(model):
    """Conservative proposal only; the native parser owns admission."""
    first,next_layer=model['layers'][:2]
    w=np.array(first['weights'],np.float64).reshape(first['in_dim'],first['out_dim'])
    bias=np.array(first['biases']);middle=np.array(next_layer['weights'],np.float32).reshape(next_layer['in_dim'],next_layer['out_dim'])
    signed=np.any(w<0,axis=0);used=set();pairs=[]
    for v in np.flatnonzero(signed):
        candidates=[u for u in range(len(bias)) if u not in used and u!=v and not signed[u]
                    and bias[u]==0 and bias[v]<=0
                    and np.array_equal(middle[u].view(np.uint32),middle[v].view(np.uint32))
                    and np.all(w[:,u]+w[:,v]>=0)]
        if len(candidates)!=1:raise ValueError('ambiguous or unsupported signed pair')
        u=candidates[0];pairs.append([u,int(v)]);used.update((u,int(v)))
    if not pairs:raise ValueError('no signed pairs to repair')
    return pairs


def trace(model_path,probes,predictions,out):
    if out.exists():raise ValueError('preserve existing trace')
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
        f32=f32_layers(raw[[i,j]],model)
        if not np.array_equal(f32[-1][:,0].view(np.uint32),native[[i,j]].astype(np.float32).view(np.uint32)):
            raise ValueError('scalar fused reference does not reproduce native failing scores')
        first=np.array(model['layers'][0]['weights']).reshape(model['layers'][0]['in_dim'],-1)
        middle=np.array(model['layers'][1]['weights'],np.float32).reshape(model['layers'][1]['in_dim'],-1)
        tied={}
        for unit,weights in enumerate(middle):tied.setdefault(weights.tobytes(),[]).append(unit)
        group_changes=[]
        for units in tied.values():
            if len(units)>1 and (first[:,units]<0).any():
                group_changes.append(dict(units=units,f64_change=float(layers[0][j,units].sum()-layers[0][i,units].sum()),
                    f32_activation_sum_change=float(f32[0][1,units].astype(np.float64).sum()-f32[0][0,units].astype(np.float64).sum())))
        cases.append(dict(pair=i,native_low=float(native[i]),native_high=float(native[j]),native_increase=float(native[j]-native[i]),
            f64_low=float(x[i,0]),f64_high=float(x[j,0]),f64_increase=float(x[j,0]-x[i,0]),
            changed_derived_inputs=changed.tolist(),scalar_fused_native_bits_equal=True,
            f32_layer_minimum_changes=[float((v[1].astype(np.float64)-v[0]).min()) for v in f32],
            tied_signed_first_layer_groups=group_changes,layer_minimum_activation_changes=[float((v[j]-v[i]).min()) for v in layers],
            layer_maximum_activation_changes=[float((v[j]-v[i]).max()) for v in layers]))
    result=dict(pins={str(p):sha(p) for p in (model_path,probes,predictions)},cases=cases,
                scope='independent f64 dense algebra from stored f32 coefficients; not native replay or qualification',
                native_failure_preserved=True)
    out.write_bytes(encoded(result));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probes',type=Path);p.add_argument('--predictions',type=Path)
    p.add_argument('--declare-pairs',action='store_true');a=p.parse_args()
    if a.declare_pairs:
        pairs=unique_signed_pairs(json.loads(a.model.read_text()))
        with a.output.open('x') as out:out.write('relu-max-pairs-v1\n'+''.join(f'{u} {v}\n' for u,v in pairs))
        print(json.dumps(dict(pairs=pairs,model_sha256=sha(a.model),declaration_sha256=sha(a.output))))
    else:
        if a.probes is None or a.predictions is None:p.error('trace requires probes and predictions')
        trace(a.model,a.probes,a.predictions,a.output)
