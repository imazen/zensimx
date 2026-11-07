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
    libm=ctypes.CDLL('libm.so.6');fmaf=libm.fmaf
    fmaf.argtypes=[ctypes.c_float]*3;fmaf.restype=ctypes.c_float
    for function in (libm.expf,libm.log1pf):
        function.argtypes=[ctypes.c_float];function.restype=ctypes.c_float
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
        elif layer['activation']=='softplus':
            # Match the native std construction; final native-bit equality below
            # remains mandatory, including on platforms with different libm bits.
            for index in np.ndindex(out.shape):
                value=float(out[index]);tail=libm.log1pf(libm.expf(-abs(value)))
                out[index]=np.float32(value+tail if value>0 else tail)
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
        elif layer['activation']=='softplus':x=np.maximum(z,0.)+np.log1p(np.exp(-np.abs(z)))
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


def repair_probe(original,probes,binaries,inspector,out):
    """Diagnostic-only repair through the existing native admission and scorer."""
    import subprocess
    if out.exists():raise ValueError('preserve existing diagnostic repair')
    build=binaries/'BUILD.json'
    for name,pin in json.loads(build.read_text())['binary_sha256'].items():
        if sha(binaries/name)!=pin:raise ValueError('diagnostic runtime drift')
    pins={str(p):sha(p) for p in (original,probes,build,inspector)}
    out.mkdir(parents=True)
    inspected=subprocess.run([str(inspector),str(original),'--weights'],capture_output=True,text=True)
    (out/'inspection.log').write_text(inspected.stderr);inspected.check_returncode()
    model=out/'MODEL.json';model.write_text(inspected.stdout)
    pairs=unique_signed_pairs(json.loads(inspected.stdout))
    declaration=out/'pairs.txt';declaration.write_text('relu-max-pairs-v1\n'+''.join(f'{u} {v}\n' for u,v in pairs))
    repaired=out/'repaired.bin'
    result=subprocess.run([str(binaries/'bake_dial_refit'),'repair-research-max-pairs','--in',str(original),'--out',str(repaired),'--pairs',str(declaration)],capture_output=True,text=True)
    (out/'repair.log').write_text(result.stdout+result.stderr);result.check_returncode()
    os.environ['REV4_V2_BIN_DIR']=str(binaries)
    import v2_common,v2_lodo_mlp as owner
    owner.FITBIN=v2_common.FITBIN=binaries/'bake_dial_refit'
    values=owner.predict(repaired,probes,out/'probes.tsv')
    # The identity row is the first of the nine rays, not an extra row.
    if len(values)!=265:raise ValueError('registered 128-pair and nine-ray probe layout required')
    changes=values[128:256]-values[:128]
    for name,pin in pins.items():
        if sha(name)!=pin:raise ValueError('diagnostic input changed')
    record=dict(pairs=pairs,identity=float(values[256]),finite=bool(np.isfinite(values).all()),
        damage_increase_violations=int(np.sum(changes>1e-5)),maximum_damage_increase=float(max(0,changes.max())),
        selection_candidate=False,scope='diagnostic training-input probes only; not complete qualification or selection admission',
        evidence_sha256={**pins,**{str(p):sha(p) for p in (model,declaration,repaired,out/'repair.log',out/'probes.tsv')}},driver_sha256=sha(__file__))
    (out/'RESULT.json').write_bytes(encoded(record));print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probes',type=Path);p.add_argument('--predictions',type=Path)
    p.add_argument('--declare-pairs',action='store_true')
    p.add_argument('--repair-probes',action='store_true');p.add_argument('--original',type=Path)
    p.add_argument('--binary-dir',type=Path);p.add_argument('--inspector',type=Path);a=p.parse_args()
    if a.repair_probes:
        if a.declare_pairs or a.model or a.predictions or not all((a.original,a.probes,a.binary_dir,a.inspector)):p.error('repair-probes requires original, probes, binary-dir and inspector only')
        repair_probe(a.original,a.probes,a.binary_dir,a.inspector,a.output)
    elif a.declare_pairs:
        if a.model is None:p.error('declare-pairs requires model')
        pairs=unique_signed_pairs(json.loads(a.model.read_text()))
        with a.output.open('x') as out:out.write('relu-max-pairs-v1\n'+''.join(f'{u} {v}\n' for u,v in pairs))
        print(json.dumps(dict(pairs=pairs,model_sha256=sha(a.model),declaration_sha256=sha(a.output))))
    else:
        if a.model is None or a.probes is None or a.predictions is None:p.error('trace requires model, probes and predictions')
        trace(a.model,a.probes,a.predictions,a.output)
