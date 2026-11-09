"""Read-only numeric-class profiling of an existing native full-state snapshot."""
import argparse,hashlib,json,math,socket,sys
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha

MAX_BYTES=(64*1024**2-4096)//3


def read_state(path):
    with path.open('rb') as stream:
        raw=stream.read(MAX_BYTES+1)
    if not 40<len(raw)<=MAX_BYTES:raise ValueError('bounded complete snapshot required')
    if hashlib.sha256(raw[40:]).digest()!=raw[8:40]:raise ValueError('snapshot payload checksum differs')
    value=json.loads(raw[40:])
    if value['identity']!=path.parent.name:raise ValueError('snapshot identity differs')
    return int.from_bytes(raw[:8],'little'),value,hashlib.sha256(raw).hexdigest()


def numeric_arrays(value,prefix=''):
    rows=[]
    if isinstance(value,dict):
        for k,v in value.items():rows.extend(numeric_arrays(v,prefix+'.'+k if prefix else k))
    elif isinstance(value,list):
        if all(type(v) in (int,float) for v in value):
            tiny=[abs(v) for v in value if math.isfinite(v) and 0<abs(v)<sys.float_info.min]
            positive=[abs(v) for v in value if math.isfinite(v) and v!=0]
            rows.append(dict(array=prefix,length=len(value),zero=sum(v==0 for v in value),
                subnormal=len(tiny),nonfinite=sum(not math.isfinite(v) for v in value),
                min_nonzero=min(positive,default=None),max_absolute=max(positive,default=None)))
        else:
            for i,v in enumerate(value):rows.extend(numeric_arrays(v,prefix+'.'+str(i)))
    return rows


def probe(binding,output):
    bound=json.loads(binding.read_text());run=Path(bound['run']);directory=Path(bound['snapshot_directory'])
    if (bound['schema']!='native-snapshot-run-binding-v1' or bound['host']!=socket.gethostname()
        or sha(run/'RUN.json')!=bound['run_sha256']):raise ValueError('local phase binding differs')
    paths=list(directory.glob('state-*.snap'))
    if not 1<=len(paths)<=2:raise ValueError('one or two committed native states required')
    states=[(p,*read_state(p)) for p in paths];path,generation,value,pin=max(states,key=lambda r:r[1])
    arrays=[]
    for key in ('adam','depth','shapes'):arrays.extend(numeric_arrays(value.get(key),key))
    row=dict(checked_utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),
        driver_sha256=sha(__file__),
        binding=str(binding),binding_sha256=sha(binding),run_sha256=bound['run_sha256'],
        snapshot=str(path),generation=generation,snapshot_sha256=pin,
        payload_checksum_verified=True,epoch=value['epoch'],pair=value['pair'],arrays=arrays,
        total_numeric_values=sum(r['length'] for r in arrays),subnormal_values=sum(r['subnormal'] for r in arrays),
        native_state_modified=False,native_training_started=False,
        interpretation='Numeric-class observation only; not a recovery qualification or a cause-of-slowdown proof.')
    with output.open('xb') as stream:stream.write(encoded(row))
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binding','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=probe(a.binding,a.output)
    print(json.dumps({k:r[k] for k in ('host','epoch','pair','total_numeric_values','subnormal_values')}))
