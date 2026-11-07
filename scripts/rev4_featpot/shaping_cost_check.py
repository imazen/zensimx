"""Exercise native cost parity and corruption refusal without collecting timings."""
from __future__ import annotations
import argparse,copy,json,subprocess
from pathlib import Path
from shaping_completion import encoded,sha


def check(binary,workload,output):
    if output.exists():raise ValueError('preserve existing qualification')
    binary=binary.resolve();workload=workload.resolve()
    original=json.loads(workload.read_text());binary_pin=sha(binary);workload_pin=sha(workload)
    output.mkdir(parents=True)
    runs=[]
    def invoke(name,payload,error=None):
        path=output/(name+'.json');path.write_bytes(encoded(payload))
        destination=output/name
        with (output/(name+'.log')).open('xb') as log:
            result=subprocess.run([str(binary),'qualify',str(path),str(destination)],stdout=log,stderr=subprocess.STDOUT)
        if sha(binary)!=binary_pin or sha(workload)!=workload_pin:raise ValueError('qualification input changed')
        if error is None:
            if result.returncode:raise ValueError('native parity failed; preserved '+str(output/(name+'.log')))
            receipt=json.loads((destination/'RESULT.json').read_text())
            if not receipt['complete'] or not receipt['preflight_exact'] or not receipt['postflight_exact'] or receipt['timing_measured']:
                raise ValueError('native parity coverage differs')
        elif result.returncode==0 or error not in (output/(name+'.log')).read_text() or (destination/'RESULT.json').exists():
            raise ValueError('corrupt input was not refused as expected')
        runs.append(dict(case=name,exit_code=result.returncode,expected_refusal=error,
                         workload_sha256=sha(path),log_sha256=sha(output/(name+'.log'))))
    for arm in original['arms']:
        path=Path(arm['model'])
        if not path.is_absolute():arm['model']=str(workload.parent/path)
    invoke('exact',original)
    wrong=copy.deepcopy(original);wrong['arms'][0]['expected_score_bits'][0]^=1
    invoke('wrong-reference',wrong,'score bits')
    wrong=copy.deepcopy(original);wrong['arms'][0]['sha256']='0'*64
    invoke('wrong-model-hash',wrong,'model hash differs')
    receipt=dict(binary_sha256=binary_pin,source_workload_sha256=workload_pin,
                 models=len(original['arms']),rows=len(original['feature_bits']),runs=runs,
                 native_instrument_qualified=True,timing_measured=False)
    (output/'QUALIFICATION.json').write_bytes(encoded(receipt));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','workload','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();print(json.dumps(check(args.binary,args.workload,args.output)))
