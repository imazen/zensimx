"""Compare inference adapters against frozen native training/parity responses."""
import argparse,csv,json,subprocess
from pathlib import Path
import numpy as np
from shaping_completion import sha,encoded


def qualify(args):
    args.output.mkdir(parents=True,exist_ok=False)
    commands=[('tree-parity',[args.binaries/'corrhead_parity','--head',args.tree_model,'--parity',args.tree_parity,'--set','test']),
        ('tree-inference',[args.binaries/'corrhead_parity','--head',args.tree_model,'--parity',args.tree_parity,'--set','test','--predict-raw',args.output/'tree.tsv']),
        ('linear-inference',[args.binaries/'bake_dial_refit','linear-predict','--bake',args.linear_model,'--inputs',args.linear_inputs,'--out',args.output/'linear.tsv'])]
    for name,command in commands:
        with (args.output/(name+'.log')).open('x') as log:subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,check=True)
    measurements={}
    for name,reference,key,column in [('tree',args.tree_parity,'test_raw','raw'),('linear',args.linear_reference,'native','pred')]:
        with np.load(reference,allow_pickle=False) as arrays:expected=arrays[key].astype(np.float64)
        with (args.output/(name+'.tsv')).open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
        if [int(row['row_idx']) for row in rows]!=list(range(len(expected))):raise ValueError('native row order differs')
        actual=np.array([float(row[column]) for row in rows],dtype=np.float64)
        differences=int(np.sum(actual.view(np.uint64)!=expected.view(np.uint64)))
        measurements[name]=dict(rows=len(expected),different_f64_bits=differences,predictions_sha256=sha(args.output/(name+'.tsv')))
        if differences:raise ValueError(name+' native inference differs from frozen responses')
    inputs=[args.tree_model,args.tree_parity,args.linear_model,args.linear_inputs,args.linear_reference]
    result=dict(passed=True,measurements=measurements,input_sha256={str(p):sha(p) for p in inputs},
                binary_sha256={name:sha(args.binaries/name) for name in ('corrhead_parity','bake_dial_refit')},
                driver_sha256=sha(__file__),training_fits_replayed=0)
    (args.output/'RESULT.json').write_bytes(encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binaries','tree-model','tree-parity','linear-model','linear-inputs','linear-reference','output'):p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(qualify(p.parse_args())),flush=True)
