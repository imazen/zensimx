"""R12 L0 strict score-bit checks on the registered pixel-gate outputs."""
import json
from pathlib import Path
import struct


def u64(value):
    return struct.unpack('<Q',struct.pack('<d',value))[0]


def main():
    r=Path('/var/tmp/byc-linear-20261104/extra/reads');h=Path.home()/'tmp/e36test'
    report={}
    for s in range(3):
        name=f'L0_s{s}'
        a=json.load(open(h/f'gates/mixF3_s{s}/steer135/mixF3_s{s}/result.json'))['rows']
        b=json.load(open(r/f'gates/{name}/steer135/{name}/result.json'))['rows']
        if len(a)!=135 or len(b)!=135:raise ValueError('complete G-STEER packet required')
        for x,y in zip(a,b):
            assert x['key']==y['key']
            for k in ('served_base_bits','served_repair_bits'):
                if x[k]!=y[k]:raise ValueError(f'seed {s}, {x["key"]}, {k}: differs')
        def identity(path):
            rows={}
            for line in path.read_text().splitlines():
                d=json.loads(line)
                if d.get('summary'):continue
                key=(d['permutation'],d['reference'])
                rows[key]=d['candidate_score_bits'][0]
            if len(rows)!=744:raise ValueError('all 62 sources and 12 identity permutations required')
            return rows
        ia=identity(h/f'gates/mixF3_s{s}-vi/identity-s0.jsonl')
        ib=identity(r/f'gates/{name}/identity-s0.jsonl')
        if ia!=ib:raise ValueError(f'seed {s}: identity score bits differ')
        def near(path):
            d=json.load(open(path));out={}
            if d['rows']!=648:raise ValueError('complete 648-row NEARID fixture required')
            for ref in d['references']:
                for ladder,v in ref['ladders'].items():
                    for rung,score in zip(v['rungs'],v['scores']):out[(ref['reference'],ladder,rung)]=u64(score)
            return out
        na=near(h/f'nearid/out-mixf3s{s}/candidate-mixf3s{s}.GATES.json')
        nb=near(r/f'nearid/{name}/candidate-l0s{s}.GATES.json')
        if len(na)!=768 or na!=nb:raise ValueError(f'seed {s}: NEARID score bits differ')
        report[name]=dict(steer_cases=135,steer_base_and_repair_bits_equal=True,
                          identity_score_bits=744,nearid_rows=648,nearid_ladder_score_bits=len(na),all_equal=True)
    dest=Path('/var/tmp/byc-linear-20261104/extra/proof/PIXEL_PROOF.json')
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
