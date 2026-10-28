import json, os, shlex, sys
FE='/mnt/v/output/zensim/reports/fulleval'
ROOT='/mnt/v/output/zensim/replication-2026-09-05'
TRAINER='/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train'
# recipe -> (source fulleval cell, anchor seed S0)
REC={'LSTAR':('LSTAR_s4021_packed',4021),'LSTAR3':('LSTAR3_s4041_packed',4041),
     'A5_r4':('A5_r4_s4004',4004),'A3b':('A3b_s4004',4004)}
SAMPLE_SEEDS=[5001,5002]; INIT_SEEDS=[5011,5012]
fits=[]
def build(tag, cell, init, samp, legacy=None):
    d=json.load(open(os.path.join(FE,cell+'.fulleval.json')))
    argv=[str(x) for x in d['repro']['argv']]
    argv[0]=TRAINER
    # replace seed flags
    out=[];i=0
    while i<len(argv):
        a=argv[i]
        if a=='--seed': i+=2; continue
        if a in ('--init-seed','--sample-seed'): i+=2; continue
        if a=='--out': out+= ['--out', f'{ROOT}/bakes/{tag}.bin']; i+=2; continue
        if a=='--dump-checkpoints-dir': out+= ['--dump-checkpoints-dir', f'{ROOT}/ckpts/{tag}_ckpts']; i+=2; continue
        out.append(a); i+=1
    if legacy is not None: out += ['--seed', str(legacy)]
    else: out += ['--init-seed', str(init), '--sample-seed', str(samp)]
    fits.append({'tag':tag,'cell':cell,'arm':tag.split('__')[1] if '__' in tag else 'ctl',
                 'init':init,'sample':samp,'legacy':legacy,'argv':out,
                 'out':f'{ROOT}/bakes/{tag}.bin','ckpt':f'{ROOT}/ckpts/{tag}_ckpts'})
# controls first
build('CTL_A_LSTAR_s4021_legacy','LSTAR_s4021_packed',4021,4021,legacy=4021)
build('CTL_B_LSTAR_s4021_split','LSTAR_s4021_packed',4021,4021)
for rec,(cell,s0) in REC.items():
    for s in SAMPLE_SEEDS:
        build(f'{rec}__S__i{s0}_p{s}', cell, s0, s)
    if rec!='A3b':
        for m in INIT_SEEDS:
            build(f'{rec}__I__i{m}_p{s0}', cell, m, s0)
json.dump(fits, open('/home/lilith/tmp/replicate/fits.json','w'), indent=1)
print("planned fits:", len(fits))
for f in fits: print("  %-32s arm=%-4s init=%-5s sample=%-5s legacy=%s" % (f['tag'],f['arm'],f['init'],f['sample'],f['legacy']))
