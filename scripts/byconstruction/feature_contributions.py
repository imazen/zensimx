"""Per-feature-family share of distance in a (near-)linear monotone cell on KADID, for global-change types vs all types."""
import json, subprocess, numpy as np, pyarrow.parquet as pq, collections
INS='/home/lilith/work/zen/zenanalyze/target/release/zenpredict-inspect'
import glob
bake=glob.glob('/home/lilith/tmp/byc/cells/mono-thr/without_kadid_s0/dense/*.bin')[0]
d=json.loads(subprocess.run([INS,bake,'--weights'],capture_output=True,text=True,check=True).stdout)
L=d['layers']; H=L[0]['out_dim']; W1=np.array(L[0]['weights']).reshape(L[0]['in_dim'],H); w2=np.array(L[1]['weights']); b1=np.array(L[0]['biases'])
live=np.where(w2!=0)[0]; print('live units', live, 'bias', b1[live])
scale=np.array(d['scaler_scale'])
der=[l.split() for l in [m for m in d['metadata'] if m['key']=='zensim.derived_inputs'][0]['value_text'].splitlines()[1:]]
ids=sorted({int(t) for r in der for t in r[1:]})
tab=pq.read_table('/mnt/v/output/zensim/v40r4-2026-10-08/v2e29/wide/main/real/kadid.parquet',columns=[f'f{i}' for i in ids])
cols={i:tab[f'f{i}'].to_numpy().astype(float) for i in ids}
X=np.stack([cols[int(r[1])] if r[0]=='in' else cols[int(r[1])]*cols[int(r[2])] for r in der],1)/scale
C=-(X*W1[:,live].sum(1))*w2[live].sum()   # per-input distance contribution (>=0), linear unit
v2="ssim_mean ssim_dev2 ssim_dev4 art det mse hf_gain hf_loss hf_mag_loss ssim_soft_peak art_soft_peak det_soft_peak masked_ssim masked_art masked_det masked_mse iw_ssim iw_art iw_det iw_mse pjnd_transducer pjnd_fragility gms transducer_lowk transducer_highk blockiness ringing banding edge_width".split()
def fam(r):
    i=int(r[1])
    if i<156: return f"basic.slot{i%13}"
    k=(i-372)%29; return "v2."+(v2[k] if k<len(v2) else str(k))
fams=[fam(r) for r in der]
panel=json.load(open('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json'))
k=[x for x in panel['datasets'] if x['name']=='kadid'][0]; cod=np.array(k['codec'])+1
def share(mask):
    tot=collections.Counter()
    for j,f in enumerate(fams): tot[f]+=C[mask,j].sum()
    s=sum(tot.values()); return {f:v/s for f,v in tot.items()}
allm=np.ones(len(cod),bool)
for name,types in (('contrast change',[25]),('saturation 1+2',[7,8]),('mean shift/brighten/darken',[16,17,18])):
    m=np.isin(cod,types); a=share(allm); b=share(m)
    top=sorted(b, key=lambda f: -(b[f]-a.get(f,0)))[:6]
    print(f"== {name}: mean distance {C[m].sum(1).mean():.1f} vs all-type mean {C.sum(1).mean():.1f}")
    for f in top: print(f"   {f:22s} share {b[f]:.3f} (all types {a.get(f,0):.3f})")
