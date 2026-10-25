"""Information ceiling on AIC2026: source-held-out boosting on fx1g, on all supplied peer metric columns (DVIFM
excluded: author constants trained on AIC2026), and on both. Diagnostic only; peers never become model inputs.
Usage: aic_allpeers.py PAIRS.tsv FOLDS OUT.json"""
import sys, numpy as np, pandas as pd, json
sys.path.insert(0,'scripts/byconstruction')
import aic_train as A
from scipy.stats import spearmanr
O='/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train'
t,p=A.load(O+'/table_hpm/table.parquet',sys.argv[1] if len(sys.argv)>1 else '/home/lilith/tmp/aictrain/pairs_curve.tsv')
X,names,C,cn=A.fx1(t,True)
y=p.jnd.to_numpy(float); g=p.group.to_numpy()
m=pd.read_csv(A.PEER_TABLE,low_memory=False); m=m[m.view=='cropped']; m.index=m.distorted.str.removesuffix('.png')
skip={'view','ptc_subset','distorted','source','codec','codec_acronym','distortion_level','bpp','codec_id','enc_setting','enc_value'}
cols=[]
for c in m.columns:
    if c in skip or c.startswith('JND_') or c.startswith('proposal-DVIFM'): continue  # DVIFM author constants trained on AIC2026
    v=pd.to_numeric(m.loc[p.key.astype(str),c],errors='coerce').to_numpy(float)
    if np.isnan(v).any(): continue
    v=np.sign(spearmanr(v,y).correlation)*v; cols.append(v-v.min())
P=np.stack(cols,1); print('peer columns', P.shape[1])
res=[]
for name,Xa in [('fx1g only',X),('peers only',P),('fx1g+all peers',np.c_[X,P])]:
    for model in ('hgbmono','hgb'):
        o=A.oof_predict(model,Xa,C,y,g,int(sys.argv[2]) if len(sys.argv)>2 else 6,1000)
        r=spearmanr(o,y).correlation; res.append({'inputs':name,'model':model,'oof_global':float(r)})
        print(f'{name:16s} {model:8s} OOF global {r:.4f}', flush=True)
if len(sys.argv)>3:
    import json; json.dump({'pairs':sys.argv[1],'folds':int(sys.argv[2]),'peer_columns':P.shape[1],'results':res},open(sys.argv[3],'w'),indent=1)
