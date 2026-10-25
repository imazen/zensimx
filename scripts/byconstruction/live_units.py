"""Count live hidden units (nonzero output weight), threshold use and W1 sign in trainer bakes (zenpredict-inspect)."""
import json, subprocess, sys, numpy as np
INS = '/home/lilith/work/zen/zenanalyze/target/release/zenpredict-inspect'
for p in sys.argv[1:]:
    d = json.loads(subprocess.run([INS, p, '--weights'], capture_output=True, text=True, check=True).stdout)
    L = d['layers']; H = L[0]['out_dim']
    W1 = np.array(L[0]['weights']).reshape(L[0]['in_dim'], H); b1 = np.array(L[0]['biases']); w2 = np.array(L[1]['weights'])
    live = w2 != 0
    print(f"{p.split('cells/')[-1][:60]:60s} live={live.sum():3d}/{H} neg_w2={int((w2<0).sum())} pos_w2={int((w2>0).sum())} "
          f"live_thr<0={int((b1[live]<0).sum())} W1min={W1.min():+.3f}")
