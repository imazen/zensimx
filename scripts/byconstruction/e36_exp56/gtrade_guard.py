"""G-TRADE (a) readout from a trade_gate TALLIES.json: per model, inversions over the pairs both references call worse, in the 'all/regular' stratum
(gauntlet and training-leg views above 64 px), pooled over sources.   gtrade_guard.py TALLIES.json MODEL..."""
import json, sys
d = json.load(open(sys.argv[1]))
for stratum in ('all/regular', 'gauntlet/regular'):
    a = d[stratum]['a']
    out = []
    for m in sys.argv[2:]:
        t = a[m]
        # the trade pairs where both references call the trade view worse (n_worse); inv = the model scores it better
        pairs = sum(v['n_worse'] for v in t.values()); inv = sum(v['inv'] for v in t.values())
        out.append(f"{m} {inv / pairs:.4f}")
    print(f"{stratum} pairs {pairs} sources {len(t)}: " + '  '.join(out))
