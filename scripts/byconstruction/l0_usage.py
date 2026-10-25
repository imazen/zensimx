"""Usage: l0_usage.py ARM [ARM ...] (cell directories under ~/tmp/byc/cells; build inspect_l0_input_norms first).
Which fx1 inputs do the monotone models use? Layer-0 input norms (inspect_l0_input_norms) per cell, aggregated
by fx1 signal. A direct input (difference d) and its fragility product d x f count as one signal instance."""
import json, re, subprocess, sys
from collections import defaultdict
import numpy as np
INS = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "target/release/inspect_l0_input_norms")
E33 = json.load(open(__import__("pathlib").Path(__file__).resolve().parents[2] / "benchmarks/e33_registration_2026-10-09.json"))["fx1"]["table"]
CELLS = "/home/lilith/tmp/byc/cells"
FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")


def norms(bake):
    out = subprocess.run([INS, "--bake", bake, "--top", "5000"], capture_output=True, text=True, check=True).stdout
    n = int(re.search(r"n_inputs=(\d+)", out).group(1))
    v = np.zeros(n)
    top = out.split("## Top-")[1].split("## Bottom-")[0]
    for m in re.finditer(r"^\| (\d+) \| ([\d.]+) \|", top, re.M):
        v[int(m.group(1))] = float(m.group(2))
    return v / v.max()


def arm_usage(arm, n_direct):
    """Per difference j (0..409): fraction of the 12 cells whose direct, gated or product input is live (>= 1% max)."""
    live = np.zeros(410)
    for f in FOLDS:
        for s in (0, 1, 2):
            v = norms(f"{CELLS}/{arm}/without_{f}_s{s}/refit/last.bin")
            blocks = [v[i * 410:(i + 1) * 410] for i in range(len(v) // 410)]  # direct, [gated], products
            live += np.max(np.stack(blocks), 0) >= 0.01
    return live / 12


arms = sys.argv[1:]
use = {a: arm_usage(a, 410) for a in arms}
by_sig = defaultdict(lambda: defaultdict(list))
for j, (fid, sig, scale, ch, frag) in enumerate(E33):
    for a in arms:
        by_sig[sig][a].append(use[a][j])
print(f"{'signal':28s} {'n':>3s} " + " ".join(f"{a[:8]:>9s}" for a in arms) + "   (mean share of cells using each instance)")
rows = []
for sig, d in by_sig.items():
    rows.append((np.mean([np.mean(d[a]) for a in arms]), sig, d))
for _, sig, d in sorted(rows):
    print(f"{sig:28s} {len(d[arms[0]]):3d} " + " ".join(f"{np.mean(d[a]):9.2f}" for a in arms))
dead = {a: int((use[a] == 0).sum()) for a in arms}
print("differences unused in every cell:", dead)
common = np.all(np.stack([use[a] == 0 for a in arms]), 0)
print("unused in every cell of every arm:", int(common.sum()), sorted({E33[j][1] for j in np.where(common)[0]}))
