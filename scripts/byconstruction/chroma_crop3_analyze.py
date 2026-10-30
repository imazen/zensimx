#!/usr/bin/env python3
"""Analyze chroma_crop3.py output: full frame vs 3-px edge crop, per family and view size."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

df = pd.DataFrame([json.loads(l) for l in open(sys.argv[1]) if l.strip()])
df["size"] = pd.cut(df[["width", "height"]].max(axis=1), [0, 64, 128, 256, 512], labels=["<=64", "65-128", "129-256", "257-512"])
df["ladder"] = df.view_id + "|" + df.family
print(f"rows {len(df)}, ladders {df.ladder.nunique()}, teacher full bits == stored: {(df.teacher_full_bits == df.teacher_stored_bits).mean():.4f}")


def ladder_stats(g, cond, model):
    t, b, m = g[f"teacher_{cond}"].values, -g[f"butter_p3_{cond}"].values, g[f"{model}_{cond}"].values
    rho = spearmanr(m, t)[0] if len(g) >= 3 and np.ptp(t) > 0 and np.ptp(m) > 0 else np.nan
    inv = agree = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            st, sb = np.sign(t[i] - t[j]), np.sign(b[i] - b[j])
            if st != 0 and st == sb:
                agree += 1
                inv += np.sign(m[i] - m[j]) == -st
    return rho, inv, agree


def table(by, model):
    out = []
    for key, g in df.groupby(by, observed=True):
        r = dict(group=key, rows=len(g), ladders=g.ladder.nunique())
        for cond in ("full", "crop"):
            r[f"MAE_{cond}"] = np.abs(g[f"{model}_{cond}"] - g[f"teacher_{cond}"]).mean()
            st = [ladder_stats(l, cond, model) for _, l in g.groupby("ladder")]
            r[f"rho_{cond}"] = np.nanmean([s[0] for s in st])
            r[f"inv_{cond}"] = sum(s[1] for s in st); r[f"agree_{cond}"] = sum(s[2] for s in st)
        out.append(r)
    t = pd.DataFrame(out)
    t["dMAE"] = t.MAE_crop - t.MAE_full
    return t


pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
for model in ("E", "ECodec"):
    print(f"\n== {model}: by family")
    print(table("family", model).round(3).to_string(index=False))
    print(f"\n== {model}: by view size")
    print(table("size", model).round(3).to_string(index=False))
    print(f"\n== {model}: family x size, MAE full -> crop")
    t = table(["family", "size"], model)
    print(t[["group", "rows", "MAE_full", "MAE_crop", "dMAE", "inv_full", "inv_crop"]].round(3).to_string(index=False))

print("\n== how much each metric moves under the crop: mean |crop - full|, and mean (crop - full)")
for key, g in df.groupby("family"):
    s = {m: (np.abs(g[f"{m}_crop"] - g[f"{m}_full"]).mean(), (g[f"{m}_crop"] - g[f"{m}_full"]).mean()) for m in ("teacher", "E", "ECodec", "butter_p3")}
    print(key, {k: (round(a, 3), round(b, 3)) for k, (a, b) in s.items()})
print("\n== by size")
for key, g in df.groupby("size", observed=True):
    s = {m: (np.abs(g[f"{m}_crop"] - g[f"{m}_full"]).mean(), (g[f"{m}_crop"] - g[f"{m}_full"]).mean()) for m in ("teacher", "E", "ECodec", "butter_p3")}
    print(key, {k: (round(a, 3), round(b, 3)) for k, (a, b) in s.items()})
