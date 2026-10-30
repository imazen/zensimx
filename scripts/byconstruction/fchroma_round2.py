#!/usr/bin/env python3
"""Profile F round 2: gather the registered selection evidence into one private JSON (it holds no AIC-2026 numbers).

  fchroma_round2.py --out ~/tmp/handoff/f-chroma/results_round2.json
Selection rule: benchmarks/byconstruction_2026-10-22.md, "Profile F round 2". Reads finished artifacts only.
"""
import argparse
import json
from pathlib import Path

S = Path("/var/tmp/byc-fchroma-20261029")
ST = S / "stage2/gates"
ARMS = {"W100": ("p1fx2", 1.0), "W50": ("r2w50", 0.5), "W25": ("r2w25", 0.25), "W12": ("r2w12", 0.125)}
BAR_MEAN, BAR_EACH = -0.002, -0.005


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else "not_run"


def e21(arm_dir):
    d = load(S / arm_dir / "scorecard/e21.json")
    if d == "not_run":
        return d
    a, b = d["Mlic"]["per_dataset_srocc"], d["Mlicfx2"]["per_dataset_srocc"]
    diff = {k: b[k] - a[k] for k in a}
    mean = sum(diff.values()) / len(diff)
    return dict(per_population_srocc=b, stored_mlic_srocc=a, delta=diff, mean_delta=mean,
                passes=bool(mean >= BAR_MEAN and all(v >= BAR_EACH for v in diff.values())))


def gsteer(path):
    r = load(path)
    if r == "not_run":
        return r
    rows = r["rows"]
    return dict(passed=sum(x["pass"] for x in rows), total=len(rows), failing=[x["key"] for x in rows if not x["pass"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    level = {}
    for f in (S / "gchroma_r2_dev/G_CHROMA_L.json", S / "gchroma_r2_dev12/G_CHROMA_L.json"):
        d = load(f)
        if d != "not_run":
            level |= d["candidates"]
    arms = {}
    for name, (d, scale) in ARMS.items():
        lv = level.get(name)
        steer = gsteer(ST / "r2-W12/steer135/r2W12/result.json") if name == "W12" else "not_run"
        e = e21(d)
        arms[name] = dict(
            weight_scale=scale, nominal_weight=scale, train_weight=1.0410771004695145 * scale, study=str(S / d),
            e21=e,
            g_chroma_l_dev=None if lv is None else dict(passes=lv["passes"], per_family=lv["per_family"]),
            rank_non_regression_dev=None if lv is None else lv["rank_non_regression"],
            g_steer_seed0=steer,
            passes_all=bool(lv and lv["passes"] and e != "not_run" and e["passes"] and isinstance(steer, dict) and steer["passed"] == 135),
        )
    chosen = [n for n in ("W100", "W50", "W25", "W12") if arms[n]["passes_all"]]
    out = dict(
        schema="fchroma-round2-v1",
        registration="benchmarks/byconstruction_2026-10-22.md, Profile F round 2 (registered 2026-10-29 after round-1 results were seen; commit 14f8ad01)",
        bars=dict(e21_mean_delta_min=BAR_MEAN, e21_each_population_delta_min=BAR_EACH, g_steer="135/135", g_chroma_l="per family candidate MAE < E and bootstrap upper < 0 (dev ladders)"),
        arms=arms, selected=(chosen[0] if chosen else None),
        verdict="negative: no arm passes G-STEER 135/135, the E21 no-regression bar and G-CHROMA-L on dev together" if not chosen else f"select {chosen[0]}",
        notes=["G-STEER was run only for the arm that passes G-CHROMA-L and could still pass E21 (W12); W100/W50/W25 fail E21 and/or the level gate, so it cannot change the selection",
               "W100 is the round-1 chroma-leg weight (Mlic-recipe seed-0 model)",
               "calibration ladders were not read for any candidate (final gate is for a chosen weight only)"],
        trainer=dict(round2_cells_and_fits="4bca1062 (main bad8e5f6)", note="low-memory trainer 10d087cd was not used for any scored round-2 model"),
    )
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for n, v in arms.items():
        e = v["e21"]
        print(n, "L", None if v["g_chroma_l_dev"] is None else v["g_chroma_l_dev"]["passes"], "E21 mean", None if e == "not_run" else round(e["mean_delta"], 4), "steer", v["g_steer_seed0"] if v["g_steer_seed0"] == "not_run" else f'{v["g_steer_seed0"]["passed"]}/135', "ALL", v["passes_all"])
    print("selected", out["selected"])


if __name__ == "__main__":
    main()
