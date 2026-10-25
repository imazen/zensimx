"""Can jxl-encoder's perceptual loop reach a requested distance when a zensim map steers it, and is a miss a placement
(fit) problem or a property of the map itself? Reads `steer_rd` outputs (zensim-workbench/loop-converge).
Exploration record: benchmarks/byconstruction_2026-10-22.md, "JPEG XL reachability with Tga".

  static  --rd DIR --label LABEL --probe TAG
      No-loop encodes over a dense distance grid, probed on the decoded output: for each requested distance t, the
      share of images whose probe reaches t at some encoder distance (largest 8x8 tile distance, and pixel maximum),
      the floor (the probe at the finest encoder distance), and how often the probe rises as the encoder distance falls.
  loop    --rd DIR --label LABEL [--label ...] [--probe TAG] [--exclude NAMES] [--by-class]
      Final reach per requested distance: the loop's own last frame (pixel max and largest tile, over the target) and,
      with --probe, the decoded output; blocks still above target. --exclude leaves out the images a placement was
      fitted on; --by-class adds each content class.
  elastic --rd DIR --label LABEL [--label ...]
      From the frames: each adjusted block's response ln(tile_next / tile) / ln(q_next / q). The loop multiplies an
      above-target block's quant field by tile / target, which assumes a response of -1; between 0 and -1 the loop
      undershoots each step, near 0 it cannot move the block. Unadjusted blocks' tile changes measure how much a
      block's distance depends on its neighbours.

TAG is a probe column tag `METRIC:MAP` as `steer_rd --probe workbench:METRIC:MAP` wrote it.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def table(rd: Path, label: str) -> pd.DataFrame:
    t = pd.read_csv(rd / "jxl" / f"loop_{label}.tsv", sep="\t")
    t["label"] = label
    return t


def cmd_static(a) -> int:
    t = table(Path(a.rd), a.label)
    tile, mx = f"probe_tile_max[{a.probe}]", f"probe_max[{a.probe}]"
    targets = [float(x) for x in a.targets.split(",")]
    out = {"label": a.label, "probe": a.probe, "images": int(t.image.nunique()), "targets": {}}
    rises = []
    floors_tile, floors_max = [], []
    per_image = {}
    for img, g in t.groupby("image"):
        g = g.sort_values("distance")
        per_image[img] = g
        floors_tile.append(g[tile].iloc[0])
        floors_max.append(g[mx].iloc[0])
        v = g[tile].to_numpy()
        rises.append(int((np.diff(v) < 0).sum()))  # probe falls while the encoder distance rises
    for x in targets:
        reach_tile = [bool((g[tile] <= x).any()) for g in per_image.values()]
        reach_max = [bool((g[mx] <= x).any()) for g in per_image.values()]
        # the encoder distance at which each image's tile probe first falls to x (interpolated in log space)
        enc = []
        for g in per_image.values():
            d, v = g.distance.to_numpy(), g[tile].to_numpy()
            ok = np.nonzero(v <= x)[0]
            if len(ok):
                i = ok[-1]
                if i + 1 < len(d) and v[i + 1] > x:
                    f = (np.log(x) - np.log(v[i])) / (np.log(v[i + 1]) - np.log(v[i]))
                    enc.append(float(np.exp(np.log(d[i]) + f * (np.log(d[i + 1]) - np.log(d[i])))))
                else:
                    enc.append(float(d[i]))
        out["targets"][x] = {
            "reach_tile": float(np.mean(reach_tile)), "reach_max": float(np.mean(reach_max)),
            "encoder_distance_median": float(np.median(enc)) if enc else None,
            "encoder_distance_p10_p90": [float(np.percentile(enc, 10)), float(np.percentile(enc, 90))] if enc else None,
        }
    out["floor_tile_median_p90"] = [float(np.median(floors_tile)), float(np.percentile(floors_tile, 90))]
    out["floor_max_median_p90"] = [float(np.median(floors_max)), float(np.percentile(floors_max, 90))]
    out["finest_encoder_distance"] = float(t.distance.min())
    out["images_with_a_nonmonotone_step"] = int(sum(r > 0 for r in rises))
    print(json.dumps(out, indent=1))
    return 0


def cmd_loop(a) -> int:
    rows = []
    drop = set(Path(a.exclude).read_text().split()) if a.exclude else set()
    for label in a.label:
        t = table(Path(a.rd), label)
        t = t[~t.image.isin(drop)]
        if a.by_class:
            t = pd.concat([t.assign(group="all"), t.assign(group=t["class"])])
        else:
            t = t.assign(group="all")
        for (cls, d), g in t.groupby(["group", "distance"]):
            r = {"label": label, "class": cls, "distance": d, "n": len(g),
                 "loop_max_over_target": float(np.median(g.last_score / g.last_target)),
                 "loop_tile_over_target": float(np.median(g.last_tile_max / g.last_target)),
                 "loop_blocks_above_median": float(np.median(g.last_blocks_above)),
                 "loop_converged_share": float(np.mean(g.last_blocks_above == 0)),
                 "bytes_median": float(np.median(g.bytes))}
            if a.probe:
                tile, mx, ab = (f"probe_tile_max[{a.probe}]", f"probe_max[{a.probe}]", f"probe_above[{a.probe}]")
                r.update({"out_max_over_d": float(np.median(g[mx] / d)), "out_tile_over_d": float(np.median(g[tile] / d)),
                          "out_tile_over_d_p90": float(np.percentile(g[tile] / d, 90)),
                          "out_reached_share": float(np.mean(g[tile] <= d * (1 + a.tol)))})
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def frames(path: Path):
    b = path.read_bytes()
    o, out = 0, []
    while o < len(b):
        it, = struct.unpack_from("<I", b, o)
        seed, score, target = struct.unpack_from("<3d", b, o + 4)
        n, = struct.unpack_from("<I", b, o + 28)
        o += 32
        arr = np.frombuffer(b, dtype="<f4", count=3 * n, offset=o).reshape(3, n)
        o += 12 * n
        out.append({"iter": it, "seed": seed, "score": score, "target": target, "tile": arr[0], "q": arr[1],
                    "q_next": arr[2]})
    return out


def cmd_elastic(a) -> int:
    rows = []
    for label in a.label:
        run = json.loads((Path(a.rd) / "jxl" / f"run_{label}.json").read_text())
        fdir = Path(run["frames_dir"])
        by_d = {}
        for p in sorted(fdir.glob(f"{label}__*.frames")):
            d = float(p.name.rsplit("__d", 1)[1].removesuffix(".frames"))
            fr = frames(p)
            for f0, f1 in zip(fr, fr[1:]):
                if f1["seed"] != f0["seed"] or f1["iter"] != f0["iter"] + 1:
                    continue
                q0, q1 = f0["q"].astype(np.float64), f1["q"].astype(np.float64)
                if not np.array_equal(f0["q_next"], f1["q"]):
                    raise SystemExit(f"{p.name}: frame {f1['iter']} quant field is not frame {f0['iter']}'s next")
                t0, t1 = f0["tile"].astype(np.float64), f1["tile"].astype(np.float64)
                ok = (t0 > 0) & (t1 > 0)
                lq = np.log(q1 / q0)
                lt = np.log(np.where(ok, t1, 1.0) / np.where(ok, t0, 1.0))
                above = t0 > f0["target"]
                moved = ok & (np.abs(lq) > a.min_step)
                still = ok & (lq == 0)
                e = by_d.setdefault(d, {"up": [], "down": [], "still": [], "above_up": []})
                e["up"].append(lt[moved & (lq > 0)] / lq[moved & (lq > 0)])
                e["down"].append(lt[moved & (lq < 0)] / lq[moved & (lq < 0)])
                e["above_up"].append(lt[moved & (lq > 0) & above] / lq[moved & (lq > 0) & above])
                e["still"].append(lt[still])
        for d, e in sorted(by_d.items()):
            cat = {k: np.concatenate(v) if v else np.array([]) for k, v in e.items()}

            def q(v, p):
                return float(np.percentile(v, p)) if len(v) else float("nan")

            rows.append({"label": label, "distance": d,
                         "response_bumped_median": q(cat["above_up"], 50),
                         "response_bumped_p25_p75": f"{q(cat['above_up'], 25):.2f}..{q(cat['above_up'], 75):.2f}",
                         "n_bumped": len(cat["above_up"]),
                         "response_coarsened_median": q(cat["down"], 50), "n_coarsened": len(cat["down"]),
                         "unadjusted_abs_log_change_median": q(np.abs(cat["still"]), 50),
                         "n_unadjusted": len(cat["still"])})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--rd", required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--probe", required=True)
    s.add_argument("--targets", default="0.3,0.45,0.6,1.0,1.6,2.5,4.0")
    lp = sub.add_parser("loop")
    lp.add_argument("--rd", required=True)
    lp.add_argument("--label", action="append", required=True)
    lp.add_argument("--probe")
    lp.add_argument("--tol", type=float, default=0.05)
    lp.add_argument("--exclude", help="file of image names to leave out (e.g. the placement fit's images)")
    lp.add_argument("--by-class", action="store_true")
    e = sub.add_parser("elastic")
    e.add_argument("--rd", required=True)
    e.add_argument("--label", action="append", required=True)
    e.add_argument("--min-step", type=float, default=0.02)
    a = ap.parse_args()
    return {"static": cmd_static, "loop": cmd_loop, "elastic": cmd_elastic}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
