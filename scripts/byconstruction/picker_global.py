"""Items for the global-edit picker page: a person picks the less-bad image of a global-edit mixture step.

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Global edits in the mixture test"). The mixture test
assumes that adding noise on top of a global edit (brightness, contrast, saturation) always makes the image worse;
models that score the noisy version higher count an inversion. This page asks a person whether that assumption holds
where models disagree with it.

Selection (seeded, no labels involved): every step where B or any Tga cell raises the score, at most four per
(edit, noise) combination, those with the largest model rise first and distinct references where possible; plus
controls where no scored model raises it. Sides and order are shuffled with a fixed seed.

  python3 picker_global.py --out DIR [--per-combo 4] [--controls 11]

Writes DIR/items.json, copies the images to DIR/img/, and writes DIR/global_edit_picker.html: the page template
(picker_global.html) with every image embedded as a lossless WebP data URI, checked pixel-identical to its PNG.
"""

import argparse
import base64
import glob
import io
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

MIX = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/mixture_global")
MODELS = {  # page label, score files, units
    "B": ("B (served zensim)", "B", "points"),
    "Tga": ("Tga (descriptor sign-free)", "Tga", "points"),
    "Tga128": ("Tga128", "Tga128", "points"),
    "Bga": ("Bga (B's recipe + AIC)", "Bga", "points"),
    "M64": ("M64 (certified max pairs)", "M64", "points"),
    "Tgg": ("Tgg (gated damage)", "Tgg", "points"),
    "DVIFM03": ("DVIFM 0.3", "DVIFM03", "JND"),
}
EDIT = {"bright8": "brighter (+8 luma)", "dark8": "darker (−8 luma)", "contrast0.85": "lower contrast (×0.85)",
        "contrast1.15": "higher contrast (×1.15)", "sat0.8": "less saturated (×0.8)", "sat1.25": "more saturated (×1.25)"}
NOISE = {"luma4": "light luma noise (σ 4)", "luma8": "luma noise (σ 8)", "luma16": "heavy luma noise (σ 16)",
         "chroma8": "light chroma noise (σ 8)", "chroma16": "chroma noise (σ 16)", "chroma32": "heavy chroma noise (σ 32)"}


def load_scores(prefix: str) -> list[dict]:
    files = [MIX / f"scores_{prefix}.json"] if prefix in ("B", "DVIFM03") else sorted(
        Path(p) for p in glob.glob(str(MIX / f"scores_{prefix}_*.json")))
    return [json.loads(f.read_text())["scores"] for f in files]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-combo", type=int, default=4)
    ap.add_argument("--controls", type=int, default=11)
    ap.add_argument("--seed", type=int, default=20261024)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=False)
    rows = json.loads((MIX / "manifest.json").read_text())["rows"]
    scores = {k: load_scores(v[1]) for k, v in MODELS.items()}
    for r in rows:
        r["rise"] = {k: [s[r["more"]] - s[r["base"]] for s in v] for k, v in scores.items()}
        r["inverted_by"] = sorted(k for k, v in r["rise"].items() if any(x > 0 for x in v))
        r["key"] = max(r["rise"]["B"][0], sum(r["rise"]["Tga"]) / len(r["rise"]["Tga"]))
    rng = random.Random(a.seed)
    by = defaultdict(list)
    for r in rows:
        if r["rise"]["B"][0] > 0 or any(x > 0 for x in r["rise"]["Tga"]):
            by[(r["d1"], r["d2"])].append(r)
    chosen = []
    for combo in sorted(by):
        seen = set()
        for r in sorted(by[combo], key=lambda r: -r["key"]):
            if len([c for c in chosen if (c["d1"], c["d2"]) == combo]) >= a.per_combo:
                break
            if r["origin"] in seen:
                continue
            seen.add(r["origin"])
            chosen.append({**r, "kind": "inversion"})
    controls = [r for r in rows if not r["inverted_by"]]
    rng.shuffle(controls)
    taken = set()
    for r in controls:
        if len([c for c in chosen if c["kind"] == "control"]) >= a.controls:
            break
        if (r["d1"], r["d2"]) in taken:
            continue
        taken.add((r["d1"], r["d2"]))
        chosen.append({**r, "kind": "control"})
    rng.shuffle(chosen)
    items, files = [], set()
    for i, r in enumerate(chosen):
        noisy_is_b = rng.random() < 0.5
        a_img, b_img = (r["base"], r["more"]) if noisy_is_b else (r["more"], r["base"])
        for f in (r["ref"], r["base"], r["more"]):
            files.add(f)
        items.append({
            "id": f"g{i:02d}-{r['origin']}-{r['d1']}-{r['d2']}", "kind": r["kind"],
            "ref": f"img/{r['ref']}", "a": f"img/{a_img}", "b": f"img/{b_img}", "noisy": "b" if noisy_is_b else "a",
            "edit": EDIT[r["d1"]], "noise": NOISE[r["d2"]], "d1": r["d1"], "d2": r["d2"],
            "models": [{"id": k, "label": MODELS[k][0], "units": MODELS[k][2], "cells": len(v),
                        "mean_rise": round(sum(v) / len(v), 4), "cells_rising": sum(x > 0 for x in v)}
                       for k, v in r["rise"].items()],
        })
    for f in sorted(files):
        shutil.copyfile(MIX / "img" / f, out / "img" / f)
    (out / "items.json").write_text(json.dumps({"seed": a.seed, "source": "mixture_global (48 imazen-26 TRAIN references, "
                                                "256 px)", "items": items}, indent=1) + "\n")
    import numpy as np
    from PIL import Image
    images = {}
    for f in sorted(files):
        a = np.asarray(Image.open(out / "img" / f).convert("RGB"))
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "WEBP", lossless=True, quality=100, method=4, exact=True)
        if not np.array_equal(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")), a):
            raise ValueError(f"{f}: lossless WebP is not pixel-identical")
        images[f"img/{f}"] = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
    data = json.dumps({"items": items, "images": images}, separators=(",", ":")).replace("</", "<\\/")
    template = (Path(__file__).resolve().parent / "picker_global.html").read_text()
    (out / "global_edit_picker.html").write_text(template.replace("__PICKER_DATA__", data))
    print(json.dumps({"items": len(items), "inversion": sum(i["kind"] == "inversion" for i in items),
                      "control": sum(i["kind"] == "control" for i in items), "images": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
