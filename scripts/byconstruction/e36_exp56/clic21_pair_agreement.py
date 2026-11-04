"""Why the CLIC 2021 validation agreement curve zigzags (owner question, 2026-11-03; record, R10 amendment 5 result).

For each model or judge score in a CLIC 2021 confirmation dir: 2AFC agreement per |score difference| bin (equal-count
bins, with the binomial standard error), the same restricted to the questions several raters answered, the share of
each bin where the library's PSNR label orders the pair the same way, and the reference metrics' sign agreement on the
same questions. Optionally writes a gallery of example questions (original, A, B, every score, the votes) under
/mnt/v/output for viewing at http://dev.lan:3300/.

    clic21_pair_agreement.py CONFIRM_DIR [--scores control_s0,mixF3_s0] [--bins 8] [--gallery OUT_DIR]
                             [--paired control_s0:mixF3_s0,control_s1:mixF3_s1,...]

--paired compares two scores on the questions near a tie (where the mix's curve zigzags): each score's own near-tie
band (|difference| < 2.4), the union band, agreement with the raters, with PSNR and with SSIMULACRA2 there, and the
questions the two scores order differently with a bootstrap interval over questions.
"""
import argparse
import csv
import html
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

LIB = Path("/mnt/tower/input/iqa/datasets/clic-2021-perceptual")
JUDGES = ("ssim2", "butteraugli", "fsimc", "haar_mask")


def load_confirm(d):
    keys = pq.read_table(d / "features.parquet", columns=["ref_basename"]).column(0).to_pylist()
    rows = json.loads((d / "eval_rows.json").read_text())
    pairs = [r for r in rows if r["kind"] == "pair"]
    with open(d / "judge_pairs.tsv") as f:
        jp = list(csv.DictReader(f, delimiter="\t"))
    path = {r["pair_key"]: r["dist_path"] for r in jp}
    refpath = {r["pair_key"]: r["ref_path"] for r in jp}

    def pred(stem):
        with open(d / f"pred_{stem}.tsv") as f:
            return dict(zip(keys, [float(r["pred"]) for r in csv.DictReader(f, delimiter="\t")]))

    def judge(name):
        with open(d / f"judge_{name}.tsv") as f:
            return {r["pair_key"]: float(r[name]) for r in csv.DictReader(f, delimiter="\t")}

    return pairs, path, refpath, pred, judge


def psnr_labels():
    """(ref, a, b) -> True when a has the higher PSNR (the library README: 0 marks the first image)."""
    out = {}
    with open(LIB / "labels/valid/clic2021_perceptual_valid_psnr.csv") as f:
        for r in csv.reader(f):
            out[(r[0], r[1], r[2])] = r[3] == "0"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("confirm_dir")
    ap.add_argument("--scores", default="control_s0,mixF3_s0")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--gallery")
    ap.add_argument("--paired", help="BASE:CHALLENGER pairs of score stems, comma separated")
    ap.add_argument("--sharpness", nargs="?", const=4, type=int, metavar="N",
                    help="test the sharper-crop reading: high-frequency energy of each crop against its original, "
                         "band statistics, and N zoomed composites of near-tie questions under --gallery")
    a = ap.parse_args()
    d = Path(a.confirm_dir)
    pairs, path, refpath, pred, judge = load_confirm(d)
    t = np.array([r["target"] for r in pairs])
    n = np.array([r.get("n", 1) for r in pairs])
    J = {j: judge(j) for j in JUDGES}
    psnr = psnr_labels()
    bn = lambda k: os.path.basename(path[k])  # noqa: E731

    def psnr_prefers_a(r):
        key = (os.path.basename(r["group"]), bn(r["a"]), bn(r["b"]))
        if key in psnr:
            return psnr[key]
        key = (key[0], key[2], key[1])
        return (not psnr[key]) if key in psnr else None

    pa = [psnr_prefers_a(r) for r in pairs]
    print(f"{len(pairs)} questions; PSNR label joined for {sum(x is not None for x in pa)}; "
          f"{int((n >= 2).sum())} answered by two or more raters")
    for stem in a.scores.split(","):
        s = pred(stem)
        diff = np.array([s[r["a"]] - s[r["b"]] for r in pairs])
        ad = np.abs(diff)
        agree = np.where(diff > 0, t, np.where(diff < 0, 1 - t, 0.5))
        psnr_same = np.array([x is not None and ((x and dd > 0) or ((not x) and dd < 0)) for x, dd in zip(pa, diff)])
        edges = np.quantile(ad, np.linspace(0, 1, a.bins + 1))
        print(f"== {stem}: 2AFC agreement by |difference| bin (equal counts)")
        print("| |difference| | n | agreement | s.e. | n (2+ raters) / agreement | PSNR orders the pair the same | "
              + " | ".join(f"{j} sign agreement" for j in JUDGES) + " |")
        print("|---|---:|---:|---:|---|---:|" + "---:|" * len(JUDGES))
        for i in range(a.bins):
            sel = (ad >= edges[i]) & ((ad < edges[i + 1]) if i < a.bins - 1 else (ad <= edges[i + 1]))
            p = agree[sel].mean()
            se = np.sqrt(p * (1 - p) / sel.sum())
            multi = sel & (n >= 2)
            js = []
            for j in JUDGES:
                dj = np.array([J[j][r["a"]] - J[j][r["b"]] for r, k in zip(pairs, sel) if k])
                if j == "butteraugli":
                    dj = -dj  # stored as a distance
                js.append(f"{np.mean(np.where(dj > 0, t[sel], np.where(dj < 0, 1 - t[sel], 0.5))):.3f}")
            print(f"| {edges[i]:.1f}-{edges[i + 1]:.1f} | {sel.sum()} | {p:.3f} | {se:.3f} | "
                  f"{multi.sum()} / {agree[multi].mean():.3f} | {psnr_same[sel].mean():.2f} | " + " | ".join(js) + " |")
    if a.paired:
        paired(pairs, t, pred, J, pa, [tuple(x.split(":")) for x in a.paired.split(",")])
    if a.sharpness is not None:
        sharpness(d, pairs, path, refpath, pred, J, pa, t, n, a.scores.split(","),
                  Path(a.gallery) if a.gallery else None, a.sharpness)
    if a.gallery:
        write_gallery(Path(a.gallery), pairs, path, refpath, pred, J, pa, t, n, a.scores.split(","))


def paired(pairs, t, pred, J, pa, couples, band=2.4):
    rng = np.random.default_rng(0)
    dp = np.array([0.0 if x is None else (1.0 if x else -1.0) for x in pa])
    ds2 = np.array([J["ssim2"][r["a"]] - J["ssim2"][r["b"]] for r in pairs])
    hum = lambda d, sel: np.mean(np.where(d[sel] > 0, t[sel], np.where(d[sel] < 0, 1 - t[sel], 0.5)))  # noqa: E731
    for base, chal in couples:
        D = {}
        for stem in (base, chal):
            s = pred(stem)
            D[stem] = np.array([s[r["a"]] - s[r["b"]] for r in pairs])
        print(f"== {base} vs {chal}: mean difference {D[base].mean():+.2f} / {D[chal].mean():+.2f}, "
              f"share positive {np.mean(D[base] > 0):.3f} / {np.mean(D[chal] > 0):.3f}, raters' share for a {t.mean():.3f}")
        for stem in (base, chal):
            own = np.abs(D[stem]) < band
            print(f"   {stem}: own band |d|<{band} n={own.sum()} agreement {hum(D[stem], own):.3f} "
                  f"(s.e. {np.sqrt(0.25 / own.sum()):.3f}), sign = PSNR {np.mean(np.sign(D[stem][own]) == dp[own]):.3f}, "
                  f"sign = SSIMULACRA2 {np.mean(np.sign(D[stem][own]) == np.sign(ds2[own])):.3f}")
        both = (np.abs(D[base]) < band) | (np.abs(D[chal]) < band)
        for stem in (base, chal):
            print(f"   union band n={both.sum()}: {stem} agreement {hum(D[stem], both):.3f}, sign = PSNR "
                  f"{np.mean(np.sign(D[stem][both]) == dp[both]):.3f}, sign = SSIMULACRA2 {np.mean(np.sign(D[stem][both]) == np.sign(ds2[both])):.3f}")
        dis = np.sign(D[base]) != np.sign(D[chal])
        idx = np.where(dis)[0]
        diffs = []
        for _ in range(2000):
            b = rng.choice(idx, len(idx))
            diffs.append(hum(D[chal], b) - hum(D[base], b))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"   ordered differently: n={dis.sum()} ({dis.mean():.1%}); raters with {base} {hum(D[base], dis):.3f}, "
              f"with {chal} {hum(D[chal], dis):.3f}, difference {hum(D[chal], dis) - hum(D[base], dis):+.3f} [{lo:+.3f}, {hi:+.3f}]")


def luma(p):
    """BT.601 luma of an 8-bit sRGB PNG as float64 (a sharpness proxy, no colour management involved)."""
    from PIL import Image
    im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
    return 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]


def hf_measures(cache, p):
    """Per image: high-pass energy (luma minus a sigma-1.5 Gaussian), Laplacian variance, gradient energy."""
    from scipy import ndimage
    if p not in cache:
        y = luma(p)
        hp = y - ndimage.gaussian_filter(y, 1.5)
        gx, gy = ndimage.sobel(y, 0), ndimage.sobel(y, 1)
        cache[p] = dict(hp=float(np.mean(hp * hp)), lap=float(np.var(ndimage.laplace(y))),
                        grad=float(np.mean(gx * gx + gy * gy)))
    return cache[p]


def hp_image(p):
    from scipy import ndimage
    y = luma(p)
    return y, y - ndimage.gaussian_filter(y, 1.5)


def sharpness(d, pairs, path, refpath, pred, J, pa, t, n, stems, gallery, n_examples, band=2.4):
    """Is the crop the raters prefer the sharper one, and does the mix side against it more than F3?

    sharp(crop) = its high-pass energy divided by the original's (1 = as much fine detail as the original, below 1 =
    smoothed, above 1 = added texture or noise); texture fidelity = correlation of the crop's high-pass with the
    original's. Stems: the first two are the base (F3) and the challenger (the mix)."""
    cache_path = d / "sharpness_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    files = sorted({refpath[r["a"]] for r in pairs} | {path[r["a"]] for r in pairs} | {path[r["b"]] for r in pairs})
    for i, p in enumerate(files):
        hf_measures(cache, p)
        if (i + 1) % 500 == 0:
            cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))
    sharp = {}
    for r in pairs:
        o = cache[refpath[r["a"]]]["hp"]
        for k in (r["a"], r["b"]):
            sharp[k] = cache[path[k]]["hp"] / o if o > 0 else 1.0
    sa = np.array([sharp[r["a"]] for r in pairs])
    sb = np.array([sharp[r["b"]] for r in pairs])
    ds = sa - sb  # positive: a is the sharper crop
    dp = np.array([0.0 if x is None else (1.0 if x else -1.0) for x in pa])
    D = {}
    for stem in stems:
        s = pred(stem)
        D[stem] = np.array([s[r["a"]] - s[r["b"]] for r in pairs])
    ds2 = np.array([J["ssim2"][r["a"]] - J["ssim2"][r["b"]] for r in pairs])
    human = np.where(t > 0.5, 1.0, np.where(t < 0.5, -1.0, 0.0))  # +1: raters chose a
    base, chal = stems[0], stems[1]
    same = lambda x, y, sel: np.mean(np.sign(x[sel]) == np.sign(y[sel]))  # noqa: E731
    bands = {
        "all questions": np.ones(len(pairs), bool),
        f"{chal} own near-tie band (|d|<{band})": np.abs(D[chal]) < band,
        f"{base} own near-tie band": np.abs(D[base]) < band,
        "union near-tie band": (np.abs(D[base]) < band) | (np.abs(D[chal]) < band),
    }
    ad = np.abs(D[base])
    e = np.quantile(ad, np.linspace(0, 1, 9))
    bands[f"{base} |d| {e[5]:.1f}-{e[6]:.1f} (before the top-end dip)"] = (ad >= e[5]) & (ad < e[6])
    bands[f"{base} |d| {e[6]:.1f}-{e[7]:.1f} (the top-end dip)"] = (ad >= e[6]) & (ad < e[7])
    bands[f"{base} |d| above {e[7]:.1f}"] = ad >= e[7]
    print("== sharper-crop test: share of questions where each side prefers the crop with the higher high-pass energy "
          "ratio (ties in sharpness, |ratio difference| < 0.02, excluded)")
    print(f"| band | n | sharpness ratio a, b (median) | raters prefer sharper | {base} prefers sharper | {chal} prefers sharper | "
          "SSIMULACRA2 prefers sharper | PSNR prefers sharper |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for name, sel in bands.items():
        s2 = sel & (np.abs(ds) >= 0.02) & (human != 0)
        print(f"| {name} | {s2.sum()} | {np.median(sa[sel]):.2f}, {np.median(sb[sel]):.2f} | {same(human, ds, s2):.3f} | "
              f"{same(D[base], ds, s2):.3f} | {same(D[chal], ds, s2):.3f} | {same(ds2, ds, s2):.3f} | {same(dp, ds, s2):.3f} |")
    from scipy.stats import spearmanr
    shift = D[chal] - D[base]
    print(f"\nSpearman of ({chal} - {base}) score difference against the sharpness difference, all questions: "
          f"{spearmanr(shift, ds).statistic:+.3f}; against the high-pass energy of a plus b: "
          f"{spearmanr(shift, sa + sb).statistic:+.3f}")
    sel = bands[f"{chal} own near-tie band (|d|<{band})"] & (np.abs(ds) >= 0.02) & (human != 0)
    wrong = sel & (np.sign(D[chal]) != human)
    right = sel & (np.sign(D[chal]) == human)
    print(f"In {chal}'s near-tie band: where the raters went against {chal} (n={wrong.sum()}) the raters' crop is the sharper "
          f"one on {np.mean(np.sign(human[wrong]) == np.sign(ds[wrong])):.3f}; where they agreed (n={right.sum()}) on "
          f"{np.mean(np.sign(human[right]) == np.sign(ds[right])):.3f}")
    dip = bands[f"{base} |d| {e[6]:.1f}-{e[7]:.1f} (the top-end dip)"] & (np.abs(ds) >= 0.02) & (human != 0)
    lost = dip & (np.sign(D[base]) != human)
    print(f"In the top-end dip: the lower-scored crop is the sharper one on {np.mean(np.sign(D[base][dip]) != np.sign(ds[dip])):.3f} "
          f"of the questions; where the raters chose the lower-scored crop (n={lost.sum()}) it is the sharper one on "
          f"{np.mean(np.sign(human[lost]) == np.sign(ds[lost])):.3f}")
    if gallery is None or n_examples <= 0:
        return
    from PIL import Image
    (gallery / "zoom").mkdir(parents=True, exist_ok=True)
    idx = [i for i in np.where(wrong)[0]]
    idx.sort(key=lambda i: (-n[i], -abs(ds[i])))
    lines = []
    for k, i in enumerate(idx[:n_examples]):
        r = pairs[i]
        yo, ho = hp_image(refpath[r["a"]])
        ya, ha = hp_image(path[r["a"]])
        yb, hb = hp_image(path[r["b"]])
        from scipy import ndimage
        diff = ndimage.uniform_filter((ha - hb) ** 2, 96)
        cy, cx = np.unravel_index(np.argmax(diff[64:-64, 64:-64]), diff[64:-64, 64:-64].shape)
        cy += 64
        cx += 64
        y0, x0 = max(0, cy - 128), max(0, cx - 128)
        out = Image.new("L", (3 * 256, 256))
        for j, y in enumerate((yo, ya, yb)):
            crop = np.clip(y[y0:y0 + 256, x0:x0 + 256], 0, 255).astype(np.uint8)
            out.paste(Image.fromarray(crop), (j * 256, 0))
        name = f"zoom/neartie_{k + 1}.png"
        out.save(gallery / name)
        lines.append(f"{name}: {chal} d={D[chal][i]:+.2f} ({base} {D[base][i]:+.2f}), raters chose {'A' if human[i] > 0 else 'B'} "
                     f"({int(n[i])}), sharpness A {sa[i]:.2f} B {sb[i]:.2f}, SSIMULACRA2 prefers {'A' if ds2[i] > 0 else 'B'}, "
                     f"PSNR prefers {'A' if dp[i] > 0 else 'B'}; crop at ({x0},{y0}), panels original | A | B")
    (gallery / "zoom" / "INDEX.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_gallery(out, pairs, path, refpath, pred, J, pa, t, n, stems):
    (out / "img").mkdir(parents=True, exist_ok=True)
    P = {s: pred(s) for s in stems}
    s0 = P[stems[0]]
    diff = np.array([s0[r["a"]] - s0[r["b"]] for r in pairs])
    ad = np.abs(diff)
    agree = np.where(diff > 0, t, np.where(diff < 0, 1 - t, 0.5))
    edges = np.quantile(ad, np.linspace(0, 1, 9))
    dip = (ad >= edges[6]) & (ad < edges[7])
    sections = []

    def card(i, title):
        r = pairs[i]
        imgs = []
        for lab, p in (("original", refpath[r["a"]]), ("A", path[r["a"]]), ("B", path[r["b"]])):
            dst = out / "img" / os.path.basename(p)
            if not dst.exists():
                shutil.copy(p, dst)
            imgs.append(f"<figure><img src='img/{dst.name}' loading='lazy'><figcaption>{lab}</figcaption></figure>")
        va = int(round(t[i] * n[i]))
        rows = [("zensim " + s, f"{P[s][r['a']]:.1f}", f"{P[s][r['b']]:.1f}") for s in stems]
        rows += [(j, f"{J[j][r['a']]:.4g}", f"{J[j][r['b']]:.4g}") for j in J]
        rows += [("higher PSNR", "A" if pa[i] else ("B" if pa[i] is not None else "?"), ""),
                 ("human votes (A, B)", str(va), str(int(n[i]) - va))]
        tbl = "<table><tr><th></th><th>A</th><th>B</th></tr>" + "".join(
            f"<tr><td>{html.escape(k)}</td><td>{x}</td><td>{y}</td></tr>" for k, x, y in rows) + "</table>"
        return f"<section><h3>{html.escape(title)}</h3><div class='row'>{''.join(imgs)}{tbl}</div></section>"

    def pick(mask, want_agree, k, title):
        idx = [i for i in np.where(mask)[0] if (agree[i] >= 0.5) == want_agree and t[i] != 0.5]
        idx.sort(key=lambda i: (-n[i], -ad[i]))
        for j, i in enumerate(idx[:k]):
            sections.append(card(i, f"{title} #{j + 1}: {stems[0]} prefers {'A' if diff[i] > 0 else 'B'} by "
                                    f"{ad[i]:.1f}; {int(n[i])} rater(s) chose {'A' if t[i] > 0.5 else 'B'}"))

    pick(dip, False, 10, f"Dip band |difference| {edges[6]:.0f}-{edges[7]:.0f}, humans chose the lower-scored image")
    pick(dip, True, 4, f"Dip band |difference| {edges[6]:.0f}-{edges[7]:.0f}, humans agreed with the metrics")
    tie = ad < edges[1]
    sign = {j: np.array([np.sign(J[j][r["a"]] - J[j][r["b"]]) * (-1 if j == "butteraugli" else 1) for r in pairs]) for j in J}
    cons = np.all([sign[j] == sign["ssim2"] for j in J], axis=0) & (sign["ssim2"] != 0)
    wrong = tie & cons & (((sign["ssim2"] > 0) & (t < 0.5)) | ((sign["ssim2"] < 0) & (t > 0.5)))
    idx = sorted(np.where(wrong)[0], key=lambda i: -n[i])
    for j, i in enumerate(idx[:8]):
        sections.append(card(i, f"Near-tie for {stems[0]} (|difference| < {edges[1]:.1f}) #{j + 1}: the reference metrics "
                                f"all prefer {'A' if sign['ssim2'][i] > 0 else 'B'}; {int(n[i])} rater(s) chose "
                                f"{'A' if t[i] > 0.5 else 'B'}"))
    print(f"gallery: {int(dip.sum())} dip questions, {int((tie & cons).sum())} near-tie questions with a metric consensus, "
          f"{len(idx)} of them answered against it")
    head = ("<!doctype html><meta charset=utf-8><title>CLIC 2021 validation: pairs behind the agreement dip</title>"
            "<style>body{font:14px system-ui;margin:16px;background:#111;color:#ddd}figure{margin:0 8px 0 0;display:inline-block;"
            "vertical-align:top}img{max-width:360px;display:block}figcaption{text-align:center;color:#aaa}.row{display:flex;"
            "flex-wrap:wrap;gap:8px;align-items:flex-start}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #444;"
            "padding:2px 8px}section{margin-bottom:28px;border-bottom:1px solid #333;padding-bottom:12px}</style>")
    intro = ("<h1>CLIC 2021 validation: the pairs behind the agreement dip</h1><p>A and B keep the question's order; "
             "'human votes' are the 2AFC answers (which crop is closer to the original). Internal view only.</p>")
    (out / "index.html").write_text(head + intro + "".join(sections))


if __name__ == "__main__":
    main()
