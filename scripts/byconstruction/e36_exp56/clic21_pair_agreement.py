"""Why the CLIC 2021 validation agreement curve zigzags (owner question, 2026-11-03; record, R10 amendment 5 result).

For each model or judge score in a CLIC 2021 confirmation dir: 2AFC agreement per |score difference| bin (equal-count
bins, with the binomial standard error), the same restricted to the questions several raters answered, the share of
each bin where the library's PSNR label orders the pair the same way, and the reference metrics' sign agreement on the
same questions. Optionally writes a gallery of example questions (original, A, B, every score, the votes) under
/mnt/v/output for viewing at http://dev.lan:3300/.

    clic21_pair_agreement.py CONFIRM_DIR [--scores control_s0,mixF3_s0] [--bins 8] [--gallery OUT_DIR]
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
    if a.gallery:
        write_gallery(Path(a.gallery), pairs, path, refpath, pred, J, pa, t, n, a.scores.split(","))


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
