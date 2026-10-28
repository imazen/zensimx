#!/usr/bin/env python3
"""Unified multi-bake eval dashboard (user: 'we need a better way to eval/dashboard all the
stuff'). Runs bake_verdict on every bake in the manifest — which now scores the NON-PHOTO
axis (imazen-26 held-out) alongside the 6 photographic corpora + corruption + dial — and
assembles ONE self-contained, color-coded HTML comparing all bakes × all corpora × all axes,
so a regression (esp. non-photo content-blindness, §8.34/§8.35) is visible at a glance.

  usage: bake_dashboard.py [--bakes label:path,label:path,...] [--out dashboard.html]
Default manifest = the current candidate set (shipped B/BHdr + the diverse-negatives line).
"""
import argparse
import re
import subprocess
from pathlib import Path

REPO = Path.home() / "work/zen/zensimx"
BV = str(REPO / "target/release/bake_verdict")
REPORTS = Path("/mnt/v/output/zensim/reports/b_negatives")
WEIGHTS = REPO / "zensim/weights"
OUTDIR = Path("/mnt/v/output/zensim/dashboards")

# label -> bake path. Shipped profiles + the diverse-negatives investigation line.
DEFAULT = [
    ("B (shipped SDR)", WEIGHTS / "b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"),
    ("A (deprecated MLP)", WEIGHTS / "v47_strict_qat_native_2026-05-27.bin"),
    ("§8.33 photo-neg MLP", REPORTS / "mlp_neg_candidate_2026-07-15.bin"),
    ("diverse POISONED", REPORTS / "mlp_diverse_dv1.0kw0.3_2026-07-15.bin"),
    ("diverse DE-POISONED", REPORTS / "mlp_diverse_depoison_dv0.5kw0.3_2026-07-15.bin"),
]

# columns: (key, display, higher_is_better)
COLS = [("cid22", "CID22", True), ("kadid", "KADID", True), ("tid", "TID", True),
        ("konjnd", "KonJND", True), ("aic3", "AIC-3", True), ("aic4", "AIC-4", True),
        ("nonphoto", "NON-PHOTO", True), ("corrupt", "corrupt<q20", True),
        ("mono", "dial-mono", True)]
DISPLAY2KEY = {"CID22": "cid22", "KADIK10k": "kadid", "TID2013": "tid",
               "KonJND-1k (full)": "konjnd", "AIC-3 CTC": "aic3", "AIC-4 sample": "aic4",
               "imazen-26 non-photo (held-out)": "nonphoto"}


def run_verdict(path, out):
    subprocess.run([BV, "--bake", str(path), "--output", str(out)],
                   capture_output=True, text=True)
    md = Path(out).read_text()
    r = {}
    for line in md.splitlines():
        m = re.match(r"\|\s*(.+?)\s*\|\s*\d+\s*\|\s*([-\d.]+)\s*\|", line)
        if m and m.group(1) in DISPLAY2KEY:
            r[DISPLAY2KEY[m.group(1)]] = float(m.group(2))
        if "corruption < q20" in line:
            mm = re.search(r"([\d.]+)%", line)
            if mm:
                r["corrupt"] = float(mm.group(1)) / 100.0
        if "monotonicity (1 −" in line or "monotonicity (1 -" in line:
            mm = re.search(r"\|\s*([\d.]+)\s*\|", line)
            if mm:
                r["mono"] = float(mm.group(1))
    return r


def color(v, lo, hi):
    """green(best)->red(worst) over [lo,hi]."""
    if v is None:
        return "#444", "#888"
    t = 0.0 if hi == lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    # red(0) -> yellow(0.5) -> green(1)
    r = int(220 * (1 - t) + 60 * t); g = int(60 * (1 - t) + 170 * t); b = 70
    return f"rgb({r},{g},{b})", "#fff"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakes", default=None, help="label:path,label:path,...")
    ap.add_argument("--out", default=str(OUTDIR / "bake_dashboard_2026-07-15.html"))
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bakes = ([(s.split(":", 1)[0], Path(s.split(":", 1)[1])) for s in a.bakes.split(",")]
             if a.bakes else DEFAULT)
    rows = []
    for label, path in bakes:
        if not Path(path).exists():
            print(f"  skip (missing): {label} {path}"); continue
        vd = run_verdict(path, OUTDIR / f"_verdict_{label.replace(' ', '_').replace('/', '_')}.md")
        rows.append((label, vd))
        print(f"  scored {label}: " + " ".join(f"{k}={vd.get(k, float('nan')):.3f}"
                                                for k, *_ in COLS if k in vd))
    # per-column min/max for color scaling
    span = {}
    for key, _, _ in COLS:
        vals = [vd[key] for _, vd in rows if key in vd]
        span[key] = (min(vals), max(vals)) if vals else (0, 1)

    th = "".join(f"<th>{d}</th>" for _, d, _ in COLS)
    trs = []
    for label, vd in rows:
        tds = []
        for key, _, _ in COLS:
            v = vd.get(key)
            lo, hi = span[key]
            bg, fg = color(v, lo, hi)
            txt = "—" if v is None else (f"{v*100:.1f}%" if key == "corrupt" else f"{v:.4f}")
            flag = ""
            if key == "nonphoto" and v is not None:
                flag = " ⚠⚠" if v < 0.50 else (" ⚠" if v < 0.88 else "")
            tds.append(f'<td style="background:{bg};color:{fg}">{txt}{flag}</td>')
        trs.append(f"<tr><td class='lbl'>{label}</td>{''.join(tds)}</tr>")

    html = f"""<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:1.5rem;background:#111;color:#ddd}}
h1{{font-size:1.3rem}} .sub{{color:#999;max-width:60rem}}
table{{border-collapse:collapse;margin-top:1rem}}
th,td{{padding:.4rem .6rem;text-align:center;border:1px solid #333}}
th{{background:#1c1c1c;position:sticky;top:0}} td.lbl{{text-align:left;background:#1c1c1c;font-weight:600;white-space:nowrap}}
.note{{margin-top:1rem;color:#aaa;font-size:.9rem;max-width:60rem}}
code{{background:#222;padding:.1rem .3rem;border-radius:3px}}
</style>
<h1>zensim bake dashboard — all bakes × corpora × axes</h1>
<p class="sub">Rank SROCC per corpus (higher=better; green=best, red=worst per column). The
<b>NON-PHOTO</b> column is the held-out imazen-26 diverse-content axis (screen/UI/doc/line-art/
AI-gen); a photographic-only bake is content-blind there (≈0.86, ⚠) while the six standard
corpora can't see it. <code>corrupt&lt;q20</code>=negative-tail ranking; <code>dial-mono</code>=
codec-dial monotonicity. Generated by <code>scripts/v_next/bake_dashboard.py</code>.</p>
<table><thead><tr><th>bake</th>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>
<p class="note">⚠ = non-photo content-weak (SROCC &lt; 0.88, below diverse-trained ~0.93);
⚠⚠ = non-photo crash (SROCC &lt; 0.50, garbage ranking). The de-poisoned diverse bake is the
only one that clears the non-photo axis while holding CID22 — see §8.34/§8.35.</p>
"""
    Path(a.out).write_text(html)
    url = str(a.out).replace("/mnt/v/output/", "http://localhost:3300/")
    print(f"\nwrote {a.out}\n  view: {url}")


if __name__ == "__main__":
    main()
