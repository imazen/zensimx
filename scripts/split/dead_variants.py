#!/usr/bin/env python3
"""Remove enum variants rustc reports as never constructed in EVERY build configuration
(2026-10-28 slim cut): the declaration (with its docs/attributes) and every match arm or
or-pattern head that names it (`Enum::Variant`, `crate::..::Enum::Variant`, and
`Self::Variant` inside the defining file). Remaining references are compile errors to fix
by hand.

    dead_variants.py --root <workspace> [--dry-run] LOG.json...
"""
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402

HERE = Path(__file__).resolve().parent


def spans(log):
    out = set()
    for line in open(log):
        if not line.startswith("{"):
            continue
        m = json.loads(line)
        if m.get("reason") != "compiler-message":
            continue
        msg = m["message"]
        if (msg.get("code") or {}).get("code") != "dead_code" or "never constructed" not in msg["message"]:
            continue
        if not msg["message"].startswith(("variant", "variants", "multiple variants")):
            continue
        for sp in msg["spans"]:
            if sp.get("is_primary"):
                out.add((sp["file_name"], sp["byte_start"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    common = set.intersection(*[spans(l) for l in a.logs])
    found = defaultdict(list)  # file -> [(enum, variant, tok)]
    for f, b in sorted(common):
        p = Path(a.root) / f
        raw = p.read_bytes()
        src = raw.decode("utf-8")
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        off = len(raw[:b].decode("utf-8"))
        i = next((k for k, t in enumerate(rw.toks) if t[1] == off), None)
        if i is None:
            continue
        # enclosing `{` and the `enum NAME` before it
        j = i - 1
        while j >= 0:
            t = rw.text(j)
            if t in (")", "]", "}"):
                j = rw.match[j] - 1
                continue
            if t == "{":
                break
            j -= 1
        k = rw.sig(j - 1, -1)
        while k >= 0 and rw.text(rw.sig(k - 1, -1)) != "enum":
            k = rw.sig(k - 1, -1)
            if rw.text(k) in (";", "}", "{"):
                k = -1
                break
        if k < 0:
            print(f"skip {f}: no enum for {rw.text(i)}")
            continue
        found[f].append((rw.text(k), rw.text(i), i))
    all_rs = sorted(str(p) for p in (Path(a.root) / "zensim" / "src").rglob("*.rs"))
    for f, items in sorted(found.items()):
        for enum, var, _ in items:
            print(f"variant {f}: {enum}::{var}")
        if a.dry_run:
            continue
        # 1. arms everywhere (full and crate paths) and Self:: arms in the defining file
        pats = []
        for enum, var, _ in items:
            pats += ["--pattern", f"{enum}::{var}"]
        subprocess.run([sys.executable, str(HERE / "remove_arms.py")] + pats + all_rs, check=False)
        # `Self::Var` arms only when no other enum in this file has a variant of that name
        # (2026-10-28: `Self::Palette` matched a live `Family::Palette` arm).
        file_src = (Path(a.root) / f).read_text()
        pats = []
        for enum, var, _ in items:
            owners = [m for m in re.finditer(r"\benum\s+([A-Za-z0-9_]+)[^{]*\{", file_src)
                      if re.search(r"(?m)^\s*" + re.escape(var) + r"\b\s*[,({=]",
                                   file_src[m.end():file_src.find("\n}", m.end())])]
            if len(owners) == 1:
                pats += ["--pattern", f"Self::{var}"]
            else:
                print(f"manual {f}: Self::{var} is ambiguous ({len(owners)} enums)")
        if pats:
            subprocess.run([sys.executable, str(HERE / "remove_arms.py")] + pats + [str(Path(a.root) / f)], check=False)
        # 2. the declarations
        p = Path(a.root) / f
        src = p.read_text()
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        names = {var for _, var, _ in items}
        enums = {enum for enum, _, _ in items}
        for idx, (kind, s0, e0) in enumerate(rw.toks):
            if kind != "ident" or rw.text(idx) not in names:
                continue
            prev = rw.text(rw.sig(idx - 1, -1))
            if prev not in ("{", ",", "]") and not rw.text(rw.sig(idx - 1, -1)).endswith("*/"):
                continue
            # inside one of our enums?
            j = idx - 1
            while j >= 0:
                t = rw.text(j)
                if t in (")", "]", "}"):
                    j = rw.match[j] - 1
                    continue
                if t == "{":
                    break
                j -= 1
            k = rw.sig(j - 1, -1)
            while k >= 0 and rw.text(rw.sig(k - 1, -1)) not in ("enum",) and rw.text(k) not in (";", "}", "{"):
                k = rw.sig(k - 1, -1)
            if k < 0 or rw.text(rw.sig(k - 1, -1)) != "enum" or rw.text(k) not in enums:
                continue
            # extent: through the following `,` (or the payload group and `,`)
            e = rw.sig(idx + 1)
            if rw.text(e) in ("(", "{"):
                e = rw.sig(rw.match[e] + 1)
            if rw.text(e) == "=":
                while rw.text(e) not in (",", "}"):
                    e = rw.sig(e + 1)
            end = e if rw.text(e) == "," else rw.sig(e - 1, -1)
            lead = rw.leading_start(idx)
            rw.edits.append((rw.line_start(lead), rw.after_line(end), ""))
        rw.edits = merge_overlaps(rw.edits)
        p.write_text(re.sub(r"\n{3,}", "\n\n", rw.apply()))


if __name__ == "__main__":
    main()
