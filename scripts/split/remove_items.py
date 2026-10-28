#!/usr/bin/env python3
"""Delete named Rust items (with their doc comments and attributes) from source files.

Companion to cfg_prune.py for the 2026-10-28 slim-crate cut. Each spec is `KIND NAME`,
where KIND is fn, struct, enum, impl, mod, const, static, type, trait, use or macro; for
`impl` the NAME is matched against the whole header text (e.g. `impl Foo for Bar`);
`impls NAME` removes every impl block whose header mentions NAME.
Every match in every given file is removed and reported with its line number:

    remove_items.py --item 'fn compute_v2_features' --item 'mod pu_linear_extended_tests' FILE...
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402


def find_items(rw, kind, name):
    """Yield (start_tok, end_tok) for items `kind name` in rw's source."""
    toks = rw.toks
    for i, (k, s, e) in enumerate(toks):
        want = {"macro": "macro_rules", "impls": "impl"}.get(kind, kind)
        if k != "ident" or rw.text(i) != want:
            continue
        # the item name follows the keyword (fn/struct/... NAME), or the header text for impl
        j = rw.sig(i + 1)
        if kind == "macro":
            if j < len(toks) and rw.text(j) == "!":
                j = rw.sig(j + 1)
        if kind == "impls":
            brace = j
            while brace < len(toks) and rw.text(brace) not in ("{", ";"):
                brace += 1
            header = rw.src[s:toks[brace][1]]
            if not re.search(r"\b" + re.escape(name) + r"\b", header):
                continue
        elif kind == "impl":
            brace = j
            while brace < len(toks) and rw.text(brace) not in ("{", ";"):
                brace += 1
            header = re.sub(r"\s+", " ", rw.src[s:toks[brace][1]]).strip()
            if header != "impl " + name and header != name:
                continue
        else:
            if j >= len(toks) or rw.text(j) != name:
                continue
        # walk back over visibility / qualifiers to the item start
        start = i
        b = rw.sig(i - 1, -1)
        while b >= 0 and (rw.text(b) in ("pub", "const", "unsafe", "async", "extern", "default") or
                          (rw.text(b) == ")" and rw.text(rw.sig(rw.match[b] - 1, -1)) == "pub")):
            if rw.text(b) == ")":
                b = rw.sig(rw.match[b] - 1, -1)
            start = b
            b = rw.sig(b - 1, -1)
        # item end: the body's closing brace, or `;`
        k2 = i
        while k2 < len(toks) and rw.text(k2) not in ("{", ";"):
            if rw.text(k2) in ("(", "["):
                k2 = rw.match[k2]
            k2 += 1
        end = rw.match[k2] if rw.text(k2) == "{" else k2
        yield start, end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item", action="append", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    specs = [tuple(x.split(None, 1)) for x in a.item]
    found = {sp: 0 for sp in specs}
    for f in a.files:
        p = Path(f)
        src = p.read_text()
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        for kind, name in specs:
            for st, en in find_items(rw, kind, name):
                lead = rw.leading_start(st)
                line = src.count("\n", 0, rw.toks[st][1]) + 1
                print(f"{f}:{line}: removing {kind} {name}")
                rw.edits.append((rw.line_start(lead), rw.after_line(en), ""))
                found[(kind, name)] += 1
        if rw.edits:
            rw.edits = merge_overlaps(rw.edits)
            out = rw.apply()
            out = re.sub(r"\n{3,}", "\n\n", out)
            p.write_text(out)
    missing = [f"{k} {n}" for (k, n), c in found.items() if c == 0]
    if missing:
        print("NOT FOUND: " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
