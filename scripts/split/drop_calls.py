#!/usr/bin/env python3
"""Delete every statement that calls into a path prefix (2026-10-28 slim cut).

Used to remove diagnostic hooks (`crate::fold_timing::...`) whose results never reach a
computed value. A statement is deleted when it is an expression statement whose first
tokens are the prefix, or a `let NAME = <prefix>...;` binding; the names such `let`s bind
are deleted transitively from statements that use them as a whole expression statement
(`if let Some(t) = NAME { .. }`, `struct(NAME)` bindings are reported, not edited).

    drop_calls.py --prefix crate::fold_timing:: FILE...
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402


def statement_end(rw, i):
    while i < len(rw.toks) and rw.text(i) != ";":
        if rw.text(i) in ("(", "[", "{"):
            i = rw.match[i]
        i += 1
    return i


def run(src, prefix):
    rw = Rewriter(src, {"feature": {}, "cfg": {}})
    sig = [i for i in range(len(rw.toks)) if rw.toks[i][0] not in ("ws", "comment", "doc")]
    T = lambda k: rw.text(sig[k]) if 0 <= k < len(sig) else ""
    parts = [p for p in re.split(r"(::)", prefix) if p]
    n = len(parts)
    bound = []
    report = []
    k = 0
    while k < len(sig):
        if [T(k + j) for j in range(n)] == parts:
            prev = T(k - 1)
            if prev in (";", "{", "}"):
                end = statement_end(rw, sig[k])
                rw.edits.append((rw.line_start(sig[k]), rw.after_line(end), ""))
            elif prev == "=" and T(k - 3) == "let":
                bound.append(T(k - 2))
                end = statement_end(rw, sig[k])
                rw.edits.append((rw.line_start(sig[k - 3]), rw.after_line(end), ""))
            else:
                report.append(rw.toks[sig[k]][1])
        k += 1
    return rw, bound, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for f in a.files:
        p = Path(f)
        src = p.read_text()
        if a.prefix.replace("::", "") not in src.replace("::", "").replace(" ", ""):
            continue
        rw, bound, report = run(src, a.prefix)
        if not rw.edits and not report:
            continue
        rw.edits = merge_overlaps(rw.edits)
        out = rw.apply()
        p.write_text(out)
        lines = [src.count("\n", 0, off) + 1 for off in report]
        print(f"{f}: {len(rw.edits)} statements; bound {sorted(set(bound))}; left at lines {lines}")


if __name__ == "__main__":
    main()
