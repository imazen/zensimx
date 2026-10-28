#!/usr/bin/env python3
"""Fix `unused variable` warnings on `let` bindings (2026-10-28 slim cut).

Reads `cargo clippy|check --message-format=short` output. For each warning that points at a
`let NAME = EXPR;` binding: when EXPR is side-effect free by inspection (no call except a
whitelist of pure methods and constructors), the statement is deleted; otherwise NAME
becomes `_NAME`. Names in tuple/struct patterns become `_` + NAME. Function parameters are
left for unused_params.py.

    unused_lets.py --root <workspace> CHECK.log
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402

LOC = re.compile(r"^([^:\s]+\.rs):(\d+):(\d+): warning: unused variable: `([^`]+)`")
PURE = {"len", "min", "max", "abs", "saturating_sub", "saturating_add", "wrapping_add",
        "wrapping_sub", "wrapping_mul", "checked_mul", "checked_add", "div_ceil", "sqrt",
        "to_bits", "is_empty", "unwrap_or", "as_slice", "iter", "count", "clone", "copied",
        "width", "height", "wide_h", "at_scale", "from", "new", "default", "is_some", "is_none",
        "map", "filter", "sum", "fin", "powi", "floor", "ceil", "round", "trunc", "contains",
        "step_by", "rev", "zip", "enumerate", "collect", "unwrap", "expect", "get", "first",
        "last", "chunks", "chunks_exact", "split_at", "as_ref", "as_deref", "into", "f64", "f32",
        "usize", "u32", "mul_add", "max_by", "min_by", "fold", "any", "all", "position"}


def offset_of(src, line, col):
    lines = src.split("\n")
    return sum(len(l) + 1 for l in lines[:line - 1]) + col - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("log")
    a = ap.parse_args()
    sites = defaultdict(set)
    for line in open(a.log):
        m = LOC.match(line.strip())
        if m:
            sites[m.group(1)].add((int(m.group(2)), int(m.group(3)), m.group(4)))
    for f, locs in sorted(sites.items()):
        p = Path(a.root) / f
        src = p.read_text()
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        by_off = {s: i for i, (_, s, _e) in enumerate(rw.toks)}
        done = 0
        for line, col, name in sorted(locs):
            name = name.replace("mut ", "").strip()
            ti = by_off.get(offset_of(src, line, col))
            if ti is None:
                continue
            if rw.text(ti) == "mut":
                ti = rw.sig(ti + 1)
            prev = rw.sig(ti - 1, -1)
            if rw.text(prev) == "mut":
                prev = rw.sig(prev - 1, -1)
            if rw.text(prev) == "let" and rw.text(rw.sig(ti + 1)) in ("=", ":"):
                # whole statement
                eq = rw.sig(ti + 1)
                if rw.text(eq) == ":":
                    while rw.text(eq) not in ("=", ";"):
                        if rw.text(eq) in ("(", "[", "{"):
                            eq = rw.match[eq]
                        eq = rw.sig(eq + 1)
                end = eq
                while end < len(rw.toks) and rw.text(end) != ";":
                    if rw.text(end) in ("(", "[", "{"):
                        end = rw.match[end]
                    end += 1
                init = rw.src[rw.toks[eq][2]:rw.toks[end][1]] if rw.text(eq) == "=" else ""
                calls = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:::<[^>]*>)?\s*\(", init)
                macros = re.search(r"[A-Za-z_]+!\s*[\(\[\{]", init)
                if not macros and all(c in PURE or c[:1].isupper() for c in calls):
                    rw.edits.append((rw.line_start(prev), rw.after_line(end), ""))
                    done += 1
                    continue
            # pattern binding or impure initializer: rename
            if not rw.text(ti).startswith("_"):
                rw.edits.append((rw.toks[ti][1], rw.toks[ti][2], "_" + rw.text(ti)))
                done += 1
        if rw.edits:
            rw.edits = merge_overlaps(rw.edits)
            p.write_text(rw.apply())
        print(f"{f}: {done} fixes")


if __name__ == "__main__":
    main()
