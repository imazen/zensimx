#!/usr/bin/env python3
"""Constant-fold removed boolean struct fields, driven by rustc's error locations.

After deleting `bool` fields from a struct (the 2026-10-28 slim-crate cut removed
research feature-family flags from `ComputeSet`), every remaining use is a compile
error. This tool reads `cargo check --message-format=short` output and fixes each site:

* E0560 (`struct X has no field named Y` in a literal): the `Y: expr,` initializer is
  deleted.
* E0609 (`no field Y on type X`): a write (`recv.Y = ..;`, `|=`, `&=`) deletes the
  statement; a read replaces the whole `recv.path.Y` expression with `false`.

Then bool_fold.py folds the literals to a fixed point. Run the compiler again afterwards.

    drop_fields.py --root <workspace> CHECK.log
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bool_fold import fold  # noqa: E402
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402

LOC = re.compile(r"^([^:\s]+\.rs):(\d+):(\d+): error\[(E0560|E0609)\]")

STOP = {";", ",", "{", "}", "(", "[", "=>", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||",
        "+", "-", "*", "/", "%", "&", "|", "^", "!", "?", ":", "return", "if", "while", "match",
        "let", "in", "else"}


def offset_of(src, line, col):
    lines = src.split("\n")
    # rustc columns count characters (1-based)
    return sum(len(l) + 1 for l in lines[:line - 1]) + col - 1


def receiver_start(rw, field_tok):
    """Token index where `recv.path.field` starts: walk back over `.ident`, `::`,
    `self`, and balanced call/index groups."""
    i = rw.sig(field_tok - 1, -1)  # the '.'
    if rw.text(i) != ".":
        return field_tok
    start = field_tok
    while True:
        j = rw.sig(i - 1, -1)
        t = rw.text(j)
        if t in (")", "]"):
            j = rw.match[j]
            k = rw.sig(j - 1, -1)
            if rw.toks[k][0] == "ident":
                j = k
            start = j
        elif rw.toks[j][0] == "ident" and t not in STOP:
            start = j
        else:
            break
        b = rw.sig(start - 1, -1)
        if rw.text(b) in (".", "::"):
            i = b
            continue
        break
    return start


def fix_sites(root, log):
    sites = defaultdict(list)
    for line in open(log):
        m = LOC.match(line.strip())
        if m:
            sites[m.group(1)].append((int(m.group(2)), int(m.group(3)), m.group(4)))
    for f, locs in sites.items():
        p = Path(root) / f
        src = p.read_text()
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        by_off = {s: i for i, (_, s, _e) in enumerate(rw.toks)}
        for line, col, code in sorted(set(locs)):
            off = offset_of(src, line, col)
            ti = by_off.get(off)
            if ti is None:
                print(f"no token at {f}:{line}:{col}")
                continue
            if code == "E0560":
                end = rw.item_end(ti)
                rw.edits.append((rw.line_start(ti), rw.after_line(end), ""))
                continue
            nxt = rw.sig(ti + 1)
            if nxt < len(rw.toks) and rw.text(nxt) in ("=", "|=", "&=", "^="):
                st = receiver_start(rw, ti)
                end = nxt
                while end < len(rw.toks) and rw.text(end) != ";":
                    if rw.text(end) in ("(", "[", "{"):
                        end = rw.match[end]
                    end += 1
                rw.edits.append((rw.line_start(st), rw.after_line(end), ""))
                continue
            st = receiver_start(rw, ti)
            rw.edits.append((rw.toks[st][1], rw.toks[ti][2], "false"))
        rw.edits = merge_overlaps(rw.edits)
        p.write_text(rw.apply())
        print(f"{f}: {len(rw.edits)} edits")
    return list(sites)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--simplify-only", nargs="*", default=None,
                    help="skip the error-driven step and only simplify these files")
    ap.add_argument("log", nargs="?")
    a = ap.parse_args()
    files = a.simplify_only if a.simplify_only is not None else fix_sites(a.root, a.log)
    for f in files:
        p = Path(a.root) / f if not Path(f).is_absolute() else Path(f)
        src, rounds = fold(p.read_text())
        p.write_text(re.sub(r"\n{3,}", "\n\n", src))
        print(f"{f}: {rounds} folds")


if __name__ == "__main__":
    main()
