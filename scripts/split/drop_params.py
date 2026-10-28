#!/usr/bin/env python3
"""Remove named parameters from a function and the matching arguments at every call site
(2026-10-28 slim cut). Call sites are `NAME(` / `.NAME(` / `Self::NAME(` in the given files;
arguments are matched by position, so the function must not be called through a pointer or
macro. Run it after the parameter has been folded to a constant inside the function.

    drop_params.py --fn stream_phase_b --param append --param r4w FILE...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402


def split_args(rw, open_tok):
    """[(start_tok, end_tok)] of the top-level comma-separated items inside the group."""
    close = rw.match[open_tok]
    items, start, i = [], None, open_tok + 1
    while i < close:
        k = rw.toks[i][0]
        t = rw.text(i)
        if k in ("ws", "comment", "doc"):
            i += 1
            continue
        if start is None:
            start = i
        if t in ("(", "[", "{"):
            i = rw.match[i] + 1
            last = i - 1
            continue
        if t == "<" and items is not None and is_generic_ctx(rw, i):
            pass
        if t == ",":
            items.append((start, prev_sig(rw, i)))
            start = None
        i += 1
    if start is not None:
        items.append((start, prev_sig(rw, close)))
    return items, close


def is_generic_ctx(rw, i):
    return False


def prev_sig(rw, i):
    j = i - 1
    while rw.toks[j][0] in ("ws", "comment", "doc"):
        j -= 1
    return j


def param_name(rw, s, e):
    """Name of a `[mut] NAME: TYPE` parameter spanning tokens s..e."""
    j = s
    while j <= e and rw.toks[j][0] in ("ws", "comment", "doc"):
        j += 1
    if rw.text(j) == "mut":
        j = rw.sig(j + 1)
    return rw.text(j)


def removal_span(rw, items, idx, close):
    """Byte span removing item idx with its separating comma and comments above it."""
    s, e = items[idx]
    # include the trailing comma (if any)
    nxt = rw.sig(e + 1)
    if nxt < close and rw.text(nxt) == ",":
        e = nxt
    start = rw.line_start(s)
    # take comment lines directly above the parameter with it
    lead = rw.leading_start(s)
    start = rw.line_start(lead)
    return start, rw.after_line(e), ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fn", required=True)
    ap.add_argument("--param", action="append", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    # 1. the definition: find positions
    positions = None
    takes_token = False
    for f in a.files:
        src = Path(f).read_text()
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        for i, (k, s, e) in enumerate(rw.toks):
            if k == "ident" and rw.text(i) == "fn" and rw.text(rw.sig(i + 1)) == a.fn:
                j = rw.sig(i + 1)
                j = rw.sig(j + 1)
                if rw.text(j) == "<":
                    depth = 0
                    while True:
                        t = rw.text(j)
                        depth += t == "<"
                        depth -= t == ">"
                        if depth == 0:
                            break
                        j += 1
                    j = rw.sig(j + 1)
                items, close = split_args(rw, j)
                names = [param_name(rw, s0, e0) for s0, e0 in items]
                positions = [names.index(p) for p in a.param]
                # `#[magetypes]` / `#[arcane]` functions take the token first; inside
                # `incant!(..)` the call omits it, so those call sites shift by one.
                takes_token = bool(names) and names[0] in ("token", "_token")
                if "\n" not in rw.src[rw.toks[j][1]:rw.toks[close][2]]:
                    kept = [rw.src[rw.toks[a][1]:rw.toks[b][2]] for q, (a, b) in enumerate(items) if q not in positions]
                    rw.edits.append((rw.toks[j][1], rw.toks[close][2], "(" + ", ".join(kept) + ")"))
                else:
                    for idx in positions:
                        rw.edits.append(removal_span(rw, items, idx, close))
                rw.edits = merge_overlaps(rw.edits)
                Path(f).write_text(rw.apply())
                print(f"{f}: fn {a.fn}({', '.join(names)}) drop positions {positions}")
                break
        if positions is not None:
            break
    if positions is None:
        sys.exit(f"fn {a.fn} not found")
    # 2. call sites
    for f in a.files:
        src = Path(f).read_text()
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        n = 0
        for i, (k, s, e) in enumerate(rw.toks):
            if k != "ident" or rw.text(i) != a.fn:
                continue
            prev = rw.text(rw.sig(i - 1, -1))
            if prev == "fn":
                continue
            j = rw.sig(i + 1)
            # skip a turbofish `::<..>`
            if j < len(rw.toks) and rw.text(j) == "::" and rw.text(rw.sig(j + 1)) == "<":
                q = rw.sig(j + 1)
                depth = 0
                while q < len(rw.toks):
                    t = rw.text(q)
                    if t in ("(", "[", "{"):
                        q = rw.match[q]
                    elif t == "<":
                        depth += 1
                    elif t == ">":
                        depth -= 1
                        if depth == 0:
                            break
                    elif t == ">>":
                        depth -= 2
                        if depth <= 0:
                            break
                    q += 1
                j = rw.sig(q + 1)
            if j >= len(rw.toks) or rw.text(j) != "(":
                continue
            items, close = split_args(rw, j)
            in_incant = (rw.text(rw.sig(i - 1, -1)) == "(" and rw.text(rw.sig(rw.sig(i - 1, -1) - 1, -1)) == "!"
                         and rw.text(rw.sig(rw.sig(rw.sig(i - 1, -1) - 1, -1) - 1, -1)) == "incant")
            shift = 1 if (takes_token and in_incant) else 0
            drop = {q - shift for q in positions if 0 <= q - shift < len(items)}
            if "\n" not in rw.src[rw.toks[j][1]:rw.toks[close][2]]:
                kept = [rw.src[rw.toks[a][1]:rw.toks[b][2]] for q, (a, b) in enumerate(items) if q not in drop]
                if drop:
                    rw.edits.append((rw.toks[j][1], rw.toks[close][2], "(" + ", ".join(kept) + ")"))
                n += 1
                continue
            for idx in sorted(drop):
                if 0 <= idx < len(items):
                    sp = removal_span(rw, items, idx, close)
                    # arguments on one line: remove just the text and the comma
                    s0, e0 = items[idx]
                    if rw.src.count("\n", rw.toks[s0][1], rw.toks[e0][2]) == 0 and \
                            rw.src.count("\n", rw.toks[j][1], rw.toks[close][2]) == 0:
                        nxt = rw.sig(e0 + 1)
                        end = rw.toks[nxt][2] if rw.text(nxt) == "," else rw.toks[e0][2]
                        st = rw.toks[s0][1]
                        if rw.text(nxt) == ",":
                            # also eat one following space
                            if rw.src[end:end + 1] == " ":
                                end += 1
                        else:
                            # last arg: eat the preceding ", "
                            p = prev_sig(rw, s0)
                            if rw.text(p) == ",":
                                st = rw.toks[p][1]
                        sp = (st, end, "")
                    rw.edits.append(sp)
            n += 1
        if rw.edits:
            rw.edits = merge_overlaps(rw.edits)
            Path(f).write_text(rw.apply())
        if n:
            print(f"{f}: {n} call sites")


if __name__ == "__main__":
    main()
