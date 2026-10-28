#!/usr/bin/env python3
"""Remove `const NAME: bool` generic parameters that always take one value (2026-10-28 slim cut).

For one function: every use of NAME in its body becomes `false` (then bool_fold.py folds the
branches), NAME is removed from the generic parameter list, and the matching turbofish
argument is removed at every call site `FN::<..>(` in the given files (including calls
inside `incant!`). Hand-written tier variants (`FN_scalar`, `FN_neon`, ...) are handled
when listed with --fn as well.

    drop_generics.py --fn gradient_block_kernel_canon --param BANDVIS --param BANK FILE...
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402
from bool_fold import fold  # noqa: E402


def angle_end(rw, j):
    """Token index of the `>` closing the `<` at j (handles nested brackets and `>>`)."""
    depth = 0
    q = j
    while q < len(rw.toks):
        t = rw.text(q)
        if t in ("(", "[", "{"):
            q = rw.match[q]
        elif t == "<":
            depth += 1
        elif t == ">":
            depth -= 1
            if depth == 0:
                return q
        elif t == ">>":
            depth -= 2
            if depth <= 0:
                return q
        q += 1
    raise ValueError("unbalanced <")


def generic_items(rw, lt, gt):
    """[(start_tok, end_tok)] of the comma-separated items between lt and gt."""
    items, start, i, depth = [], None, lt + 1, 0
    while i < gt:
        k = rw.toks[i][0]
        t = rw.text(i)
        if k in ("ws", "comment", "doc"):
            i += 1
            continue
        if start is None:
            start = i
        if t in ("(", "[", "{"):
            i = rw.match[i] + 1
            continue
        if t == "<":
            depth += 1
        elif t == ">":
            depth -= 1
        if t == "," and depth == 0:
            items.append((start, prev_sig(rw, i)))
            start = None
        i += 1
    if start is not None:
        items.append((start, prev_sig(rw, gt)))
    return items


def prev_sig(rw, i):
    j = i - 1
    while rw.toks[j][0] in ("ws", "comment", "doc"):
        j -= 1
    return j


def remove_item_span(rw, items, idx, gt):
    s, e = items[idx]
    nxt = rw.sig(e + 1)
    if nxt < gt and rw.text(nxt) == ",":
        st, en = rw.toks[s][1], rw.toks[nxt][2]
        # eat following whitespace on the same line
        while en < len(rw.src) and rw.src[en] == " ":
            en += 1
        # a whole line of its own: take the line
        ls = rw.src.rfind("\n", 0, st) + 1
        if rw.src[ls:st].strip() == "" and rw.src[en:en + 1] == "\n":
            return ls, en + 1, ""
        return st, en, ""
    # last item: eat the preceding comma
    p = prev_sig(rw, s)
    st = rw.toks[p][1] if rw.text(p) == "," else rw.toks[s][1]
    return st, rw.toks[e][2], ""


def rewrite_list(rw, lt, gt, items, drop, turbofish_colons=None):
    """Edit replacing `<..>` (and a preceding `::` for a turbofish) with the kept items."""
    kept = [rw.src[rw.toks[s0][1]:rw.toks[e0][2]] for k, (s0, e0) in enumerate(items) if k not in drop]
    start = rw.toks[turbofish_colons][1] if (turbofish_colons is not None and not kept) else rw.toks[lt][1]
    end = rw.toks[gt][2]
    if rw.text(gt) == ">>":
        end -= 1  # the second `>` closes an outer list
    if not kept:
        return start, end, ""
    return start, end, "<" + ", ".join(kept) + ">"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fn", action="append", required=True)
    ap.add_argument("--param", action="append", required=True)
    ap.add_argument("--value", default="false", choices=["false", "true"],
                    help="the constant every listed parameter always takes")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for fname in a.fn:
        positions = None
        for f in a.files:
            src = Path(f).read_text()
            rw = Rewriter(src, {"feature": {}, "cfg": {}})
            for i, (k, s, e) in enumerate(rw.toks):
                if k == "ident" and rw.text(i) == "fn" and rw.text(rw.sig(i + 1)) == fname:
                    lt = rw.sig(rw.sig(i + 1) + 1)
                    if rw.text(lt) != "<":
                        break
                    gt = angle_end(rw, lt)
                    items = generic_items(rw, lt, gt)
                    names = []
                    for s0, e0 in items:
                        txt = rw.src[rw.toks[s0][1]:rw.toks[e0][2]]
                        m = re.match(r"const\s+([A-Za-z0-9_]+)\s*:", txt)
                        names.append(m.group(1) if m else txt.split(":")[0].strip())
                    positions = [names.index(p) for p in a.param if p in names]
                    params = [p for p in a.param if p in names]
                    # body: replace uses with `false`
                    body_open = gt
                    while rw.text(body_open) != "{":
                        if rw.text(body_open) in ("(", "["):
                            body_open = rw.match[body_open]
                        body_open += 1
                    body_close = rw.match[body_open]
                    for q in range(body_open, body_close):
                        if rw.toks[q][0] == "ident" and rw.text(q) in params:
                            rw.edits.append((rw.toks[q][1], rw.toks[q][2], a.value))
                    rw.edits.append(rewrite_list(rw, lt, gt, items, set(positions)))
                    rw.edits = merge_overlaps(rw.edits)
                    out, _ = fold(rw.apply())
                    Path(f).write_text(re.sub(r"\n{3,}", "\n\n", out))
                    print(f"{f}: fn {fname}<{', '.join(names)}> drop {params} at {positions}")
                    break
            if positions is not None:
                break
        if not positions:
            print(f"fn {fname}: nothing to drop")
            continue
        for f in a.files:
            src = Path(f).read_text()
            rw = Rewriter(src, {"feature": {}, "cfg": {}})
            n = 0
            for i, (k, s, e) in enumerate(rw.toks):
                if k != "ident" or rw.text(i) != fname or rw.text(rw.sig(i - 1, -1)) == "fn":
                    continue
                j = rw.sig(i + 1)
                if rw.text(j) != "::" or rw.text(rw.sig(j + 1)) != "<":
                    continue
                lt = rw.sig(j + 1)
                gt = angle_end(rw, lt)
                items = generic_items(rw, lt, gt)
                rw.edits.append(rewrite_list(rw, lt, gt, items, set(positions), turbofish_colons=j))
                n += 1
            if rw.edits:
                rw.edits = merge_overlaps(rw.edits)
                Path(f).write_text(rw.apply())
            if n:
                print(f"{f}: {n} turbofish call sites")


if __name__ == "__main__":
    main()
