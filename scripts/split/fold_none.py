#!/usr/bin/env python3
"""Fold an `Option` that is always `None` (2026-10-28 slim cut).

For each NAME (a local, parameter or `self.`/`x.` field path ending in NAME) inside the
given byte range of a file, one rewrite per round until nothing changes:

* `if let Some(..) = NAME<chain> [&& ..] { A } [else B]` -> `B` (or nothing);
* `NAME<chain>.is_some()` -> `false`, `.is_none()` -> `true`;
* any other `NAME<chain>` expression -> `None`;

where `<chain>` is any run of `.as_deref_mut()`, `.as_deref()`, `.as_mut()`, `.as_ref()`,
`.take()`, `.clone()`, `.copied()`, `.cloned()`, `.filter(..)`, `.map(..)`, `.and_then(..)`.
Bindings, struct fields and parameters are left for the compiler to report as unused.
Run bool_fold.py afterwards.

    fold_none.py --name retention [--fn walk_fn] FILE
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402
from bool_fold import fold  # noqa: E402

CHAIN0 = {"as_deref_mut", "as_deref", "as_mut", "as_ref", "take", "clone", "copied", "cloned"}
CHAIN1 = {"filter", "map", "and_then"}
# `.map_or(d, f)` / `.map_or_else(|| d, f)` on None is `d`: handled in once().


class F:
    def __init__(self, src):
        self.rw = Rewriter(src, {"feature": {}, "cfg": {}})
        rw = self.rw
        self.sig = [i for i in range(len(rw.toks)) if rw.toks[i][0] not in ("ws", "comment", "doc")]
        self.pos = {t: k for k, t in enumerate(self.sig)}

    def T(self, k):
        return self.rw.text(self.sig[k]) if 0 <= k < len(self.sig) else ""

    def mate(self, k):
        return self.pos[self.rw.match[self.sig[k]]]

    def off(self, k):
        return self.rw.toks[self.sig[k]][1]

    def end(self, k):
        return self.rw.toks[self.sig[k]][2]


def enclosing(f, k):
    """The open bracket enclosing sig index k."""
    j = k - 1
    while j >= 0:
        t = f.T(j)
        if t in (")", "]", "}"):
            j = f.mate(j) - 1
            continue
        if t in ("(", "[", "{"):
            return t
        j -= 1
    return ""


def chain_end(f, k):
    """Last sig index of `NAME<chain>` starting with NAME at k."""
    j = k
    while f.T(j + 1) == "." and f.T(j + 3) == "(":
        m = f.T(j + 2)
        if m in CHAIN0 and f.T(j + 4) == ")":
            j += 4
        elif m in CHAIN1:
            j = f.mate(j + 3)
        else:
            break
    return j


def path_start(f, k):
    """First sig index of a `a.b.NAME` / `self.NAME` path ending at NAME (k)."""
    s = k
    while f.T(s - 1) == "." and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", f.T(s - 2) or ""):
        s -= 2
    return s


def once(src, name, lo, hi):
    f = F(src)
    n = len(f.sig)
    for k in range(n):
        if f.T(k) != name or not (lo <= f.off(k) < hi):
            continue
        if f.rw.toks[f.sig[k]][0] != "ident":
            continue
        nxt = f.T(k + 1)
        prev = f.T(k - 1)
        if nxt in (":", "(", "!", "::") or prev in ("let", "fn", "struct", "::"):
            continue
        if prev == "mut" and f.T(k - 2) != "&":
            continue  # `let mut NAME` / `mut NAME` parameter
        if (prev in (",", "{") and nxt in (",", "}")) and enclosing(f, k) == "{":
            continue  # struct-pattern / literal shorthand
        if prev == "(" and f.T(k - 2) == "Some" and nxt == ")":
            continue  # a `Some(NAME)` pattern binding, not a use
        s = path_start(f, k)
        e = chain_end(f, k)
        # `if let Some(..) = [&[mut]] NAME..`
        q = s
        if f.T(q - 1) == "mut" and f.T(q - 2) == "&":
            q -= 2
        elif f.T(q - 1) == "&":
            q -= 1
        if f.T(q - 1) == "=" and f.T(q - 2) == ")":
            o = f.mate(q - 2)
            if f.T(o - 1) == "Some" and f.T(o - 2) == "let" and f.T(o - 3) in ("if", "&&"):
                # find the `if`
                i = o - 3
                while f.T(i) != "if":
                    i -= 1
                    if i < 0:
                        break
                # body brace: first `{` at depth 0 after e
                b = e + 1
                while b < n and f.T(b) != "{":
                    if f.T(b) in ("(", "["):
                        b = f.mate(b)
                    b += 1
                close = f.mate(b)
                if f.T(close + 1) == "else":
                    # replace `if .. { A } else` with nothing, keeping the else branch
                    if f.T(close + 2) == "if":
                        f.rw.edits.append((f.off(i), f.off(close + 2), ""))
                    else:
                        f.rw.edits.append((f.off(i), f.off(close + 2), ""))
                else:
                    if f.T(i - 1) == "else":
                        f.rw.edits.append((f.off(i - 1), f.end(close), ""))
                    else:
                        f.rw.edits.append((f.rw.line_start(f.sig[i]), f.rw.after_line(f.sig[close]), ""))
                return f
        if f.T(e + 1) == "." and f.T(e + 2) in ("map_or", "map_or_else") and f.T(e + 3) == "(":
            close = f.mate(e + 3)
            # first argument: up to the first top-level comma
            a = e + 4
            b = a
            while f.T(b) != ",":
                if f.T(b) in ("(", "[", "{"):
                    b = f.mate(b)
                b += 1
            first = f.rw.src[f.off(a):f.end(b - 1)]
            if f.T(e + 2) == "map_or_else":
                if not first.startswith("||"):
                    continue
                first = first[2:].strip()
            f.rw.edits.append((f.off(s), f.end(close), first))
            return f
        if f.T(e + 1) == "." and f.T(e + 2) in ("is_some", "is_none") and f.T(e + 3) == "(" and f.T(e + 4) == ")":
            f.rw.edits.append((f.off(s), f.end(e + 4), "false" if f.T(e + 2) == "is_some" else "true"))
            return f
        if f.T(e + 1) == "=" or prev == "=" and f.T(s - 2) in ("let", "mut"):
            continue  # an assignment target or a binding's initializer side is left alone
        if e == k and nxt not in (",", ")", ";", "}", "]", "."):
            continue
        f.rw.edits.append((f.off(s), f.end(e), "None"))
        return f
    return None


def once_false(src, name, lo, hi):
    f = F(src)
    for k in range(len(f.sig)):
        if f.T(k) != name or not (lo <= f.off(k) < hi) or f.rw.toks[f.sig[k]][0] != "ident":
            continue
        nxt, prev = f.T(k + 1), f.T(k - 1)
        if nxt in (":", "(", "!", "::", "=") or prev in ("let", "mut", "fn", "struct", "::", ".", "&"):
            continue
        if prev == "(" and f.T(k - 2) == "Some" and nxt == ")":
            continue
        if prev in (",", "{") and nxt in (",", "}"):
            # struct-literal shorthand `NAME,` -> `NAME: false,`; a call argument stays `false`
            pass
        f.rw.edits.append((f.off(k), f.end(k), "false"))
        return f
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--false", dest="false_names", action="append", default=[],
                    help="a bool that is always false: replace its uses with `false`")
    ap.add_argument("--fn", action="append", default=[], help="restrict to these functions")
    ap.add_argument("file")
    a = ap.parse_args()
    p = Path(a.file)
    src = p.read_text()
    rounds = 0
    while True:
        changed = False
        ranges = [(0, len(src))]
        if a.fn:
            ranges = []
            for fname in a.fn:
                for m in re.finditer(r"\bfn\s+" + re.escape(fname) + r"\b", src):
                    j = src.index("{", m.end())
                    rw = Rewriter(src, {"feature": {}, "cfg": {}})
                    ti = next(i for i, t in enumerate(rw.toks) if t[1] == j)
                    ranges.append((m.start(), rw.toks[rw.match[ti]][2]))
        for name, fn_once in [(n, once) for n in a.name] + [(n, once_false) for n in a.false_names]:
            for lo, hi in ranges:
                f = fn_once(src, name, lo, hi)
                if f:
                    f.rw.edits = merge_overlaps(f.rw.edits)
                    src = f.rw.apply()
                    changed = True
                    rounds += 1
                    break
            if changed:
                break
        if not changed:
            break
    src, n = fold(src)
    p.write_text(re.sub(r"\n{3,}", "\n\n", src))
    print(f"{a.file}: {rounds} none-folds, {n} bool folds")


if __name__ == "__main__":
    main()
