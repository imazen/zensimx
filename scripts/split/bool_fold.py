#!/usr/bin/env python3
"""Fold boolean literals left behind by drop_fields.py (2026-10-28 slim-crate cut).

One rewrite per round, re-tokenizing between rounds, until nothing changes:

* `!false` -> `true`, `!true` -> `false`;
* `LIT && X`, `X && LIT`, `LIT || X`, `X || LIT`, with operands bounded by Rust precedence
  (`&&` binds tighter than `||`; comparisons bind tighter than both);
* `(LIT)` -> `LIT` where the parentheses are not a call;
* `if false { A }` -> nothing, `if false { A } else B` -> `B`, `if true { A } [else ..]` ->
  `{ A }`; `else if false { A }` -> nothing, `else if true { A } [else ..]` -> `else { A }`;
* `let NAME = LIT;` (no attributes, not `mut`) -> the binding is deleted and later uses of
  NAME in the same block become LIT (`NAME: LIT` in struct-literal shorthand), up to the
  next `let NAME`.

    bool_fold.py FILE...
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402

LIT = ("true", "false")
HARD_STOP = {"(", "[", "{", ",", ";", "=", "=>", ":", "return", "if", "while", "let", "match",
             "in", "else", "}", "|=", "&=", "+=", "-=", "break"}


class Folder:
    def __init__(self, src):
        self.src = src
        self.rw = Rewriter(src, {"feature": {}, "cfg": {}})
        rw = self.rw
        self.sig = [i for i in range(len(rw.toks)) if rw.toks[i][0] not in ("ws", "comment", "doc")]
        self.pos = {t: k for k, t in enumerate(self.sig)}

    def T(self, k):
        return self.rw.text(self.sig[k]) if 0 <= k < len(self.sig) else ""

    def kind(self, k):
        return self.rw.toks[self.sig[k]][0]

    def mate(self, k):
        return self.pos[self.rw.match[self.sig[k]]]

    def operand_start(self, k, op):
        """First sig index of the operand ending at sig index k, for operator op."""
        stops = HARD_STOP | ({"&&", "||"} if op == "&&" else {"||"})
        j = k
        while j >= 0:
            t = self.T(j)
            if t in (")", "]"):
                j = self.mate(j) - 1
                continue
            if t in stops:
                return j + 1
            j -= 1
        return 0

    def operand_end(self, k, op):
        stops = {",", ";", ")", "]", "}", "{", "=>", "else"} | ({"&&", "||"} if op == "&&" else {"||"})
        j = k
        while j < len(self.sig):
            t = self.T(j)
            if t in ("(", "["):
                j = self.mate(j) + 1
                continue
            if t in stops:
                return j - 1
            j += 1
        return len(self.sig) - 1

    def span(self, a, b):
        return self.rw.toks[self.sig[a]][1], self.rw.toks[self.sig[b]][2]

    def text(self, a, b):
        s, e = self.span(a, b)
        return self.src[s:e]

    def enclosing_open(self, k):
        j = k - 1
        while j >= 0:
            t = self.T(j)
            if t in (")", "]", "}"):
                j = self.mate(j) - 1
                continue
            if t in ("(", "[", "{"):
                return j
            j -= 1
        return -1

    def is_struct_literal_brace(self, k):
        """Sig index k is `{`: is it a struct-literal body?"""
        p = k - 1
        if p < 0:
            return False
        t = self.T(p)
        if t == "Self":
            return True
        if self.kind(p) == "ident" and t[:1].isupper():
            # `Foo {` / `path::Foo {` / `Foo::<T> {` (generic args handled loosely)
            return True
        if t == ">":
            return True
        return False

    def edit(self, a, b, rep):
        s, e = self.span(a, b)
        self.rw.edits.append((s, e, rep))

    def fold_once(self):
        T = self.T
        n = len(self.sig)
        for k in range(n):
            t = T(k)
            # !LIT
            if t == "!" and T(k + 1) in LIT:
                self.edit(k, k + 1, "true" if T(k + 1) == "false" else "false")
                return True
            # (LIT)
            if t == "(" and T(k + 1) in LIT and T(k + 2) == ")":
                prev = T(k - 1)
                if not (k > 0 and (self.kind(k - 1) == "ident" and prev not in ("if", "while", "return", "match", "in", "let")) or prev in (")", "]", ">")):
                    self.edit(k, k + 2, T(k + 1))
                    return True
            # && / ||
            if t in ("&&", "||"):
                ls = self.operand_start(k - 1, t)
                re_ = self.operand_end(k + 1, t)
                left_lit = ls == k - 1 and T(k - 1) in LIT
                right_lit = re_ == k + 1 and T(k + 1) in LIT
                if not (left_lit or right_lit):
                    continue
                ltxt = self.text(ls, k - 1)
                rtxt = self.text(k + 1, re_)
                if t == "&&":
                    if (left_lit and T(k - 1) == "false") or (right_lit and T(k + 1) == "false"):
                        rep = "false"
                    elif left_lit:
                        rep = rtxt
                    else:
                        rep = ltxt
                else:
                    if (left_lit and T(k - 1) == "true") or (right_lit and T(k + 1) == "true"):
                        rep = "true"
                    elif left_lit:
                        rep = rtxt
                    else:
                        rep = ltxt
                self.edit(ls, re_, rep)
                return True
            # if LIT { .. }
            if t == "if" and T(k + 1) in LIT and T(k + 2) == "{":
                lit = T(k + 1)
                close = self.mate(k + 2)
                has_else = T(close + 1) == "else"
                after_else_if = has_else and T(close + 2) == "if"
                chained = T(k - 1) == "else"
                # end of the whole remaining chain
                chain_end = close
                e = close + 1
                while T(e) == "else":
                    nx = e + 1
                    if T(nx) == "if":
                        while T(nx) != "{":
                            if T(nx) in ("(", "["):
                                nx = self.mate(nx)
                            nx += 1
                    chain_end = self.mate(nx)
                    e = chain_end + 1
                if lit == "false":
                    if chained:
                        # `else if false { A }` [else ..] -> [else ..]
                        if has_else:
                            self.rw.edits.append((self.span(k - 1, k - 1)[0], self.span(close + 1, close + 1)[0], ""))
                        else:
                            self.edit(k - 1, close, "")
                        return True
                    if not has_else:
                        s, e2 = self.span(k, close)
                        self.rw.edits.append((self.rw.line_start(self.sig[k]), self.rw.after_line(self.sig[close]), ""))
                        return True
                    # `if false { A } else ..` -> `..` (the else's `if` or block)
                    s = self.span(k, k)[0]
                    e2 = self.span(close + 2, close + 2)[0]
                    self.rw.edits.append((s, e2, ""))
                    return True
                else:
                    if chained:
                        # `else if true { A } ..` -> `else { A }`
                        self.rw.edits.append((self.span(k, k)[0], self.span(k + 2, k + 2)[0], ""))
                        if chain_end != close:
                            self.rw.edits.append((self.span(close, close)[1], self.span(chain_end, chain_end)[1], ""))
                        return True
                    self.rw.edits.append((self.span(k, k)[0], self.span(k + 2, k + 2)[0], ""))
                    if chain_end != close:
                        self.rw.edits.append((self.span(close, close)[1], self.span(chain_end, chain_end)[1], ""))
                    return True
            # let NAME = LIT;
            if t == "let" and self.kind(k + 1) == "ident" and T(k + 1) != "mut" and T(k + 2) == "=" and T(k + 3) in LIT and T(k + 4) == ";":
                # no attributes on the statement
                if T(k - 1) == "]":
                    continue
                name, val = T(k + 1), T(k + 3)
                op = self.enclosing_open(k)
                end = self.mate(op) if op >= 0 else n
                self.rw.edits.append((self.rw.line_start(self.sig[k]), self.rw.after_line(self.sig[k + 4]), ""))
                q = k + 5
                while q < end:
                    tq = T(q)
                    if tq == "let" and T(q + 1) == name:
                        inner = self.enclosing_open(q)
                        if inner == op:
                            break  # shadowed in the same block
                        q = self.mate(inner) + 1  # shadowed in a nested block: skip it
                        continue
                    if tq == name and self.kind(q) == "ident":
                        before, after = T(q - 1), T(q + 1)
                        if before in (".", "::") or after in ("(", "::", "!"):
                            q += 1
                            continue
                        if after == ":" and before in ("{", ","):
                            q += 1
                            continue  # field label in a struct literal
                        if before in ("{", ",") and after in (",", "}"):
                            o = self.enclosing_open(q)
                            if o >= 0 and T(o) == "{" and self.is_struct_literal_brace(o):
                                self.edit(q, q, f"{name}: {val}")
                                q += 1
                                continue
                        self.edit(q, q, val)
                    q += 1
                return True
        return False


def fold(src):
    rounds = 0
    while True:
        f = Folder(src)
        if not f.fold_once():
            return src, rounds
        f.rw.edits = merge_overlaps(f.rw.edits)
        src = f.rw.apply()
        rounds += 1


def main():
    for f in sys.argv[1:]:
        p = Path(f)
        src, rounds = fold(p.read_text())
        p.write_text(re.sub(r"\n{3,}", "\n\n", src))
        print(f"{f}: {rounds} folds")


if __name__ == "__main__":
    main()
