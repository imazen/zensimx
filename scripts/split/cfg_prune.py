#!/usr/bin/env python3
"""Fix Cargo features to constants in Rust source and delete the code that becomes dead.

Used once to cut the slim published crate (imazen/zensim, 2026-10-28) out of this research
crate: features the slim crate does not have are fixed to true or false; every `#[cfg(..)]`
and `#[cfg_attr(..)]` whose predicate becomes constant is resolved, false items are deleted
with the doc comments and attributes attached to them, and `cfg!(..)` macros with a constant
value become `true` / `false`. Predicates that still depend on a live feature are simplified
and kept.

    cfg_prune.py --set feature-regime-v2=1 --set custom-profiles=0 [--set name=0 ...] FILE...

`--set name=v` fixes `feature = "name"`; `--set cfg:name=v` fixes a bare cfg such as
`steerfix_instrument`. Files are rewritten in place; `--check` only reports. A file whose
inner `#![cfg(..)]` becomes false is reported as DEAD-FILE and left for the caller to delete,
along with its `mod` declaration if that was not gated the same way.

The scanner tokenizes enough Rust to skip comments, strings, raw strings, byte strings, char
literals and lifetimes, so brackets inside them never count. Item extents are found by bracket
depth: an item ends at `;` or `,` at depth 0, at the closing bracket of the enclosing group,
or after a top-level `}` that is not followed by a continuation (`else`, `.`, `?`, `as`, an
operator). Run the compiler afterwards: a wrong extent shows up as a build error.
"""

import argparse
import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------------- tokenizer

KEYWORD_ITEM_END_AT_BRACE = {"fn", "impl", "trait", "mod", "enum", "union", "struct", "macro_rules"}


def tokenize(src):
    """Return a list of (kind, start, end). kinds: ws, comment, doc, str, char, life, ident,
    num, punct. Only what bracket matching and item scanning need."""
    toks = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            j = i
            while j < n and src[j] in " \t\r\n":
                j += 1
            toks.append(("ws", i, j))
            i = j
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            kind = "doc" if (src.startswith("///", i) and not src.startswith("////", i)) or src.startswith("//!", i) else "comment"
            toks.append((kind, i, j))
            i = j
            continue
        if src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            kind = "doc" if src.startswith("/**", i) or src.startswith("/*!", i) else "comment"
            toks.append((kind, i, j))
            i = j
            continue
        m = re.match(r'b?r(#*)"', src[i:])
        if m:
            hashes = m.group(1)
            close = '"' + hashes
            j = src.find(close, i + m.end())
            j = n if j < 0 else j + len(close)
            toks.append(("str", i, j))
            i = j
            continue
        if c == '"' or (c == "b" and i + 1 < n and src[i + 1] == '"'):
            j = i + (2 if c == "b" else 1)
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            toks.append(("str", i, j))
            i = j
            continue
        if c == "'" or (c == "b" and i + 1 < n and src[i + 1] == "'"):
            # char literal or lifetime
            k = i + (2 if c == "b" else 1)
            m = re.match(r"(\\(x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\}|.)|[^\\'\n])'", src[k:])
            if m:
                j = k + m.end()
                toks.append(("char", i, j))
                i = j
                continue
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", src[k:])
            j = k + (m.end() if m else 0)
            toks.append(("life", i, j))
            i = j
            continue
        m = re.match(r"r#[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*", src[i:])
        if m:
            toks.append(("ident", i, i + m.end()))
            i += m.end()
            continue
        m = re.match(r"[0-9][0-9A-Za-z_]*(\.[0-9][0-9A-Za-z_]*)?([eE][+-]?[0-9_]+)?", src[i:])
        if m:
            toks.append(("num", i, i + m.end()))
            i += m.end()
            continue
        for p in ("::", "=>", "->", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "..=", "...", ".."):
            if src.startswith(p, i):
                toks.append(("punct", i, i + len(p)))
                i += len(p)
                break
        else:
            toks.append(("punct", i, i + 1))
            i += 1
    return toks


OPEN = {"(": ")", "[": "]", "{": "}"}
CLOSE = {")", "]", "}"}


# ------------------------------------------------------------------------- predicates

class Pred:
    """Parsed cfg predicate. eval() returns True, False or a simplified source string."""

    def __init__(self, kind, args=None, name=None, value=None):
        self.kind, self.args, self.name, self.value = kind, args or [], name, value


def parse_pred(text):
    toks = [t for t in re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z0-9_-]*|[(),=]', text)]
    pos = 0

    def parse():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t in ("all", "any", "not") and pos < len(toks) and toks[pos] == "(":
            pos += 1
            args = []
            while toks[pos] != ")":
                args.append(parse())
                if toks[pos] == ",":
                    pos += 1
            pos += 1
            return Pred(t, args)
        if pos < len(toks) and toks[pos] == "=":
            pos += 1
            v = toks[pos]
            pos += 1
            return Pred("kv", name=t, value=v.strip('"'))
        return Pred("name", name=t)

    p = parse()
    if pos != len(toks):
        raise ValueError(f"trailing tokens in cfg predicate: {text!r}")
    return p


def evaluate(p, fixed):
    """Three-valued: True / False / str (simplified predicate)."""
    if p.kind == "kv":
        key = f"{p.name}={p.value}"
        if p.name == "feature" and p.value in fixed["feature"]:
            return fixed["feature"][p.value]
        return f'{p.name} = "{p.value}"'
    if p.kind == "name":
        if p.name in fixed["cfg"]:
            return fixed["cfg"][p.name]
        return p.name
    vals = [evaluate(a, fixed) for a in p.args]
    if p.kind == "not":
        v = vals[0]
        return (not v) if isinstance(v, bool) else f"not({v})"
    if p.kind == "all":
        if any(v is False for v in vals):
            return False
        rest = [v for v in vals if v is not True]
        if not rest:
            return True
        return rest[0] if len(rest) == 1 else "all(" + ", ".join(rest) + ")"
    if p.kind == "any":
        if any(v is True for v in vals):
            return True
        rest = [v for v in vals if v is not False]
        if not rest:
            return False
        return rest[0] if len(rest) == 1 else "any(" + ", ".join(rest) + ")"
    raise ValueError(p.kind)


def split_top_commas(text):
    parts, depth, cur, i = [], 0, [], 0
    toks = tokenize(text)
    last = 0
    for kind, s, e in toks:
        t = text[s:e]
        if kind == "punct" and t in OPEN:
            depth += 1
        elif kind == "punct" and t in CLOSE:
            depth -= 1
        elif kind == "punct" and t == "," and depth == 0:
            parts.append(text[last:s].strip())
            last = e
    parts.append(text[last:].strip())
    return [p for p in parts if p]


# ------------------------------------------------------------------------- rewriting

class Rewriter:
    def __init__(self, src, fixed):
        self.src = src
        self.fixed = fixed
        self.toks = tokenize(src)
        # index of matching bracket for every bracket token
        self.match = {}
        stack = []
        for idx, (kind, s, e) in enumerate(self.toks):
            if kind != "punct":
                continue
            t = src[s:e]
            if t in OPEN:
                stack.append(idx)
            elif t in CLOSE:
                if not stack:
                    raise ValueError(f"unbalanced {t!r} at offset {s}")
                o = stack.pop()
                self.match[o] = idx
                self.match[idx] = o
        if stack:
            raise ValueError("unbalanced brackets at end of file")
        self.edits = []  # (start, end, replacement)
        self.dead_file = False

    def text(self, idx):
        k, s, e = self.toks[idx]
        return self.src[s:e]

    def sig(self, idx, step=1):
        """Next significant token index from idx (inclusive) in direction step."""
        while 0 <= idx < len(self.toks) and self.toks[idx][0] in ("ws", "comment", "doc"):
            idx += step
        return idx

    def attr_at(self, idx):
        """If an attribute starts at token idx (`#` [`!`] `[`), return (inner, open_idx, close_idx)."""
        if self.text(idx) != "#":
            return None
        j = idx + 1
        inner = False
        if j < len(self.toks) and self.text(j) == "!":
            inner, j = True, j + 1
        if j < len(self.toks) and self.text(j) == "[":
            return inner, j, self.match[j]
        return None

    def run(self):
        idx = 0
        n = len(self.toks)
        consumed_until = -1
        while idx < n:
            if idx <= consumed_until:
                idx += 1
                continue
            a = self.attr_at(idx)
            if not a:
                if self.toks[idx][0] == "ident" and self.text(idx) == "cfg":
                    self.cfg_macro(idx)
                idx += 1
                continue
            inner, o, c = a
            body_start = self.sig(o + 1)
            head = self.text(body_start)
            if head not in ("cfg", "cfg_attr"):
                idx = c + 1
                continue
            paren = self.sig(body_start + 1)
            if self.text(paren) != "(":
                idx = c + 1
                continue
            pclose = self.match[paren]
            ptext = self.src[self.toks[paren][2]:self.toks[pclose][1]]
            if head == "cfg":
                v = evaluate(parse_pred(ptext), self.fixed)
                if v is True:
                    self.delete_attr(idx, c)
                    consumed_until = c
                elif v is False:
                    if inner:
                        self.dead_file = True
                        return
                    end = self.item_end(c + 1)
                    start = self.leading_start(idx)
                    self.edits.append((self.line_start(start), self.after_line(end), ""))
                    consumed_until = end
                else:
                    self.edits.append((self.toks[paren][2], self.toks[pclose][1], v))
                    consumed_until = c
            else:  # cfg_attr
                parts = split_top_commas(ptext)
                v = evaluate(parse_pred(parts[0]), self.fixed)
                attrs = parts[1:]
                bang = "!" if inner else ""
                if v is True:
                    rep = "\n".join(f"#{bang}[{x}]" for x in attrs)
                    self.edits.append((self.toks[idx][1], self.toks[c][2], rep))
                elif v is False:
                    self.delete_attr(idx, c)
                else:
                    self.edits.append((self.toks[paren][2], self.toks[pclose][1], ", ".join([v] + attrs)))
                consumed_until = c
            idx = consumed_until + 1
        return

    def cfg_macro(self, idx):
        bang = self.sig(idx + 1)
        if bang >= len(self.toks) or self.text(bang) != "!":
            return
        paren = self.sig(bang + 1)
        if paren >= len(self.toks) or self.text(paren) != "(":
            return
        # skip `cfg` that is part of a path or an attribute (handled elsewhere)
        prev = self.sig(idx - 1, -1)
        if prev >= 0 and self.text(prev) in ("::", "["):
            return
        pclose = self.match[paren]
        ptext = self.src[self.toks[paren][2]:self.toks[pclose][1]]
        try:
            v = evaluate(parse_pred(ptext), self.fixed)
        except Exception:
            return
        if isinstance(v, bool):
            self.edits.append((self.toks[idx][1], self.toks[pclose][2], "true" if v else "false"))
        else:
            self.edits.append((self.toks[paren][2], self.toks[pclose][1], v))

    def delete_attr(self, idx, c):
        s, e = self.toks[idx][1], self.toks[c][2]
        ls = self.line_start_offset(s)
        le = self.src.find("\n", e)
        le = len(self.src) if le < 0 else le
        if self.src[ls:s].strip() == "" and self.src[e:le].strip() == "":
            self.edits.append((ls, le + 1 if le < len(self.src) else le, ""))
        else:
            self.edits.append((s, e, ""))

    def line_start_offset(self, off):
        p = self.src.rfind("\n", 0, off)
        return p + 1

    def line_start(self, tok_idx):
        off = self.toks[tok_idx][1]
        ls = self.line_start_offset(off)
        return ls if self.src[ls:off].strip() == "" else off

    def after_line(self, tok_idx):
        off = self.toks[tok_idx][2]
        le = self.src.find("\n", off)
        if le < 0:
            return len(self.src)
        return le + 1 if self.src[off:le].strip() == "" else off

    def leading_start(self, attr_idx):
        """Extend backwards over doc comments, comments and attributes directly attached
        (no blank line) to the attribute at attr_idx."""
        start = attr_idx
        i = attr_idx - 1
        while i >= 0:
            kind, s, e = self.toks[i]
            if kind == "ws":
                if self.src[s:e].count("\n") > 1:
                    break
                i -= 1
                continue
            if kind in ("doc", "comment"):
                # a trailing comment on a code line is not attached
                ls = self.line_start_offset(s)
                code = self.src[ls:s].strip()
                if code and not code.startswith("#["):
                    break  # a trailing comment on a code line is not attached
                if code:
                    i -= 1  # trailing comment on an attribute line: the attribute follows
                    continue
                start = i
                i -= 1
                continue
            if kind == "punct" and self.text(i) == "]" and i in self.match:
                o = self.match[i]
                h = o - 1
                if h >= 0 and self.text(h) == "!":
                    h -= 1
                if h >= 0 and self.text(h) == "#":
                    start = h
                    i = h - 1
                    continue
            break
        return start

    CONT = {"else", ".", "?", "as", "=>", "=", ":", "@", "|", "||", "&&", "+", "-", "*", "/", "%",
            "==", "!=", "<", ">", "<=", ">=", "&", "^", "..", "..="}

    def generic_open(self, i):
        """`<` glued to an identifier, `::` or `>` opens generics (rustfmt style); a spaced `<`
        is a comparison."""
        if i == 0:
            return False
        pk, ps, pe = self.toks[i - 1]
        if pe != self.toks[i][1]:
            return False
        return pk == "ident" or self.src[ps:pe] in ("::", ">")

    def guard_ahead(self, i):
        """At an `if` after a brace pattern: a match guard reaches `=>` before any `{`."""
        j = i + 1
        while j < len(self.toks):
            k = self.toks[j][0]
            if k in ("ws", "comment", "doc"):
                j += 1
                continue
            t = self.text(j)
            if t == "=>":
                return True
            if t in ("{", ";"):
                return False
            if t in OPEN:
                j = self.match[j] + 1
                continue
            if t in CLOSE:
                return False
            j += 1
        return False

    def item_end(self, after_attr):
        """Token index of the last token of the item following an outer attribute."""
        i = self.sig(after_attr)
        # further attributes / docs belong to the item
        while i < len(self.toks):
            a = self.attr_at(i)
            if not a:
                break
            i = self.sig(a[2] + 1)
        start = i
        # classify by leading keywords
        j = start
        words = []
        while j < len(self.toks) and len(words) < 8:
            k = self.toks[j][0]
            if k in ("ws", "comment", "doc"):
                j += 1
                continue
            t = self.text(j)
            if k == "ident":
                words.append(t)
                j += 1
                if t == "pub" and self.text(self.sig(j)) == "(":
                    j = self.match[self.sig(j)] + 1
                continue
            if k == "str":  # extern "C"
                j += 1
                continue
            if t == "!" and words and words[-1] == "macro_rules":
                j += 1
                continue
            break
        kw = [w for w in words if w not in ("pub", "unsafe", "async", "const", "extern", "default", "static", "crate", "super", "self", "in")]
        first = kw[0] if kw else (words[0] if words else None)
        brace_item = first in KEYWORD_ITEM_END_AT_BRACE
        angle = 0
        i = start
        while i < len(self.toks):
            kind = self.toks[i][0]
            if kind in ("ws", "comment", "doc"):
                i += 1
                continue
            t = self.text(i)
            if kind == "punct" and t in OPEN:
                close = self.match[i]
                if t == "{":
                    nxt = self.sig(close + 1)
                    nt = self.text(nxt) if nxt < len(self.toks) else ""
                    if brace_item:
                        return nxt if nt == ";" else close
                    if nt in (";", ","):
                        return nxt
                    if nt in self.CONT or (nt == "if" and self.guard_ahead(nxt)):
                        i = nxt
                        continue
                    return close
                i = close + 1
                continue
            if kind == "punct" and t in CLOSE:
                # the enclosing group closes: the item ends just before it
                return self.sig(i - 1, -1)
            if kind == "punct" and t == ";":
                return i
            if kind == "punct" and t == "," and not brace_item and angle == 0:
                return i
            if kind == "punct" and t == "<" and self.generic_open(i):
                angle += 1
            elif kind == "punct" and t == ">" and angle > 0:
                angle -= 1
            elif kind == "punct" and t == "=>" and not brace_item:
                body = self.sig(i + 1)
                if body < len(self.toks) and self.text(body) == "{":
                    close = self.match[body]
                    nxt = self.sig(close + 1)
                    return nxt if nxt < len(self.toks) and self.text(nxt) == "," else close
            i += 1
        return len(self.toks) - 1

    def apply(self):
        out = self.src
        for s, e, rep in sorted(self.edits, key=lambda x: (x[0], x[1]), reverse=True):
            out = out[:s] + rep + out[e:]
        return out


def merge_overlaps(edits):
    edits = sorted(edits)
    merged = []
    for s, e, r in edits:
        if merged and s < merged[-1][1]:
            ps, pe, pr = merged[-1]
            if e <= pe:
                continue  # nested inside a deletion
            raise ValueError(f"overlapping edits {ps}-{pe} and {s}-{e}")
        merged.append((s, e, r))
    return merged


def prune(src, fixed):
    rw = Rewriter(src, fixed)
    rw.run()
    if rw.dead_file:
        return None
    rw.edits = merge_overlaps(rw.edits)
    out = rw.apply()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], help="feature=0|1 or cfg:name=0|1")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    fixed = {"feature": {}, "cfg": {}}
    for s in a.set:
        k, v = s.split("=", 1)
        val = v in ("1", "true", "on")
        if k.startswith("cfg:"):
            fixed["cfg"][k[4:]] = val
        else:
            fixed["feature"][k] = val
    rc = 0
    for f in a.files:
        p = Path(f)
        src = p.read_text()
        try:
            out = prune(src, fixed)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {f}: {e}", file=sys.stderr)
            rc = 1
            continue
        if out is None:
            print(f"DEAD-FILE {f}")
            continue
        if out != src:
            print(f"changed {f}")
            if not a.check:
                p.write_text(out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
