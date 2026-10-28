#!/usr/bin/env python3
"""Delete items rustc reports as dead code in EVERY given build configuration.

Companion to cfg_prune.py / remove_items.py for the 2026-10-28 slim-crate cut. Each input is
the `--message-format=json` output of one `cargo check` of the same source tree (different
features or targets). A value item (function, method, const, static, macro) is deleted only if every log
reports it unused at the same byte offset, so code that
one configuration still needs (a no-threads fallback, an aarch64 kernel) survives. Types, fields and
enum variants are listed, not deleted.

    dead_prune.py --root <workspace> [--dry-run] LOG.json...
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402

# Types can stay referenced after their constructors die (signatures, impls, fields), so
# only value items are deleted automatically; dead types are reported for manual removal.
ITEM_KW = {"fn", "const", "static", "macro_rules"}


def dead_spans(log):
    out = {}
    for line in open(log):
        line = line.strip()
        if not line.startswith("{"):
            continue
        m = json.loads(line)
        if m.get("reason") != "compiler-message":
            continue
        msg = m["message"]
        code = (msg.get("code") or {}).get("code")
        if code != "dead_code":
            continue
        text = msg["message"]
        for sp in msg["spans"]:
            if not sp.get("is_primary"):
                continue
            key = (sp["file_name"], sp["byte_start"])
            out[key] = (text, sp["byte_end"], sp["line_start"])
    return out


def item_bounds(rw, byte_start):
    """Token range (start, end) of the item whose name token starts at byte_start."""
    idx = next((i for i, (_, s, _e) in enumerate(rw.toks) if s == byte_start), None)
    if idx is None:
        return None
    k = rw.sig(idx - 1, -1)
    if k < 0:
        return None
    kw = rw.text(k)
    if kw == "!" :  # macro_rules! NAME
        k = rw.sig(k - 1, -1)
        kw = rw.text(k)
    if kw not in ITEM_KW:
        return None  # a field or variant
    start = k
    b = rw.sig(k - 1, -1)
    while b >= 0 and (rw.text(b) in ("pub", "const", "unsafe", "async", "extern", "default") or
                      (rw.text(b) == ")" and rw.text(rw.sig(rw.match[b] - 1, -1)) == "pub")):
        if rw.text(b) == ")":
            b = rw.sig(rw.match[b] - 1, -1)
        start = b
        b = rw.sig(b - 1, -1)
    j = idx
    while j < len(rw.toks) and rw.text(j) not in ("{", ";"):
        if rw.text(j) in ("(", "["):
            j = rw.match[j]
        j += 1
    if j >= len(rw.toks):
        return None
    if rw.text(j) == "{":
        end = rw.match[j]
        nxt = rw.sig(end + 1)
        if nxt < len(rw.toks) and rw.text(nxt) == ";":
            end = nxt
    else:
        end = j
    return start, end


def in_test_context(rw, tok):
    """Whether the item at token `tok` is test-only: it, or an enclosing module, carries
    `#[cfg(test)]` (or a cfg whose every branch needs test)."""
    off = rw.toks[tok][1]
    test_attr = re.compile(r"#\[cfg\((test|all\(test)")
    lead = rw.leading_start(tok)
    if test_attr.search(rw.src[rw.toks[lead][1]:off]):
        return True
    for i, (k, s, e) in enumerate(rw.toks):
        if s >= off:
            break
        if k == "ident" and rw.text(i) == "mod":
            j = rw.sig(i + 1)
            j = rw.sig(j + 1)
            if j < len(rw.toks) and rw.text(j) == "{" and rw.toks[rw.match[j]][2] > off:
                vis = i
                b = rw.sig(i - 1, -1)
                if b >= 0 and rw.text(b) == ")" and rw.text(rw.sig(rw.match[b] - 1, -1)) == "pub":
                    vis = rw.sig(rw.match[b] - 1, -1)
                elif b >= 0 and rw.text(b) == "pub":
                    vis = b
                ml = rw.leading_start(vis)
                if test_attr.search(rw.src[rw.toks[ml][1]:s]):
                    return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-only", action="store_true",
                    help="delete only items inside #[cfg(test)] code (use with test-build logs)")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    sets = [dead_spans(l) for l in a.logs]
    common = set(sets[0])
    for s in sets[1:]:
        common &= set(s)
    by_file = defaultdict(list)
    for f, b in common:
        by_file[f].append((b, sets[0][(f, b)]))
    removed = skipped = 0
    for f, items in sorted(by_file.items()):
        p = Path(a.root) / f
        raw = p.read_bytes()
        src = raw.decode("utf-8")
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        for b, (text, _be, line) in sorted(items):
            bounds = item_bounds(rw, len(raw[:b].decode("utf-8")))
            if bounds is None:
                print(f"skip {f}:{line}: {text}")
                skipped += 1
                continue
            st, en = bounds
            if a.test_only and not in_test_context(rw, st):
                print(f"skip-nontest {f}:{line}: {text}")
                skipped += 1
                continue
            lead = rw.leading_start(st)
            rw.edits.append((rw.line_start(lead), rw.after_line(en), ""))
            print(f"remove {f}:{line}: {text}")
            removed += 1
            # A hand-written tier variant (`NAME_scalar`, `NAME_v3`, ...) of a `#[magetypes]` /
            # `#[arcane]` family: rustc never reports the macro-generated siblings as unused,
            # so a dead fallback means the whole family is unused. Remove the generic source
            # item and every hand-written sibling with it; the next compile proves it.
            m = re.match(r"(?:function|method) `([A-Za-z0-9_]+?)_(scalar|v1|v2|v3|v4|v4x|neon|wasm128)`", text)
            if m:
                stem = m.group(1)
                for i, (k, s0, e0) in enumerate(rw.toks):
                    if k != "ident" or rw.text(i) != "fn":
                        continue
                    nm = rw.sig(i + 1)
                    name = rw.text(nm)
                    if name == stem or re.fullmatch(re.escape(stem) + r"_(scalar|v1|v2|v3|v4|v4x|neon|wasm128)", name):
                        fb = item_bounds(rw, rw.toks[nm][1])
                        if fb is None:
                            continue
                        fl = rw.leading_start(fb[0])
                        rw.edits.append((rw.line_start(fl), rw.after_line(fb[1]), ""))
                        print(f"remove-family {f}: fn {name}")
        if rw.edits and not a.dry_run:
            rw.edits = merge_overlaps(rw.edits)
            out = re.sub(r"\n{3,}", "\n\n", rw.apply())
            p.write_text(out)
    print(f"removed {removed}, skipped {skipped}, configs {len(sets)}, dead-in-all {len(common)}")


if __name__ == "__main__":
    main()
