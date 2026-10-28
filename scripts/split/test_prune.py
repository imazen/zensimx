#!/usr/bin/env python3
"""Remove test code that no longer compiles after the slim-crate cut (2026-10-28).

Reads `cargo test --no-run --message-format=json` output. For every error, the innermost
enclosing item inside test-only code is deleted: a `#[test]` function, a helper function or
`use` declaration inside a `#[cfg(test)]` module, or a `const`/`static` there. Errors outside
test-only code are reported and left alone, because they mean the library itself is wrong.
Iterate until the build is clean: deleting a helper turns its callers into the next round's
errors.

    test_prune.py --root <workspace> [--dry-run] LOG.json
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402

ITEMS = {"fn", "use", "const", "static", "struct", "enum", "impl", "mod", "type", "trait", "macro_rules"}


def errors(log):
    out = defaultdict(set)
    for line in open(log):
        if not line.startswith("{"):
            continue
        m = json.loads(line)
        if m.get("reason") != "compiler-message":
            continue
        msg = m["message"]
        if msg["level"] != "error":
            continue
        for sp in msg["spans"]:
            if sp.get("is_primary"):
                out[sp["file_name"]].add((sp["byte_start"], msg["message"]))
    return out


def item_ranges(rw):
    """(start_tok, end_tok, kind, attrs_text, body_open_tok) for every item, nested ones too."""
    res = []
    toks = rw.toks
    for i, (k, s, e) in enumerate(toks):
        if k != "ident" or rw.text(i) not in ITEMS:
            continue
        kind = rw.text(i)
        prev = rw.sig(i - 1, -1)
        if kind == "fn" and prev >= 0 and rw.text(prev) in ("impl", "dyn", "&", "(", ",", "<", "->", ":", "="):
            continue  # fn pointer type
        if kind == "impl" and prev >= 0 and rw.text(prev) in ("->", ":", "(", ",", "<", "&", "="):
            continue  # impl Trait in type position
        if kind == "use" and prev >= 0 and rw.text(prev) == "+":
            continue  # precise capturing `+ use<...>`
        # item end
        j = i
        while j < len(toks) and rw.text(j) not in ("{", ";"):
            if rw.text(j) in ("(", "["):
                j = rw.match[j]
            j += 1
        if j >= len(toks):
            continue
        end = rw.match[j] if rw.text(j) == "{" else j
        body_open = j if rw.text(j) == "{" else None
        start = rw.leading_start(i)
        attrs = rw.src[rw.toks[start][1]:s]
        res.append((start, end, kind, attrs, body_open, i))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("log")
    a = ap.parse_args()
    errs = errors(a.log)
    total_removed = 0
    unresolved = []
    for f, items in sorted(errs.items()):
        p = Path(a.root) / f
        raw = p.read_bytes()
        src = raw.decode("utf-8")
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        ranges = item_ranges(rw)
        whole_file_test = bool(re.search(r"(?m)^\s*#!\[cfg\(test\)\]", src))
        chosen = set()
        for b, text in sorted(items):
            off = len(raw[:b].decode("utf-8"))
            enclosing = [r for r in ranges if rw.toks[r[0]][1] <= off <= rw.toks[r[1]][2]]
            if not enclosing:
                unresolved.append((f, src.count("\n", 0, off) + 1, text))
                continue
            enclosing.sort(key=lambda r: rw.toks[r[0]][1])  # outermost first
            # test context: an enclosing item with #[test] / #[cfg(test)] / cfg(all(test..)),
            # or a file included only into a test module. (feature_v2_oracle.rs is NOT one:
            # it also holds the production era-2 dense kernels.)
            in_test = whole_file_test or f.endswith("_tests.rs")
            target = None
            for r in enclosing:
                attrs = r[3]
                if re.search(r"#\[(test|cfg\((all\()?test)", attrs) or re.search(r"#\[cfg\(any\(test", attrs):
                    in_test = True
                if in_test and r[2] != "mod":
                    target = r  # keep descending: innermost non-module item in test context wins
                    if r[2] == "fn" and re.search(r"#\[test\]", attrs):
                        break
            if target is None:
                unresolved.append((f, src.count("\n", 0, off) + 1, text))
                continue
            chosen.add((target[0], target[1]))
        for st, en in chosen:
            line = src.count("\n", 0, rw.toks[st][1]) + 1
            print(f"remove {f}:{line}")
            rw.edits.append((rw.line_start(st), rw.after_line(en), ""))
        total_removed += len(chosen)
        if rw.edits and not a.dry_run:
            rw.edits = merge_overlaps(rw.edits)
            p.write_text(re.sub(r"\n{3,}", "\n\n", rw.apply()))
    for f, line, text in unresolved:
        print(f"UNRESOLVED {f}:{line}: {text}")
    print(f"removed {total_removed} items; unresolved {len(unresolved)}")


if __name__ == "__main__":
    main()
