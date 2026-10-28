#!/usr/bin/env python3
"""Delete match arms whose pattern starts with a given path (e.g. `FrontEnd::Hdr`).

Companion to cfg_prune.py for the 2026-10-28 slim-crate cut: after an enum variant is
removed, every arm matching it must go. An arm is recognised by the path tokens appearing
right after `{`, `,` or `}` (an arm boundary) with `=>` following at the same depth; its end
is found with cfg_prune's item-extent rules (`=> { .. }` with an optional comma, or the
expression up to `,` or the closing brace). Each removal is reported.

    remove_arms.py --pattern FrontEnd::Hdr [--pattern ...] FILE...
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps, tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", action="append", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    total = 0
    for f in a.files:
        p = Path(f)
        src = p.read_text()
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        sig = [i for i, t in enumerate(rw.toks) if t[0] not in ("ws", "comment", "doc")]
        texts = [rw.text(i) for i in sig]
        for pat in a.pattern:
            ptoks = [src_t for src_t in (pat[s:e] for _, s, e in tokenize(pat)) if src_t.strip()]
            n = len(ptoks)
            for k in range(len(sig) - n):
                if texts[k:k + n] != ptoks:
                    continue
                if k > 0 and texts[k - 1] not in ("{", ",", "}", "|"):
                    continue
                # an or-pattern member (`A | B =>`) is not removed as a whole arm
                if k > 0 and texts[k - 1] == "|":
                    continue
                # `=>` must follow before the next `,`/`;` at this depth
                j = sig[k]
                seen_arrow = False
                while j < len(rw.toks):
                    t = rw.text(j)
                    if t in ("(", "[", "{") and rw.toks[j][0] == "punct":
                        j = rw.match[j] + 1
                        continue
                    if t == "=>":
                        seen_arrow = True
                        break
                    if t in (",", ";", ")", "]", "}"):
                        break
                    j += 1
                if not seen_arrow:
                    continue
                start = rw.leading_start(sig[k])
                end = rw.item_end(sig[k])
                line = src.count("\n", 0, rw.toks[sig[k]][1]) + 1
                print(f"{f}:{line}: removing arm {pat}")
                rw.edits.append((rw.line_start(start), rw.after_line(end), ""))
                total += 1
        if rw.edits:
            rw.edits = merge_overlaps(rw.edits)
            p.write_text(re.sub(r"\n{3,}", "\n\n", rw.apply()))
    print(f"removed {total} arms")


if __name__ == "__main__":
    main()
