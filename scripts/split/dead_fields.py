#!/usr/bin/env python3
"""Delete struct fields rustc reports as never read in EVERY build configuration
(2026-10-28 slim cut). Initializers and writes that then fail to compile are removed by
drop_fields.py from the next `cargo check --message-format=short` log.

    dead_fields.py --root <workspace> [--dry-run] LOG.json...
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402


def field_spans(log):
    out = set()
    for line in open(log):
        if not line.startswith("{"):
            continue
        m = json.loads(line)
        if m.get("reason") != "compiler-message":
            continue
        msg = m["message"]
        if (msg.get("code") or {}).get("code") != "dead_code" or "never read" not in msg["message"]:
            continue
        for sp in msg["spans"]:
            if sp.get("is_primary"):
                out.add((sp["file_name"], sp["byte_start"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    sets = [field_spans(l) for l in a.logs]
    common = set.intersection(*sets)
    by_file = defaultdict(list)
    for f, b in common:
        by_file[f].append(b)
    n = 0
    for f, offs in sorted(by_file.items()):
        p = Path(a.root) / f
        raw = p.read_bytes()
        src = raw.decode("utf-8")
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        for b in sorted(offs):
            off = len(raw[:b].decode("utf-8"))
            i = next((k for k, t in enumerate(rw.toks) if t[1] == off), None)
            if i is None:
                continue
            colon = rw.sig(i + 1)
            if rw.text(colon) != ":":
                # tuple-struct field `0` and the like: report only
                print(f"skip {f}:{src.count(chr(10), 0, off) + 1} {rw.text(i)}")
                continue
            # field extent: up to the `,` at depth 0 or the closing brace
            j = colon + 1
            depth = 0
            while j < len(rw.toks):
                t = rw.text(j)
                if t in ("(", "[", "{"):
                    j = rw.match[j] + 1
                    continue
                if t == "<":
                    depth += 1
                elif t == ">":
                    depth -= 1
                elif t == ">>":
                    depth -= 2
                elif t in (",", "}") and depth <= 0:
                    break
                j += 1
            end = j if rw.text(j) == "," else rw.sig(j - 1, -1)
            # visibility before the name
            start = i
            b0 = rw.sig(i - 1, -1)
            if rw.text(b0) == ")" and rw.text(rw.sig(rw.match[b0] - 1, -1)) == "pub":
                start = rw.sig(rw.match[b0] - 1, -1)
            elif rw.text(b0) == "pub":
                start = b0
            lead = rw.leading_start(start)
            rw.edits.append((rw.line_start(lead), rw.after_line(end), ""))
            print(f"remove {f}:{src.count(chr(10), 0, off) + 1} field {rw.text(i)}")
            n += 1
        if rw.edits and not a.dry_run:
            rw.edits = merge_overlaps(rw.edits)
            p.write_text(re.sub(r"\n{3,}", "\n\n", rw.apply()))
    print(f"removed {n} fields")


if __name__ == "__main__":
    main()
