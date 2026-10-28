#!/usr/bin/env python3
"""Unwrap `= { EXPR };` and `= { EXPR }` single-expression blocks that bool_fold.py leaves
behind (`let a = if true { x } else { y };` -> `let a = { x };`), and drop comment lines
that name removed items when given with --stale-comment REGEX. 2026-10-28 slim cut.

    unwrap_blocks.py FILE...
"""
import re
import sys
from pathlib import Path

PAT = re.compile(r"= \{\n([ \t]*)([^;{}\n]+)\n[ \t]*\}(;|,)")
PAT1 = re.compile(r"= \{ ([^;{}\n]+) \}(;|,)")


def unwrap(src):
    src = PAT.sub(lambda m: f"= {m.group(2).strip()}{m.group(3)}", src)
    src = PAT1.sub(lambda m: f"= {m.group(1).strip()}{m.group(2)}", src)
    return src


if __name__ == "__main__":
    for f in sys.argv[1:]:
        p = Path(f)
        s = p.read_text()
        t = unwrap(s)
        if t != s:
            p.write_text(t)
            print(f"{f}: {len(PAT.findall(s)) + len(PAT1.findall(s))} blocks")
