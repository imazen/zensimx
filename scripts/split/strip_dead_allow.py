#!/usr/bin/env python3
"""Remove `dead_code` allowances so rustc reports every unused item (2026-10-28 slim cut).

* `#[allow(dead_code)]`, `#![allow(dead_code)]` and `#[cfg_attr(COND, allow(dead_code))]`
  lines are deleted, together with a trailing `//` comment on the same line;
* `dead_code` is dropped from a longer `allow(...)` list.

Run dead_prune.py over all build configurations afterwards: an item one configuration still
uses survives, because deletion requires every configuration to report it unused.

    strip_dead_allow.py FILE...
"""
import re
import sys
from pathlib import Path

ALONE = re.compile(r"^\s*#!?\[(?:allow\(dead_code\)|cfg_attr\(.*,\s*allow\(dead_code\)\))\]\s*(?://.*)?$")
IN_LIST = re.compile(r"(#!?\[(?:cfg_attr\(.*?,\s*)?allow\()([^)]*)(\))")


def strip(src):
    out, n = [], 0
    for line in src.split("\n"):
        if ALONE.match(line):
            n += 1
            continue
        if "dead_code" in line and "allow(" in line and not line.lstrip().startswith("//"):
            def fix(m):
                items = [x.strip() for x in m.group(2).split(",") if x.strip()]
                if "dead_code" not in items:
                    return m.group(0)
                items.remove("dead_code")
                return m.group(1) + ", ".join(items) + m.group(3)
            new = IN_LIST.sub(fix, line)
            if new != line:
                n += 1
                line = new
        out.append(line)
    return "\n".join(out), n


if __name__ == "__main__":
    for f in sys.argv[1:]:
        p = Path(f)
        s, n = strip(p.read_text())
        if n:
            p.write_text(s)
            print(f"{f}: {n}")
