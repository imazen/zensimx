#!/usr/bin/env python3
"""Drop function parameters rustc reports unused in EVERY build configuration (2026-10-28
slim cut), with the matching argument at every call site (via drop_params.py).

Only `fn` parameters are handled; unused closure parameters and `let` bindings are listed
for manual review. A parameter whose name starts with `_` is never reported by rustc.

    unused_params.py --root <workspace> [--dry-run] LOG.json...
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter  # noqa: E402

HERE = Path(__file__).resolve().parent


def unused(log):
    out = {}
    for line in open(log):
        if not line.startswith("{"):
            continue
        m = json.loads(line)
        if m.get("reason") != "compiler-message":
            continue
        msg = m["message"]
        if (msg.get("code") or {}).get("code") != "unused_variables":
            continue
        for sp in msg["spans"]:
            if sp.get("is_primary"):
                out[(sp["file_name"], sp["byte_start"])] = sp["text"][0]["text"][sp["text"][0]["highlight_start"] - 1:sp["text"][0]["highlight_end"] - 1]
    return out


def classify(rw, off):
    """('fn', fn_name) when the token at char offset `off` is a parameter of a `fn`."""
    idx = next((i for i, t in enumerate(rw.toks) if t[1] == off), None)
    if idx is None:
        return None
    # walk back to the enclosing open bracket
    j = idx - 1
    while j >= 0:
        t = rw.text(j)
        if t in (")", "]", "}"):
            j = rw.match[j] - 1
            continue
        if t in ("(", "[", "{", "|"):
            break
        j -= 1
    if j < 0 or rw.text(j) != "(":
        return None
    b = rw.sig(j - 1, -1)
    if rw.text(b) == ">":
        # generics: walk back to the matching `<`
        depth = 0
        while b >= 0:
            t = rw.text(b)
            if t == ">":
                depth += 1
            elif t == ">>":
                depth += 2
            elif t == "<":
                depth -= 1
                if depth == 0:
                    break
            b -= 1
        b = rw.sig(b - 1, -1)
    name = rw.text(b)
    if rw.text(rw.sig(b - 1, -1)) == "fn":
        return name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    sets = [unused(l) for l in a.logs]
    common = set(sets[0])
    for s in sets[1:]:
        common &= set(s)
    todo = defaultdict(set)
    manual = []
    for f, b in sorted(common):
        p = Path(a.root) / f
        raw = p.read_bytes()
        src = raw.decode("utf-8")
        rw = Rewriter(src, {"feature": {}, "cfg": {}})
        off = len(raw[:b].decode("utf-8"))
        tok = next((k for k, t in enumerate(rw.toks) if t[1] == off), None)
        if tok is not None and rw.text(tok) == "mut":
            tok = rw.sig(tok + 1)
        name = rw.text(tok) if tok is not None else sets[0][(f, b)]
        fn = classify(rw, off)
        line = src.count("\n", 0, off) + 1
        if fn and not name.startswith("_"):
            todo[(f, fn)].add(name[4:].strip() if name.startswith("mut ") else name.strip())
        else:
            manual.append(f"{f}:{line} {name}")
    files = sorted({str(Path(a.root) / f) for f, _ in todo} | {str(Path(a.root) / f) for f, _ in common})
    all_rs = sorted(str(p) for p in (Path(a.root) / "zensim" / "src").rglob("*.rs"))
    import re as _re
    crate_src = "\n".join(Path(x).read_text() for x in all_rs)
    for (f, fn), params in sorted(todo.items()):
        defs = list(_re.finditer(r"\bfn\s+" + _re.escape(fn) + r"\b\s*(?:<[^{(]*?>)?\s*\(\s*([^,)]*)", crate_src))
        if len(defs) != 1:
            print(f"manual {f}: fn {fn} has {len(defs)} definitions ({', '.join(sorted(params))})")
            continue
        if _re.match(r"(&\s*(mut\s+)?)?(mut\s+)?self\b", defs[0].group(1).strip()):
            print(f"manual {f}: method {fn}({', '.join(sorted(params))})")
            continue
        print(f"drop {f}: fn {fn}({', '.join(sorted(params))})")
        if a.dry_run:
            continue
        cmd = [sys.executable, str(HERE / "drop_params.py"), "--fn", fn]
        for prm in sorted(params):
            cmd += ["--param", prm]
        cmd += all_rs
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("  FAILED:", r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.stdout)
    for m in manual:
        print("manual", m)


if __name__ == "__main__":
    main()
