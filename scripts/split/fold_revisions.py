#!/usr/bin/env python3
"""Remove formula revisions the slim crate cannot reach (2026-10-28 cut).

The slim crate computes at revision 1 (the v1 profiles and caller-built configs) and
revision 7 (the curated bakes); revision 5 survives as the accumulator-arithmetic value the
revision-7 walk passes internally. Revisions DROP (default 2, 3, 4, 6) are removed:

* or-pattern alternatives naming a dropped revision are deleted;
* whole match arms `Self::RevN => ...` / `FormulaRevision::RevN => ...` are deleted;
* comparisons against a dropped revision are re-pointed at the next kept revision above it
  (`>= Rev3` becomes `>= Rev5`, `< Rev6` becomes `< Rev7`), which is exact on the kept
  values {1, 5, 7};
* `matches!(x, Self::Rev6)` alone becomes `false`.

References this does not handle (tests constructing a dropped revision) are left for the
compiler to report.

    fold_revisions.py [--drop 2,3,4,6] FILE...
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import Rewriter, merge_overlaps  # noqa: E402

PATH = r"(?:crate::feature_defs::)?(?:FormulaRevision|Self)::Rev"


def next_kept(n, drop):
    k = n + 1
    while k in drop:
        k += 1
    return k


def delete_arms(src, drop):
    rw = Rewriter(src, {"feature": {}, "cfg": {}})
    toks = rw.toks
    sig = [i for i in range(len(toks)) if toks[i][0] not in ("ws", "comment", "doc")]
    T = lambda k: rw.text(sig[k]) if 0 <= k < len(sig) else ""
    for k in range(len(sig)):
        t = T(k)
        m = re.fullmatch(r"Rev(\d+)", t)
        if not m or int(m.group(1)) not in drop:
            continue
        # path start
        s = k
        while T(s - 1) == "::" and T(s - 2) not in ("", ):
            s -= 2
        if T(k + 1) != "=>":
            continue
        prev = T(s - 1)
        if prev not in ("{", ",", "}"):
            continue  # part of an or-pattern; handled by the regex pass
        # arm end: walk to the `,` at depth 0, or to the closing `}` of the match
        e = k + 2
        while e < len(sig):
            te = T(e)
            if te in ("(", "[", "{"):
                e = sig.index(rw.match[sig[e]]) + 1
                if T(e) == ",":
                    break
                if T(e) == "}":
                    e -= 1
                    break
                continue
            if te == ",":
                break
            if te == "}":
                e -= 1
                break
            e += 1
        rw.edits.append((rw.line_start(sig[s]), rw.after_line(sig[e]), ""))
    rw.edits = merge_overlaps(rw.edits)
    return rw.apply()


def rewrite(src, drop):
    d = "".join(str(x) for x in sorted(drop))
    src = delete_arms(src, drop)
    # or-pattern alternatives: leading `X |` and trailing `| X`
    src = re.sub(rf"{PATH}[{d}]\b\s*\|(?!\|)\s*", "", src)
    src = re.sub(rf"(?<!\|)\s*\|(?!\|)\s*{PATH}[{d}]\b(?!\s*=>)", "", src)
    # comparisons
    def cmp(m):
        return f"{m.group(1)}{m.group(2)}{next_kept(int(m.group(3)), drop)}"
    src = re.sub(rf"((?:>=|<=|<|>)\s*)({PATH})([{d}])\b", cmp, src)
    src = re.sub(rf"matches!\(\s*self\s*,\s*Self::Rev[{d}]\s*\)", "false", src)
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop", default="2,3,4,6")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    drop = {int(x) for x in a.drop.split(",")}
    for f in a.files:
        p = Path(f)
        s = p.read_text()
        t = rewrite(s, drop)
        if t != s:
            p.write_text(t)
            print(f"{f}: rewritten")


if __name__ == "__main__":
    main()
