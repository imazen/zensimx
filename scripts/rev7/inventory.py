#!/usr/bin/env python3
"""Reproducible lexical inventory for the rev7 dependency audit (not a Rust call-graph resolver).

Comments and strings are masked before finding revision sites; function/module ranges are
brace matched. Output retains each occurrence, including tests, with source line numbers.
Review reachability in docs/REV7_DEPENDENCY_MAP_2026-10-25.md; a textual caller is not proof
that its branch executes. No dependencies or network access.
"""
import argparse
import json
import re
from pathlib import Path


def mask(text, strings=True):
    pattern = re.compile(r'//[^\n]*|/\*|(?:b?r)(\#*)"|(?:b)?"|\'(?:\\.|[^\'\n])\'')
    out = list(text)
    pos = 0
    while m := pattern.search(text, pos):
        start, end = m.span()
        token = m.group()
        comment = token.startswith(("//", "/*"))
        if token == "/*":
            depth = 1
            while depth:
                n = re.search(r"/\*|\*/", text[end:])
                if n is None:
                    raise ValueError("unterminated block comment")
                end += n.end()
                depth += 1 if n.group() == "/*" else -1
        elif not comment and token.endswith('"'):
            if m.group(1) is not None:
                end = text.index('"' + m.group(1), end) + 1 + len(m.group(1))
            else:
                while end < len(text):
                    if text[end] == "\\":
                        end += 2
                    elif text[end] == '"':
                        end += 1
                        break
                    else:
                        end += 1
        if comment or strings:
            out[start:end] = ["\n" if c == "\n" else " " for c in text[start:end]]
        pos = end
    return "".join(out)


def spans(code, pattern):
    result = []
    for m in re.finditer(pattern, code):
        cursor, nesting = m.end(), 0
        while cursor < len(code):
            c = code[cursor]
            if c in "([":
                nesting += 1
            elif c in ")]":
                nesting -= 1
            elif not nesting and c in "{;":
                break
            cursor += 1
        opening = cursor
        if opening == len(code) or code[opening] != "{":
            continue
        depth, end = 1, opening + 1
        while depth and end < len(code):
            depth += (code[end] == "{") - (code[end] == "}")
            end += 1
        result.append((m.start(), end, m.group(1)))
    return result


def enclosing(items, offset):
    return [x for x in items if x[0] <= offset < x[1]]


def collect(root):
    defs = (root / "zensim/src/feature_defs.rs").read_text()
    impl = defs[defs.index("impl FormulaRevision"):defs.index("/// Every slot id ONE")]
    predicates = re.findall(r"fn\s+(\w+)\s*\(", mask(impl))
    predicates += ["active_revision", "effective_revision", "for_revision", "refuse_rev4_mix",
                   "refuse_rev4_strips", "process_ring_high_pass", "check_route"]
    site_re = re.compile(r"(?:FormulaRevision|Self)::Rev[1-7]\b|\b(?:" + "|".join(predicates) + r")\s*\(")
    revisions, envs, functions = [], [], []
    for base in ["zensim/src", "zensim-validate/src"]:
        for path in sorted((root / base).rglob("*.rs")):
            text = path.read_text()
            code = mask(text)
            fn = spans(code, r"\bfn\s+(\w+)")
            modules = spans(code, r"\bmod\s+(\w+)")
            def location(offset):
                owners = enclosing(fn, offset)
                owner = min(owners, key=lambda x: x[1] - x[0]) if owners else None
                mods = [x[2] for x in enclosing(modules, offset)]
                begin = owner[0] if owner else offset
                attrs = text[max(0, begin - 400):begin]
                # A lexical label; cfgs on external module declarations need manual review.
                test_module = any(re.search(r"#\[cfg\([^\]]*\btest\b", text[max(0, x[0] - 150):x[0]])
                                  for x in enclosing(modules, offset))
                test = (test_module or any("test" in n for n in mods)
                        or path.stem.endswith("_tests") or "instrument" in path.stem
                        or bool(re.search(r"#\[test\]", attrs.split("}")[-1])))
                return {"file": str(path.relative_to(root)), "line": text.count("\n", 0, offset) + 1,
                        "function": owner[2] if owner else "<item>",
                        "context": "test" if test else "runtime/cfg-review"}
            if base == "zensim/src":
                for m in site_re.finditer(code):
                    revisions.append({**location(m.start()), "site": m.group().strip()})
                for start, end, name in fn:
                    functions.append({**location(start), "function": name,
                                      "end": text.count("\n", 0, end) + 1,
                                      "lines": text.count("\n", start, end) + 1})
            literals = mask(text, strings=False)
            for m in re.finditer(r'(?:std::env::)?(?:var|var_os)\s*\(\s*"(ZENSIM_[A-Z0-9_]+)"', literals):
                envs.append({**location(m.start()), "variable": m.group(1)})
            # One test helper reads a constant rather than a string at its call site.
            for m in re.finditer(r'const\s+(\w+)\s*:\s*&str\s*=\s*"(ZENSIM_[A-Z0-9_]+)"', literals):
                for use in re.finditer(r'\b(?:var|var_os)\s*\(\s*' + m.group(1) + r'\s*\)', code):
                    envs.append({**location(use.start()), "variable": m.group(2)})
    return {"predicates": predicates, "revision_sites": revisions, "environment_reads": envs,
            "functions": functions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['revision_sites'])} revision/predicate occurrences; "
          f"{len(result['environment_reads'])} ZENSIM environment reads; "
          f"{len(result['functions'])} function bodies")


if __name__ == "__main__":
    main()
