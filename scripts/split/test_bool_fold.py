#!/usr/bin/env python3
"""Tests for bool_fold.py: run `python3 scripts/split/test_bool_fold.py`."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bool_fold import fold  # noqa: E402


def f(src):
    # whitespace is left to rustfmt; compare token text
    return re.sub(r"[ \t]+\n", "\n", fold(src)[0])


class BoolFoldTest(unittest.TestCase):
    def test_not(self):
        self.assertEqual(f("fn a() { if !false { b(); } }\n"), "fn a() { { b(); } }\n")

    def test_and_or_precedence(self):
        self.assertEqual(f("g(a && false || c);\n"), "g(c);\n")
        self.assertEqual(f("g(a && b || true);\n"), "g(true);\n")
        self.assertEqual(f("g(false || a && b);\n"), "g(a && b);\n")
        self.assertEqual(f("g(a == b && true);\n"), "g(a == b);\n")
        self.assertEqual(f("g(f(a, b) && false);\n"), "g(false);\n")
        self.assertEqual(f("S { m: layout && false, n: 1 }\n"), "S { m: false, n: 1 }\n")

    def test_if(self):
        self.assertEqual(f("fn a() {\n    if false {\n        b();\n    }\n    c();\n}\n"), "fn a() {\n    c();\n}\n")
        self.assertEqual(f("let x = if false { 1 } else { 2 };\n"), "let x = { 2 };\n")
        self.assertEqual(f("if false { a(); } else if c { b(); } else { d(); }\n"), "if c { b(); } else { d(); }\n")
        self.assertEqual(f("if c { a(); } else if false { b(); } else { d(); }\n"), "if c { a(); } else { d(); }\n")
        self.assertEqual(f("if c { a(); } else if false { b(); }\n"), "if c { a(); }\n")
        self.assertEqual(f("if true { a(); } else { b(); }\n"), "{ a(); }\n")
        self.assertEqual(f("if c { a(); } else if true { b(); } else { d(); }\n"), "if c { a(); } else { b(); }\n")

    def test_let(self):
        src = "fn a() {\n    let on = false;\n    f(on, x);\n    let s = S { on, y };\n    if on { z(); }\n}\nfn b(on: bool) { g(on); }\n"
        self.assertEqual(f(src), "fn a() {\n    f(false, x);\n    let s = S { on: false, y };\n}\nfn b(on: bool) { g(on); }\n")

    def test_let_shadowed_in_nested_block(self):
        src = "fn a() {\n    let on = false;\n    {\n        let on = x;\n        g(on);\n    }\n    h(on);\n}\n"
        self.assertEqual(f(src), "fn a() {\n    {\n        let on = x;\n        g(on);\n    }\n    h(false);\n}\n")

    def test_let_with_attribute_is_kept(self):
        src = "#[cfg(x)]\nlet on = false;\nf(on);\n"
        self.assertEqual(f(src), src)

    def test_macro_bang_untouched(self):
        src = "assert!(x);\n"
        self.assertEqual(f(src), src)

    def test_call_parens_kept(self):
        self.assertEqual(f("let a = g(false);\n"), "let a = g(false);\n")
        self.assertEqual(f("g((false));\n"), "g(false);\n")


if __name__ == "__main__":
    unittest.main()
