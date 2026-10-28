#!/usr/bin/env python3
"""Tests for cfg_prune.py: run `python3 scripts/split/test_cfg_prune.py`."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_prune import prune  # noqa: E402

FIXED = {"feature": {"on": True, "off": False}, "cfg": {"instr": False}}


def p(src):
    return prune(src, FIXED)


class CfgPruneTest(unittest.TestCase):
    def test_false_fn_with_docs_removed(self):
        src = "fn a() {}\n\n/// doc\n/// more\n#[cfg(feature = \"off\")]\npub fn b() { let s = \"}\"; }\n\nfn c() {}\n"
        self.assertEqual(p(src), "fn a() {}\n\nfn c() {}\n")

    def test_true_attr_dropped(self):
        src = "/// doc\n#[cfg(feature = \"on\")]\npub fn b() {}\n"
        self.assertEqual(p(src), "/// doc\npub fn b() {}\n")

    def test_unknown_kept_and_simplified(self):
        src = "#[cfg(all(feature = \"on\", feature = \"threads\"))]\nfn b() {}\n"
        self.assertEqual(p(src), "#[cfg(feature = \"threads\")]\nfn b() {}\n")
        src = "#[cfg(any(feature = \"off\", test))]\nfn b() {}\n"
        self.assertEqual(p(src), "#[cfg(test)]\nfn b() {}\n")
        src = "#[cfg(not(feature = \"off\"))]\nfn b() {}\n"
        self.assertEqual(p(src), "fn b() {}\n")
        src = "#[cfg(all(feature = \"off\", test))]\nfn b() {}\nfn c() {}\n"
        self.assertEqual(p(src), "fn c() {}\n")

    def test_struct_field_and_last_field(self):
        src = "struct S {\n    a: u8,\n    #[cfg(feature = \"off\")]\n    b: u8,\n    /// d\n    #[cfg(feature = \"off\")]\n    c: Vec<u8>\n}\n"
        self.assertEqual(p(src), "struct S {\n    a: u8,\n}\n")

    def test_enum_variant(self):
        src = "enum E {\n    A,\n    /// doc\n    #[cfg(feature = \"off\")]\n    B { x: u8 },\n    C,\n}\n"
        self.assertEqual(p(src), "enum E {\n    A,\n    C,\n}\n")

    def test_match_arms(self):
        src = ("fn f(x: E) -> u8 {\n    match x {\n        E::A => 1,\n        #[cfg(feature = \"off\")]\n"
               "        E::B => { let y = 2; y }\n        #[cfg(feature = \"off\")]\n        E::C => 3,\n        _ => 0,\n    }\n}\n")
        self.assertEqual(p(src), "fn f(x: E) -> u8 {\n    match x {\n        E::A => 1,\n        _ => 0,\n    }\n}\n")

    def test_statements(self):
        src = ("fn f() {\n    let a = 1;\n    #[cfg(feature = \"off\")]\n    let b = 2;\n    #[cfg(feature = \"off\")]\n"
               "    {\n        g();\n    }\n    #[cfg(feature = \"off\")]\n    if a > 0 { g(); } else { h(); }\n"
               "    #[cfg(feature = \"off\")]\n    foo.bar().baz();\n    a\n}\n")
        self.assertEqual(p(src), "fn f() {\n    let a = 1;\n    a\n}\n")

    def test_cfg_attr(self):
        src = "#[cfg_attr(feature = \"on\", inline)]\nfn a() {}\n#[cfg_attr(feature = \"off\", inline)]\nfn b() {}\n#[cfg_attr(test, allow(dead_code))]\nfn c() {}\n"
        self.assertEqual(p(src), "#[inline]\nfn a() {}\nfn b() {}\n#[cfg_attr(test, allow(dead_code))]\nfn c() {}\n")
        src = "#[cfg_attr(not(feature = \"off\"), allow(dead_code), inline)]\nfn a() {}\n"
        self.assertEqual(p(src), "#[allow(dead_code)]\n#[inline]\nfn a() {}\n")

    def test_cfg_macro(self):
        src = "fn f() -> bool { cfg!(feature = \"on\") && !cfg!(feature = \"off\") || cfg!(test) }\n"
        self.assertEqual(p(src), "fn f() -> bool { true && !false || cfg!(test) }\n")

    def test_dead_file(self):
        self.assertIsNone(p("//! doc\n#![cfg(feature = \"off\")]\nfn a() {}\n"))
        self.assertEqual(p("#![cfg(feature = \"on\")]\nfn a() {}\n"), "fn a() {}\n")

    def test_strings_chars_lifetimes_comments(self):
        src = ("fn a<'a>(x: &'a str) -> char { let _ = \"{(\"; let _ = r#\"}\"#; /* { */ '{' }\n"
               "#[cfg(feature = \"off\")]\nfn b() { let c = '}'; // }\n}\nfn c() {}\n")
        self.assertEqual(p(src), "fn a<'a>(x: &'a str) -> char { let _ = \"{(\"; let _ = r#\"}\"#; /* { */ '{' }\nfn c() {}\n")

    def test_impl_mod_use_const(self):
        src = ("#[cfg(feature = \"off\")]\nimpl X for Y {\n    fn a(&self) {}\n}\n#[cfg(feature = \"off\")]\nmod m;\n"
               "#[cfg(feature = \"off\")]\nuse a::{b, c};\n#[cfg(feature = \"off\")]\nconst X: usize = { 1 + 2 };\n"
               "#[cfg(feature = \"off\")]\npub(crate) mod n {\n    fn a() {}\n}\nfn keep() {}\n")
        self.assertEqual(p(src), "fn keep() {}\n")

    def test_stacked_attributes_and_leading(self):
        src = "fn z() {}\n\n#[inline]\n#[cfg(feature = \"off\")]\n#[allow(dead_code)]\nfn x() {}\n\nfn y() {}\n"
        self.assertEqual(p(src), "fn z() {}\n\nfn y() {}\n")

    def test_struct_literal_and_args(self):
        src = "fn f(a: u8, #[cfg(feature = \"off\")] b: u8) -> S {\n    S { a: 1, #[cfg(feature = \"off\")] b: 2, c: 3 }\n}\n"
        # inline deletions leave spacing for rustfmt
        self.assertEqual(p(src), "fn f(a: u8, ) -> S {\n    S { a: 1,  c: 3 }\n}\n")

    def test_bare_cfg_name(self):
        src = "#[cfg(instr)]\nfn a() {}\n#[cfg(not(instr))]\nfn b() {}\n"
        self.assertEqual(p(src), "fn b() {}\n")

    def test_block_followed_by_method_chain(self):
        src = "fn f() {\n    #[cfg(feature = \"off\")]\n    S { a: 1 }.go();\n    keep();\n}\n"
        self.assertEqual(p(src), "fn f() {\n    keep();\n}\n")

    def test_brace_pattern_arms_and_guards(self):
        src = ("fn f(x: &E) -> u8 {\n    match x {\n        E::A => 1,\n        #[cfg(feature = \"off\")]\n"
               "        E::N { n, m } => {\n            *n + m\n        }\n        #[cfg(feature = \"off\")]\n"
               "        E::C { name, .. } => name,\n        #[cfg(feature = \"off\")]\n        E::D { a } if a > 0 => a,\n"
               "        _ => 0,\n    }\n}\n")
        self.assertEqual(p(src), "fn f(x: &E) -> u8 {\n    match x {\n        E::A => 1,\n        _ => 0,\n    }\n}\n")

    def test_generic_commas(self):
        src = ("#[cfg(feature = \"off\")]\npub fn with(self, h: H) -> Result<Self, crate::x::Err> {\n    Ok(self)\n}\n"
               "struct S {\n    a: u8,\n    #[cfg(feature = \"off\")]\n    m: HashMap<u8, Vec<u16>>,\n    b: u8,\n}\n"
               "fn g() {\n    #[cfg(feature = \"off\")]\n    let m: HashMap<u8, u8> = HashMap::new();\n    if a < b { c(); }\n}\n")
        self.assertEqual(p(src), "struct S {\n    a: u8,\n    b: u8,\n}\nfn g() {\n    if a < b { c(); }\n}\n")

    def test_block_then_if_statement(self):
        src = "fn f() {\n    #[cfg(feature = \"off\")]\n    {\n        setup();\n    }\n    if x { y(); }\n}\n"
        self.assertEqual(p(src), "fn f() {\n    if x { y(); }\n}\n")


if __name__ == "__main__":
    unittest.main()
