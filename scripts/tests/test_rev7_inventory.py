"""Negative controls for the lexical audit: quoted/commented gates are not code."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "rev7_inventory", Path(__file__).resolve().parents[1] / "rev7/inventory.py"
)
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)


class InventoryTests(unittest.TestCase):
    def test_comment_ending_in_quote_does_not_swallow_following_code(self):
        text = '// arm "7"\nfn kept() { x.row_lanes(); }\n'
        masked = INVENTORY.mask(text)
        self.assertIn("x.row_lanes()", masked)
        self.assertNotIn('"7"', masked)
        self.assertEqual(len(text), len(masked))

    def test_nested_comments_raw_strings_and_escaped_quotes_are_not_sites(self):
        text = '/* outer /* x.row_lanes() */ inner */\n' + r'let a = r##"x.stable_forms()"##; let b = "\"Rev7";'
        self.assertNotIn("row_lanes", INVENTORY.mask(text))
        self.assertNotIn("stable_forms", INVENTORY.mask(text))
        self.assertNotIn("Rev7", INVENTORY.mask(text))

    def test_array_semicolons_do_not_hide_function_bodies(self):
        code = 'fn kept(a: &[f32; 8]) -> [f32; 8] { a.map(|x| { x + 1. }) } fn next() {}'
        spans = INVENTORY.spans(code, r"\bfn\s+(\w+)")
        self.assertEqual([s[2] for s in spans], ["kept", "next"])
        self.assertEqual(code[spans[0][1]:].strip(), "fn next() {}")

    def test_constant_environment_names_report_each_read_not_the_declaration(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            source = root / "zensim/src"
            source.mkdir(parents=True)
            (source / "feature_defs.rs").write_text(
                "impl FormulaRevision { fn row_lanes(self) -> bool { true } }\n"
                "/// Every slot id ONE\n"
            )
            (source / "helper.rs").write_text(
                'const CHILD: &str = "ZENSIM_TEST_CHILD";\n'
                'fn helper() {\n'
                '    std::env::var(CHILD);\n'
                '    std::env::var_os(CHILD);\n'
                '}\n'
            )
            reads = INVENTORY.collect(root)["environment_reads"]
            self.assertEqual([r["line"] for r in reads], [3, 4])
            self.assertEqual({r["variable"] for r in reads}, {"ZENSIM_TEST_CHILD"})


if __name__ == "__main__":
    unittest.main()
