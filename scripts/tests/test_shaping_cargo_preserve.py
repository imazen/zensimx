import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from shaping_cargo_preserve import preserve
from shaping_temp_storage import targets


class CargoPreserveTests(unittest.TestCase):
    def fixture(self, root):
        home = root / 'home'
        target = home / 'tmp/zensim-fixture/target'
        (target / 'debug/deps').mkdir(parents=True)
        (target / '.rustc_info.json').write_text('{}')
        inventory = root / 'INVENTORY.json'
        inventory.write_text(json.dumps(dict(host=socket.gethostname(), cargo_targets=[dict(
            path=str(target), active_processes=[], active_use_check_complete=True)])))
        return home, target, inventory

    def test_programs_models_and_unknown_files_survive_compiler_object_filter(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory);home, target, inventory = self.fixture(root)
            elf = b'\x7fELF\x02\x01' + b'\0' * 10
            for name, data in [('golden-program', elf + b'\x03\0reference'),
                               ('checkpoint.bin', b'unique model'),
                               ('run.json', b'{"seed":1}'),
                               ('libdependency.rlib', b'!<arch>\ncompiler'),
                               ('libdependency.rmeta', b'rustmetadata'),
                               ('kernel.o', elf + b'\x01\0object'),
                               ('unknown.rmeta', b'not compiler metadata')]:
                (target / 'debug/deps' / name).write_bytes(data)
            # A top-level reference library is preserved as well.
            (target / 'debug/libreference.rlib').write_bytes(b'!<arch>\nreference')
            before = {p.relative_to(target): p.read_bytes() for p in target.rglob('*') if p.is_file()}
            out = root / 'references'
            with patch('shaping_cargo_preserve.process_uses', return_value=({str(target): []}, [])):
                result = preserve(inventory, out, home)
            saved = {row['path'] for row in result['targets'][0]['preserved']}
            self.assertIn('debug/deps/golden-program', saved)
            self.assertIn('debug/deps/checkpoint.bin', saved)
            self.assertIn('debug/deps/unknown.rmeta', saved)
            self.assertIn('debug/libreference.rlib', saved)
            self.assertEqual(result['targets'][0]['recognized_compiler_objects'], 3)
            self.assertFalse(result['data_deleted'])
            self.assertEqual(before, {p.relative_to(target): p.read_bytes() for p in target.rglob('*') if p.is_file()})
            for row in result['targets'][0]['preserved']:
                self.assertEqual((out / row['archive_path']).read_bytes(), before[Path(row['path'])])
            self.assertEqual(list(targets(out)), [], 'reference archive must never resemble disposable Cargo output')

    def test_live_users_or_incomplete_visibility_protect_the_whole_target(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory);home, target, inventory = self.fixture(root)
            for users, unreadable in [([dict(pid=123)], []), ([], [123])]:
                with patch('shaping_cargo_preserve.process_uses', return_value=({str(target): users}, unreadable)):
                    with self.assertRaisesRegex(ValueError, 'no active Cargo users'):
                        preserve(inventory, root / 'references', home)
            self.assertFalse((root / 'references').exists())
            self.assertTrue(target.is_dir())

    def test_symlink_and_existing_backup_refuse_without_removal(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory);home, target, inventory = self.fixture(root)
            (target / 'debug/deps/link').symlink_to(root / 'outside')
            with patch('shaping_cargo_preserve.process_uses', return_value=({str(target): []}, [])):
                with self.assertRaisesRegex(ValueError, 'symlink in Cargo output'):
                    preserve(inventory, root / 'references', home)
                with self.assertRaisesRegex(ValueError, 'existing reference archive'):
                    preserve(inventory, root / 'references', home)
            self.assertTrue((target / 'debug/deps/link').is_symlink())


if __name__ == '__main__':
    unittest.main()
