import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from shaping_cargo_preserve import preserve, sha
from shaping_cargo_reclaim import reclaim, no_nested_mounts


class CargoReclaimTests(unittest.TestCase):
    def fixture(self, root):
        home = root / 'home';target = home / 'tmp/zensim-fixture/target'
        (target / 'debug/deps').mkdir(parents=True)
        (target / '.rustc_info.json').write_text('{}')
        (target / 'debug/deps/libexample.rmeta').write_bytes(b'rustmetadata')
        (target / 'debug/deps/checkpoint.bin').write_bytes(b'irreplaceable model')
        inventory = root / 'INVENTORY.json'
        inventory.write_text(json.dumps(dict(host=socket.gethostname(), cargo_targets=[dict(
            path=str(target), active_processes=[], active_use_check_complete=True)])))
        local = root / 'references'
        preserve(inventory, local, home)
        durable = root / 'durable';shutil.copytree(local, durable)
        receipt = root / 'RECEIPT.json'
        receipt.write_text(json.dumps(dict(all_artifact_sha256_verified=True,
            readback_method='fsync, POSIX_FADV_DONTNEED and client SHA-256',destination=str(durable))))
        shutil.copy2(receipt, durable / 'MIRROR_VERIFIED.json')
        manifest = root / 'SHA256SUMS'
        manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(durable).as_posix()}\n'
                                   for p in sorted(durable.rglob('*')) if p.is_file()))
        return home, target, local, durable, receipt, manifest

    def inactive(self):
        return patch('shaping_cargo_preserve.process_uses',
                     side_effect=lambda paths: ({str(p): [] for p in paths}, []))

    def test_reclamation_keeps_durable_reference_and_searchable_local_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory, self.inactive():
            home, target, local, durable, receipt, manifest = self.fixture(Path(directory))
            result = reclaim(local, receipt, manifest, sha(receipt), sha(manifest), home, prune=True)
            self.assertEqual(result['stage'], 'complete')
            self.assertFalse(target.exists());self.assertFalse((local / 'objects').exists())
            self.assertTrue((local / 'PRESERVATION.json').is_file())
            self.assertTrue((local / 'CARGO_RECLAIM.json').is_file())
            self.assertIn(b'irreplaceable model', [p.read_bytes() for p in (durable / 'objects').iterdir()])
            with self.assertRaisesRegex(ValueError, 'previous reclamation'):
                reclaim(local, receipt, manifest, sha(receipt), sha(manifest), home)

    def test_changed_source_bad_witness_and_live_users_protect_cargo(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory, self.inactive():
            home, target, local, durable, receipt, manifest = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, 'witness checksum'):
                reclaim(local, receipt, manifest, '0'*64, sha(manifest), home)
            with patch('shaping_cargo_preserve.process_uses',
                       side_effect=lambda paths: ({str(p): [dict(pid=123)] for p in paths}, [])):
                with self.assertRaisesRegex(ValueError, 'no active Cargo users'):
                    reclaim(local, receipt, manifest, sha(receipt), sha(manifest), home)
            (target / 'debug/deps/checkpoint.bin').write_bytes(b'new unique model')
            with self.assertRaisesRegex(ValueError, 'changed after reference preservation'):
                reclaim(local, receipt, manifest, sha(receipt), sha(manifest), home)
            self.assertTrue(target.is_dir());self.assertTrue((local / 'objects').is_dir())

    def test_durable_root_and_unarchived_objects_cannot_be_pruned(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory, self.inactive():
            home, target, local, durable, receipt, manifest = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, 'separate from durable archive'):
                reclaim(durable, receipt, manifest, sha(receipt), sha(manifest), home, references_only=True)
            (local / 'objects/new-unique-data').write_bytes(b'not archived')
            with self.assertRaisesRegex(ValueError, 'object inventory changed'):
                reclaim(local, receipt, manifest, sha(receipt), sha(manifest), home, references_only=True)
            self.assertTrue(target.is_dir())
            self.assertEqual((local / 'objects/new-unique-data').read_bytes(), b'not archived')

    def test_local_reference_copy_cleanup_never_inspects_original_host_targets(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory, self.inactive():
            home, target, local, durable, receipt, manifest = self.fixture(Path(directory))
            with patch('shaping_cargo_reclaim.metadata', side_effect=AssertionError('original target must stay untouched')):
                result = reclaim(local, receipt, manifest, sha(receipt), sha(manifest), home, references_only=True)
            self.assertEqual(result['completed_targets'], [])
            self.assertTrue(target.is_dir());self.assertFalse((local / 'objects').exists())

    def test_same_device_bind_mount_is_still_protected(self):
        mountinfo = '42 1 8:1 / /scratch/target/bound rw,relatime - ext4 /dev/example rw\n'
        with patch('shaping_cargo_reclaim.Path.read_text', return_value=mountinfo):
            with self.assertRaisesRegex(ValueError, 'mount point protects'):
                no_nested_mounts(Path('/scratch/target'))


if __name__ == '__main__':
    unittest.main()
