import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import shaping_evaluation_runtime as owner


class RuntimeTests(unittest.TestCase):
    def test_runtime_registration_and_missing_copy_preserve_existing_different_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
            binaries=root/'bin';binaries.mkdir();native=binaries/'baker';native.write_bytes(b'native');native.chmod(0o755)
            (binaries/'BUILD.json').write_text(json.dumps(dict(binary_sha256={'baker':owner.sha(native)})))
            inspector=root/'inspector';inspector.write_bytes(b'inspector');inspector.chmod(0o755)
            owner.register(study,root/'source',[binaries],inspector,root/'capsule')
            manifest=root/'capsule/REGISTRATION.json'
            with patch.object(owner.socket,'gethostname',return_value='r7900x'):
                self.assertEqual(owner.verify(manifest)[1],0)
                native.unlink();self.assertEqual(owner.verify(manifest,True)[1],6)
                native.write_bytes(b'broken');native.chmod(0o755)
                with self.assertRaisesRegex(ValueError,'preserve different'):owner.verify(manifest,True)
            self.assertEqual(native.read_bytes(),b'broken')

    def test_runtime_registration_refuses_a_changed_native_before_creating_capsule(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            root=Path(tmp);binaries=root/'bin';binaries.mkdir();native=binaries/'baker';native.write_bytes(b'native')
            (binaries/'BUILD.json').write_text(json.dumps(dict(binary_sha256={'baker':'changed'})))
            with self.assertRaisesRegex(ValueError,'binary differs'):
                owner.register(root/'study',root/'source',[binaries],root/'inspector',root/'capsule')
            self.assertFalse((root/'capsule').exists())


if __name__=='__main__':unittest.main()
