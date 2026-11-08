import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_rescue_wave_prepare import protocol_snapshot


class RescueProtocolTests(unittest.TestCase):
    def test_exported_registration_uses_only_identical_frozen_public_policy(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';study.mkdir();source=root/'export';(source/'docs').mkdir(parents=True)
            policy=source/'docs/SHAPING_REDUCED_SCOPE.md';policy.write_text('frozen public study policy')
            (study/'REDUCED_SCOPE.json').write_bytes(encoded(dict(evidence_sha256={'/original/checkout/docs/'+policy.name:sha(policy)})))
            target=protocol_snapshot(study,source);self.assertEqual(target.read_bytes(),policy.read_bytes())
            before=target.stat().st_mtime_ns;self.assertEqual(protocol_snapshot(study,source),target)
            self.assertEqual(target.stat().st_mtime_ns,before)
            target.write_text('changed')
            with self.assertRaisesRegex(ValueError,'preserve changed'):protocol_snapshot(study,source)
            self.assertEqual(target.read_text(),'changed')


if __name__=='__main__':unittest.main()
