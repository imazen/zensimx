import fcntl,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_snapshot_retention as retention


class SnapshotRetentionTests(unittest.TestCase):
    def test_completed_reservations_and_pending_claims_stop_admission_before_native_start(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            root=Path(temporary);study=root/'study';(study/'cpu-admission').mkdir(parents=True)
            (study/'cpu-admission/CLAIMS.json').write_text(json.dumps(dict(old={},new={})))
            store=root/'shared-training-snapshots';store.mkdir()
            for i in range(62):(store/f'{i:064x}').mkdir()
            observed=dict(native_jobs=[dict(cell=dict(id='old'),snapshot_directory=str(store/f'{0:064x}'))])
            with patch.object(retention,'observe',return_value=observed):
                self.assertTrue(retention.capacity(study,'new')['can_start'])
                (store/f'{62:064x}').mkdir();self.assertFalse(retention.capacity(study,'new')['can_start'])
                with self.assertRaisesRegex(ValueError,'CPU claim'):retention.capacity(study,'unclaimed')
            self.assertFalse((study/'runs').exists())

    def fixture(self,base):
        root=base/'shared-training-snapshots';directory=root/('a'*64);directory.mkdir(parents=True)
        (root/'.lock').write_bytes(b'');(directory/'.lock').write_bytes(b'')
        for name in retention.STATES:(directory/name).write_bytes(name.encode())
        run=base/'run';run.mkdir();identity=dict(cell=dict(id='cell'),stage='running',epochs=120,build_sha256='build')
        (run/'RUN.json').write_text(json.dumps(identity))
        (run/'RESULT.json').write_text(json.dumps(dict(identity,stage='trained',exit_code=0)))
        observation=dict(run=str(run),snapshot_directory=str(directory))
        binding=run/'SNAPSHOT_BINDING.json'
        binding.write_text(json.dumps(dict(schema='native-snapshot-run-binding-v1',run=str(run),
            run_sha256=retention.sha(run/'RUN.json'),snapshot_directory=str(directory),native_observation=observation,
            host=__import__('socket').gethostname())))
        return directory,binding

    def test_live_lock_and_unfinished_write_refuse_without_staging(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            base=Path(temporary);directory,binding=self.fixture(base);output=base/'archive-stage'
            with (directory/'.lock').open('r+b') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):retention.stage(binding,output)
                self.assertFalse(output.exists())
            (directory/'writing.snap').write_bytes(b'unfinished')
            with self.assertRaisesRegex(ValueError,'unfinished'):retention.stage(binding,output)
            self.assertFalse(output.exists())

    def test_failed_runs_are_never_staged(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            base=Path(temporary);directory,binding=self.fixture(base)
            path=binding.parent/'RESULT.json';result=json.loads(path.read_text());result['stage']='interrupted';path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'completed native'):retention.stage(binding,base/'archive-stage')
            self.assertTrue(directory.exists())

    def test_exact_readback_frees_whole_reservation_preserving_archive_and_binding(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            base=Path(temporary);directory,binding=self.fixture(base);stage=base/'stage'
            manifest=retention.stage(binding,stage);mirror=base/'durable';mirror.mkdir()
            hashes={}
            for path in stage.iterdir():
                (mirror/path.name).write_bytes(path.read_bytes());hashes[path.name]=retention.sha(path)
            ledger=''.join(pin+'  '+name+'\n' for name,pin in hashes.items())
            (stage/'SHA256SUMS').write_text(ledger);(mirror/'SHA256SUMS').write_text(ledger)
            (stage/'MIRROR_VERIFIED.json').write_text(json.dumps(dict(destination=str(mirror),all_artifact_sha256_verified=True)))
            receipt=base/'retired.json'
            remote=mirror/'state-0.snap';remote.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'cold readback'):retention.retire(binding,stage,receipt)
            self.assertTrue(directory.exists());self.assertFalse(receipt.exists())
            remote.write_bytes((stage/remote.name).read_bytes())
            with patch('shaping_artifact_retention.no_users') as users:
                result=retention.retire(binding,stage,receipt);users.assert_called_once()
            self.assertEqual(result['stage'],'completed_store_retired');self.assertFalse(directory.exists())
            self.assertTrue((directory.parent/'.lock').exists());self.assertTrue(binding.exists())
            self.assertEqual(retention.sha(remote),manifest['files'][remote.name])

    def test_nested_snapshot_archive_requires_the_exact_prefixed_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            base=Path(temporary);directory,binding=self.fixture(base);archive=base/'stage';archive.mkdir()
            stage=archive/'snapshots';retention.stage(binding,stage)
            mirror=base/'durable';(mirror/'snapshots').mkdir(parents=True);hashes={}
            for path in stage.iterdir():
                relative=path.relative_to(archive);(mirror/relative).write_bytes(path.read_bytes())
                hashes[relative.as_posix()]=retention.sha(path)
            ledger=''.join(pin+'  '+name+'\n' for name,pin in hashes.items())
            (archive/'SHA256SUMS').write_text(ledger);(mirror/'SHA256SUMS').write_text(ledger)
            (archive/'MIRROR_VERIFIED.json').write_text(json.dumps(dict(destination=str(mirror),all_artifact_sha256_verified=True)))
            receipt=base/'retired.json';remote=mirror/'snapshots/state-1.snap';remote.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'cold readback'):retention.retire(binding,stage,receipt,archive)
            self.assertTrue(directory.exists());self.assertFalse(receipt.exists())
            remote.write_bytes((stage/remote.name).read_bytes())
            with patch('shaping_artifact_retention.no_users'):
                result=retention.retire(binding,stage,receipt,archive)
            self.assertEqual(result['snapshot_archive_prefix'],'snapshots');self.assertFalse(directory.exists())


if __name__=='__main__':unittest.main()
