"""Use real snapshot staging/cold-readback gates with small fixture states."""
import json,shutil,socket,sys,tempfile,unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_snapshot_batch as batch
import shaping_snapshot_retention as owner


class SnapshotBatchTests(unittest.TestCase):
    def fixture(self,root):
        study=root/'study';run=study/'runs/cell/attempt-001';run.mkdir(parents=True)
        stores=root/'shared-training-snapshots';store=stores/('a'*64);store.mkdir(parents=True)
        (stores/'.lock').touch();(store/'.lock').touch()
        for name in owner.STATES:(store/name).write_bytes(name.encode())
        identity=dict(cell={'id':'cell'},epochs=120,build_sha256='frozen')
        (run/'RUN.json').write_text(json.dumps(identity));(run/'RESULT.json').write_text(json.dumps(dict(identity,stage='trained',exit_code=0)))
        (run/'model.bin').write_bytes(b'keep model')
        binding=run/'SNAPSHOT_BINDING.json';binding.write_text(json.dumps(dict(schema='native-snapshot-run-binding-v1',
            run=str(run),run_sha256=owner.sha(run/'RUN.json'),snapshot_directory=str(store),host=socket.gethostname(),
            native_observation=dict(run=str(run),snapshot_directory=str(store)))))
        return study,run,store,binding

    def archive(self,root,stage):
        durable=root/'durable';shutil.copytree(stage,durable)
        lines=''.join(owner.sha(p)+'  '+p.relative_to(stage).as_posix()+'\n' for p in sorted(stage.rglob('*')) if p.is_file())
        for base in (stage,durable):(base/'SHA256SUMS').write_text(lines)
        (stage/'MIRROR_VERIFIED.json').write_text(json.dumps(dict(destination=str(durable),all_artifact_sha256_verified=True)))
        return durable

    def test_only_archived_completed_stores_retire_and_models_stay(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run,store,binding=self.fixture(root);stage=root/'stage'
            storage=SimpleNamespace(reservation=lambda *args,**kwargs:nullcontext())
            result=batch.stage(study,stage,owner,storage);self.assertEqual(result['stores'],1)
            durable=self.archive(root,stage)
            with patch('shaping_checkpoint_retention.no_users'):
                result=batch.retire(stage,root/'RETIRED.json',owner)
            self.assertEqual(result['retired_stores'],1);self.assertFalse(store.exists())
            self.assertEqual((run/'model.bin').read_bytes(),b'keep model')
            for name in owner.STATES:self.assertEqual((durable/('a'*64)/name).read_bytes(),name.encode())
            self.assertEqual(json.loads((binding.with_name('SNAPSHOT_RETIREMENT.json')).read_text())['stage'],'completed_store_retired')
            with self.assertRaisesRegex(ValueError,'fresh retirement receipt'):batch.retire(stage,root/'RETIRED.json',owner)

    def test_corrupt_durable_state_refuses_before_any_local_removal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run,store,binding=self.fixture(root);stage=root/'stage'
            batch.stage(study,stage,owner,SimpleNamespace(reservation=lambda *args,**kwargs:nullcontext()))
            durable=self.archive(root,stage);(durable/('a'*64)/'state-0.snap').write_bytes(b'corrupt')
            with patch('shaping_checkpoint_retention.no_users'):
                with self.assertRaisesRegex(ValueError,'cold readback differs'):batch.retire(stage,root/'RETIRED.json',owner)
            for name in owner.STATES:self.assertEqual((store/name).read_bytes(),name.encode())
            self.assertFalse(binding.with_name('SNAPSHOT_RETIREMENT.json').exists())

    def test_unfinished_or_failed_training_never_stages(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run,store,binding=self.fixture(root)
            result=json.loads((run/'RESULT.json').read_text());result.update(stage='failed',exit_code=137)
            (run/'RESULT.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'no completed native snapshot'):
                batch.stage(study,root/'stage',owner,SimpleNamespace(reservation=lambda *args,**kwargs:nullcontext()))
            for name in owner.STATES:self.assertEqual((store/name).read_bytes(),name.encode())

    def test_staged_duplicates_trim_only_after_complete_retirement_and_cold_readback(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run,store,binding=self.fixture(root);stage=root/'stage'
            batch.stage(study,stage,owner,SimpleNamespace(reservation=lambda *args,**kwargs:nullcontext()))
            durable=self.archive(root,stage);retirement=root/'RETIRED.json'
            with patch('shaping_checkpoint_retention.no_users'):
                batch.retire(stage,retirement,owner)
            with patch('shaping_checkpoint_retention.no_users') as users:
                result=batch.trim(stage,retirement,root/'TRIMMED.json',owner)
                self.assertEqual(len(users.call_args.args[0]),2)
            self.assertEqual(result['files'],2)
            for name in owner.STATES:
                self.assertFalse((stage/('a'*64)/name).exists())
                self.assertEqual((durable/('a'*64)/name).read_bytes(),name.encode())
            self.assertTrue((stage/('a'*64)/'SNAPSHOTS.json').exists())
            self.assertEqual((run/'model.bin').read_bytes(),b'keep model')

    def test_corrupt_archive_prevents_staged_duplicate_removal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run,store,binding=self.fixture(root);stage=root/'stage'
            batch.stage(study,stage,owner,SimpleNamespace(reservation=lambda *args,**kwargs:nullcontext()))
            durable=self.archive(root,stage);retirement=root/'RETIRED.json'
            with patch('shaping_checkpoint_retention.no_users'):batch.retire(stage,retirement,owner)
            (durable/('a'*64)/'state-1.snap').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'cold readback differs'):
                batch.trim(stage,retirement,root/'TRIMMED.json',owner)
            for name in owner.STATES:self.assertEqual((stage/('a'*64)/name).read_bytes(),name.encode())
            self.assertFalse((root/'TRIMMED.json').exists())

    def test_combined_retirement_checks_and_removes_both_worker_copies(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,run,store,binding=self.fixture(root);stage=root/'stage'
            batch.stage(study,stage,owner,SimpleNamespace(reservation=lambda *args,**kwargs:nullcontext()))
            durable=self.archive(root,stage)
            with patch('shaping_checkpoint_retention.no_users') as users:
                batch.retire(stage,root/'RETIRED.json',owner,trim_staged=True)
                self.assertEqual(len(users.call_args.args[0]),4)
            self.assertFalse(store.exists())
            for name in owner.STATES:
                self.assertFalse((stage/('a'*64)/name).exists())
                self.assertEqual((durable/('a'*64)/name).read_bytes(),name.encode())


if __name__=='__main__':unittest.main()
