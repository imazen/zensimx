import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_checkpoint_restore import restore


class CheckpointRestoreTests(unittest.TestCase):
    def fixture(self,root):
        study=root/'study';directory=study/'runs/cell';run=directory/'attempt-001'
        checkpoints=run/'checkpoints';checkpoints.mkdir(parents=True);mirror=root/'archive';mirror.mkdir()
        (study/'PLAN.json').write_text('{}');(directory/'SHA256SUMS').write_text('bound fixture')
        cell=dict(id='cell',stage='inner',family='relu')
        (run/'RUN.json').write_bytes(encoded(dict(cell=cell,epochs=240,plan_sha256=sha(study/'PLAN.json'))))
        (run/'RESULT.json').write_bytes(encoded(dict(stage='trained',exit_code=0)))
        (directory/'MIRROR_VERIFIED.json').write_bytes(encoded(dict(artifact_bytes=240)))
        inventory={}
        for epoch in range(240):
            name=f'attempt-001/checkpoints/ckpt_epoch{epoch:03}.bin';path=mirror/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(bytes([epoch]));inventory[name]=sha(path)
        (checkpoints/'ckpt_epoch239.bin').write_bytes(bytes([239]))
        return study,cell,mirror,inventory

    def test_missing_checkpoints_are_restored_without_overwriting_an_existing_one(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,cell,mirror,inventory=self.fixture(root)
            existing=study/'runs/cell/attempt-001/checkpoints/ckpt_epoch239.bin';before=existing.stat().st_mtime_ns
            with patch('shaping_execution.load_registration',return_value=(dict(minimum_ssd_free_bytes=0),[cell],{})), \
                 patch('shaping_checkpoint_retention.archive_inventory',return_value=(mirror,inventory)):
                result=restore(study,'cell',root/'proof')
            self.assertEqual(len(result['restored']),239);self.assertEqual(len(result['already_present']),1)
            self.assertTrue(result['all_240_original_checkpoints_verified']);self.assertEqual(existing.stat().st_mtime_ns,before)

    def test_changed_archive_cannot_be_published_as_an_original_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,cell,mirror,inventory=self.fixture(root)
            (mirror/'attempt-001/checkpoints/ckpt_epoch000.bin').write_bytes(b'drift')
            with patch('shaping_execution.load_registration',return_value=(dict(minimum_ssd_free_bytes=0),[cell],{})), \
                 patch('shaping_checkpoint_retention.archive_inventory',return_value=(mirror,inventory)):
                with self.assertRaisesRegex(ValueError,'archived checkpoint differs'):restore(study,'cell',root/'proof')
            self.assertFalse((study/'runs/cell/attempt-001/checkpoints/ckpt_epoch000.bin').exists())
            self.assertTrue((study/'runs/cell/attempt-001/checkpoints/ckpt_epoch000.bin.restoring').exists())


if __name__=='__main__':unittest.main()
