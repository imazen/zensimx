import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_neutral_trajectory as trajectory


class NeutralTrajectoryTests(unittest.TestCase):
    def fixture(self,root):
        original=root/'original';neutral=root/'neutral';original.mkdir();neutral.mkdir()
        for epoch in range(3):
            for directory,label in [(original,'old'),(neutral,'new')]:
                (directory/f'ckpt_epoch{epoch:03}.bin').write_text(json.dumps(dict(
                    weights=[epoch,.25],scaler=[0.,1.],metadata={key:label for key in trajectory.STRIPPED_KEYS})))
        return original,neutral

    def strip(self,command,**kwargs):
        source=Path(command[command.index('--in')+1]);out=Path(command[command.index('--out')+1]);key=command[command.index('--key')+1]
        model=json.loads(source.read_text());del model['metadata'][key]
        out.write_text(json.dumps(model,sort_keys=True))

    def test_every_epoch_is_compared_with_only_declared_metadata_removed(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);old,new=self.fixture(root)
            with patch.object(trajectory.subprocess,'run',side_effect=self.strip) as strip:
                record=trajectory.compare_checkpoints(old,new,Path('native-tool'),root/'proof',3)
            self.assertEqual(set(record['compared']),{f'ckpt_epoch{e:03}.bin' for e in range(3)})
            self.assertTrue(record['all_equal']);self.assertEqual(strip.call_count,12)
            self.assertEqual(record['stripped_metadata_keys'],list(trajectory.STRIPPED_KEYS))

    def test_final_equality_does_not_hide_an_intermediate_weight_or_scaler_difference(self):
        for field in ('weights','scaler'):
            with self.subTest(field=field),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);old,new=self.fixture(root);path=new/'ckpt_epoch001.bin'
                model=json.loads(path.read_text());model[field][0]+=1;path.write_text(json.dumps(model))
                with patch.object(trajectory.subprocess,'run',side_effect=self.strip):
                    record=trajectory.compare_checkpoints(old,new,Path('native-tool'),root/'proof',3)
                self.assertFalse(record['all_equal']);self.assertTrue(record['compared']['ckpt_epoch002.bin']['equal'])
                self.assertFalse(record['compared']['ckpt_epoch001.bin']['equal'])

    def test_missing_epoch_cannot_be_skipped_or_counted_as_trajectory_parity(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);old,new=self.fixture(root);(old/'ckpt_epoch001.bin').unlink()
            with patch.object(trajectory.subprocess,'run') as native:
                with self.assertRaisesRegex(ValueError,'complete checkpoint inventories'):
                    trajectory.compare_checkpoints(old,new,Path('native-tool'),root/'proof',3)
                native.assert_not_called();self.assertFalse((root/'proof').exists())


if __name__=='__main__':unittest.main()
