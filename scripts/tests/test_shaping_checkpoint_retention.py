import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_checkpoint_retention import reclaim,retained_epochs,open_targets
from shaping_completion import sha
from shaping_evaluation import diagnostic_epochs

class RetentionTests(unittest.TestCase):
    def fixture(self,root):
        run=root/'runs/cell/attempt-001';(run/'checkpoints').mkdir(parents=True)
        (run/'RUN.json').write_text('{"epochs":120}')
        mirror=root/'mirror';mirror.mkdir();rows=[]
        for epoch in [0,1,2,49,99,119]:
            p=run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin';p.write_bytes(str(epoch).encode())
            q=mirror/p.name;q.write_bytes(p.read_bytes())
            if epoch in [1,2]:rows.append(dict(path=str(p),durable_path=str(q),sha256=sha(p),epoch=epoch,cell='cell',bytes=p.stat().st_size,retained_epochs=sorted(retained_epochs(120))))
        return rows

    def test_required_epochs(self):
        for epochs in [120,240,480]:
            self.assertTrue({0,119,epochs-1,*diagnostic_epochs(epochs)}<=retained_epochs(epochs))
        self.assertEqual(open_targets('p123\nn/a\nn/b\n',{'/b'}),['/b'])

    def test_verified_removal_preserves_needed_and_durable(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d,patch('shaping_checkpoint_retention.no_users'):
            root=Path(d);rows=self.fixture(root);receipt=root/'receipt.json'
            reclaim(root,rows,receipt)
            for row in rows:
                self.assertFalse(Path(row['path']).exists());self.assertEqual(sha(row['durable_path']),row['sha256'])
            self.assertTrue((root/'runs/cell/attempt-001/checkpoints/ckpt_epoch119.bin').exists())
            self.assertTrue((root/'runs/cell/LOCAL_CHECKPOINT_RETENTION.json').exists())
            with self.assertRaisesRegex(ValueError,'existing'):reclaim(root,rows,receipt)

    def test_late_corruption_removes_nothing(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d,patch('shaping_checkpoint_retention.no_users'):
            root=Path(d);rows=self.fixture(root);Path(rows[-1]['durable_path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'readback'):reclaim(root,rows,root/'receipt.json')
            self.assertTrue(all(Path(r['path']).exists() for r in rows))
            self.assertFalse((root/'receipt.json').exists())

    def test_busy_or_changed_sources_refused(self):
        for change in [False,True]:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);rows=self.fixture(root)
                def inspect(_):
                    if not change:raise ValueError('busy')
                    if inspect.calls:Path(rows[0]['path']).write_bytes(b'changed')
                    inspect.calls+=1
                inspect.calls=0
                with patch('shaping_checkpoint_retention.no_users',side_effect=inspect):
                    with self.assertRaises(ValueError):reclaim(root,rows,root/'receipt.json')
                self.assertTrue(all(Path(r['path']).exists() for r in rows))

    def test_required_epoch_and_duplicate_paths_refused(self):
        for mode in ['required','duplicate','outside']:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d,patch('shaping_checkpoint_retention.no_users'):
                root=Path(d);rows=self.fixture(root)
                if mode=='required':
                    rows[0]['epoch']=119
                    rows[0]['path']=str(root/'runs/cell/attempt-001/checkpoints/ckpt_epoch119.bin')
                elif mode=='duplicate':rows.append(dict(rows[0]))
                else:rows[0]['path']=str(root/'unrelated.bin')
                with self.assertRaises(ValueError):reclaim(root,rows,root/'receipt.json')
                self.assertTrue((root/'runs/cell/attempt-001/checkpoints/ckpt_epoch001.bin').exists())
                self.assertFalse((root/'receipt.json').exists())

if __name__=='__main__':unittest.main()
