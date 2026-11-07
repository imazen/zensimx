import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_checkpoint_retention import reclaim,retained_epochs,open_targets,candidates
from shaping_completion import sha
from shaping_evaluation import diagnostic_epochs

class RetentionTests(unittest.TestCase):
    def rescue_fixture(self,root,pretrained=True):
        (root/'PLAN.json').write_text('{"input_pins":{}}')
        cell=dict(id='rescue',stage='rescue',family='product_pretrained_frozen' if pretrained else 'relu')
        directory=root/'runs/rescue';run=directory/'attempt-001';run.mkdir(parents=True)
        layout=[('pretraining',120),('fit',120)] if pretrained else [('fit',240)]
        (run/'RUN.json').write_text(json.dumps(dict(cell=cell,epochs=layout[-1][1])))
        (run/'RESULT.json').write_text(json.dumps(dict(stage='trained',phases=[dict(epochs=n) for _,n in layout])))
        for phase,epochs in layout:
            base=run/phase;(base/'checkpoints').mkdir(parents=True)
            (base/'RUN.json').write_text(json.dumps(dict(epochs=epochs,pairs_per_epoch=50000)))
            for epoch in range(epochs):(base/'checkpoints'/f'ckpt_epoch{epoch:03}.bin').write_bytes(f'{phase}:{epoch}'.encode())
        mirror=root/'durable';mirror.mkdir();hashes={}
        for path in directory.rglob('*'):
            if not path.is_file():continue
            name=path.relative_to(directory).as_posix();q=mirror/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(path.read_bytes());hashes[name]=sha(path)
        sums=''.join(f'{pin}  {name}\n' for name,pin in sorted(hashes.items()))
        for base in [directory,mirror]:(base/'SHA256SUMS').write_text(sums)
        (directory/'MIRROR_VERIFIED.json').write_text(json.dumps(dict(destination=str(mirror),all_artifact_sha256_verified=True)))
        rows=[]
        for phase,epochs in layout:
            for epoch in [1,2]:
                p=run/phase/'checkpoints'/f'ckpt_epoch{epoch:03}.bin';relative=p.relative_to(directory).as_posix()
                rows.append(dict(path=str(p),durable_path=str(mirror/relative),sha256=sha(p),cell='rescue',phase=phase,epoch=epoch,retained_epochs=sorted(retained_epochs(epochs)),bytes=p.stat().st_size))
        return cell,run,rows

    def rescue_gates(self,cell):
        return patch('shaping_checkpoint_retention.load_registration',return_value=({},[cell],{})),patch('shaping_checkpoint_retention.rescue_eligible',return_value=True)

    def test_rescue_phase_candidates_are_explicit_and_complete(self):
        for pretrained in [False,True]:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);cell,run,rows=self.rescue_fixture(root,pretrained);registration,gate=self.rescue_gates(cell)
                with registration,gate:
                    self.assertEqual(candidates(root),[])
                    found=candidates(root,rescue=True)
                phases={r['phase'] for r in found};self.assertEqual(phases,{'fit','pretraining'} if pretrained else {'fit'})
                for row in found:self.assertNotIn(row['epoch'],row['retained_epochs'])
                self.assertEqual(len(found),232)
                if not pretrained:self.assertFalse(any(r['epoch']==219 for r in found))

    def test_rescue_reclamation_preserves_both_phase_endpoints_and_durable_copies(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cell,run,rows=self.rescue_fixture(root);registration,gate=self.rescue_gates(cell)
            with registration,gate,patch('shaping_checkpoint_retention.no_users'):
                reclaim(root,rows,root/'receipt.json')
            for phase in ['fit','pretraining']:
                for epoch in retained_epochs(120):self.assertTrue((run/phase/'checkpoints'/f'ckpt_epoch{epoch:03}.bin').exists())
            for row in rows:self.assertFalse(Path(row['path']).exists());self.assertEqual(sha(row['durable_path']),row['sha256'])
            note=json.loads((run.parent/'LOCAL_CHECKPOINT_RETENTION.json').read_text());self.assertEqual(set(note['phase_retention']),{'fit','pretraining'})
            self.assertEqual(note['removed_epochs'],[1,2])

    def test_partial_second_phase_refuses_without_reclaiming_first(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cell,run,rows=self.rescue_fixture(root);registration,gate=self.rescue_gates(cell)
            (run/'fit/checkpoints/ckpt_epoch050.bin').unlink()
            with registration,gate:
                with self.assertRaisesRegex(ValueError,'partial local checkpoint inventory'):candidates(root,rescue=True)
            self.assertEqual(len(list((run/'pretraining/checkpoints').glob('*.bin'))),120)

    def test_rescue_late_corruption_removes_no_phase(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cell,run,rows=self.rescue_fixture(root);registration,gate=self.rescue_gates(cell);Path(rows[-1]['durable_path']).write_bytes(b'wrong')
            with registration,gate,patch('shaping_checkpoint_retention.no_users'):
                with self.assertRaisesRegex(ValueError,'readback'):reclaim(root,rows,root/'receipt.json')
            self.assertTrue(all(Path(r['path']).exists() for r in rows));self.assertFalse((root/'receipt.json').exists())

    def test_failed_rescue_gate_is_rechecked_before_removal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cell,run,rows=self.rescue_fixture(root)
            with patch('shaping_checkpoint_retention.load_registration',return_value=({},[cell],{})),patch('shaping_checkpoint_retention.rescue_eligible',return_value=False):
                self.assertEqual(candidates(root,rescue=True),[])
                with self.assertRaisesRegex(ValueError,'not qualified'):reclaim(root,rows,root/'receipt.json')
            self.assertTrue(all(Path(r['path']).exists() for r in rows))

    def test_rescue_phase_archive_scope_and_exposure_refusals(self):
        for mode in ['phase','archive','exposure','ordinary_run']:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);cell,run,rows=self.rescue_fixture(root);registration,gate=self.rescue_gates(cell)
                if mode=='phase':rows[0]['phase']='other'
                elif mode=='archive':
                    alias=root/'unregistered-copy.bin';alias.write_bytes(Path(rows[0]['path']).read_bytes());rows[0]['durable_path']=str(alias)
                elif mode=='exposure':(run/'pretraining/RUN.json').write_text('{"epochs":120,"pairs_per_epoch":1000}')
                else:(run/'RUN.json').write_text(json.dumps(dict(cell={**cell,'stage':'inner'},epochs=120)))
                with registration,gate,patch('shaping_checkpoint_retention.no_users'):
                    with self.assertRaises(ValueError):reclaim(root,rows,root/'receipt.json')
                self.assertTrue(all(Path(r['path']).exists() for r in rows));self.assertFalse((root/'receipt.json').exists())

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
