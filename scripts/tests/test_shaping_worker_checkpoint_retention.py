import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_worker_checkpoint_retention import remove_remote,selected,sha
from shaping_checkpoint_retention import retained_epochs


class WorkerRetentionTests(unittest.TestCase):
    def fixture(self,base):
        root=base/'study';run=root/'runs'/('a'*20)/'attempt-001';(run/'checkpoints').mkdir(parents=True)
        (root/'PLAN.json').write_text('{"input_pins":{}}')
        result=dict(stage='trained',exit_code=0,cell={'id':'a'*20},epochs=120)
        for name in ['RUN.json','RESULT.json']:(run/name).write_text(json.dumps(result))
        files={}
        for i in range(120):
            name=f'checkpoints/ckpt_epoch{i:03}.bin';p=run/name;p.write_bytes(bytes([i]));files[name]=dict(bytes=1,sha256=sha(p))
        row=dict(cell='a'*20,epochs=120,keep=sorted(retained_epochs(120)),files=files,metadata={n:sha(run/n) for n in ['RUN.json','RESULT.json']})
        return dict(study=str(root),receipt=str(base/'receipt.json'),plan_sha256=sha(root/'PLAN.json'),cells=[row]),run

    def test_removes_only_intermediates(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp,patch('shaping_worker_checkpoint_retention.no_users') as users:
            payload,run=self.fixture(Path(temp));result=remove_remote(payload)
            self.assertEqual(result['removed_files'],120-len(retained_epochs(120)))
            self.assertEqual(users.call_count,2)
            self.assertEqual({int(p.stem.removeprefix('ckpt_epoch')) for p in (run/'checkpoints').glob('*.bin')},retained_epochs(120))
            with self.assertRaisesRegex(ValueError,'preserve'):remove_remote(payload)

    def test_refuses_changes_busy_pins_and_extra_attempts(self):
        for mode in ['hash','metadata','busy','pinned','attempt','duplicate']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp,patch('shaping_worker_checkpoint_retention.no_users') as users:
                payload,run=self.fixture(Path(temp));root=Path(payload['study'])
                if mode=='hash':(run/'checkpoints/ckpt_epoch001.bin').write_bytes(b'x')
                if mode=='metadata':(run/'RESULT.json').write_text('{}')
                if mode=='busy':users.side_effect=ValueError('busy')
                if mode=='pinned':
                    (root/'PLAN.json').write_text(json.dumps({'input_pins':{str(run/'checkpoints/ckpt_epoch001.bin'):'pin'}}));payload['plan_sha256']=sha(root/'PLAN.json')
                if mode=='attempt':(run.parent/'attempt-002').mkdir()
                if mode=='duplicate':payload['cells']*=2
                with self.assertRaises(ValueError):remove_remote(payload)
                self.assertEqual(len(list((run/'checkpoints').glob('*.bin'))),120)

    def test_inventory_and_required_epochs(self):
        with self.assertRaisesRegex(ValueError,'inventory'):selected({},120,[0,119])
        files={f'checkpoints/ckpt_epoch{i:03}.bin':{} for i in range(120)}
        with self.assertRaisesRegex(ValueError,'required'):selected(files,120,[0])


if __name__=='__main__':unittest.main()
