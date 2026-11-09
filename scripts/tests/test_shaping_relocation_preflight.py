import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import sha
from shaping_relocation_preflight import checkpoint_prefix,capture,preflight


class RelocationPreflightTests(unittest.TestCase):
    def test_complete_saved_prefix_matches_original_native_receipts_and_detects_drift(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            phase=Path(directory)/'fit';(phase/'checkpoints').mkdir(parents=True)
            rows=[]
            for epoch in range(2):
                path=phase/'checkpoints'/f'ckpt_epoch{epoch:03}.bin';path.write_bytes(bytes([epoch]))
                rows.append(dict(epoch=epoch,path=str(path),bytes=1,sha256=sha(path)))
            state=dict(epoch=2,checkpoint_receipts=['  checkpoint dump: '+json.dumps(r) for r in rows])
            self.assertEqual(checkpoint_prefix(state,phase),rows)
            (phase/'checkpoints/ckpt_epoch001.bin').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'prefix differs'):checkpoint_prefix(state,phase)

    def test_missing_epoch_or_duplicate_receipt_is_not_a_complete_prefix(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            phase=Path(directory)/'fit';(phase/'checkpoints').mkdir(parents=True)
            path=phase/'checkpoints/ckpt_epoch000.bin';path.write_bytes(b'x')
            row=json.dumps(dict(epoch=0,path=str(path),bytes=1,sha256=sha(path)))
            for receipts in ([row],[row,row]):
                with self.assertRaisesRegex(ValueError,'coverage differs'):
                    checkpoint_prefix(dict(epoch=2,checkpoint_receipts=receipts),phase)

    def test_withdrawn_source_is_refused_before_reading_or_staging_a_snapshot(self):
        with patch('shaping_compute_pool.allowed',return_value={'r3500'}), \
             patch('shaping_relocation_preflight.socket.gethostname',return_value='i270'), \
             patch('shaping_relocation_preflight.read_state') as read:
            with self.assertRaisesRegex(ValueError,'eligible workers'):
                capture(Path('/unused-study'),'a'*20,'r3500',Path('/unused-output'))
        read.assert_not_called()

    def test_changed_plan_refuses_before_storage_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}')
            placement=root/'placement.json';placement.write_text(json.dumps(dict(plan_sha256='wrong',outer_results_read=False,placements={'a'*20:'r3500'})))
            with patch('shaping_storage.reservation') as reserve:
                with self.assertRaisesRegex(ValueError,'plan-bound'):preflight(root,placement,root/'output')
            reserve.assert_not_called();self.assertFalse((root/'output').exists())


if __name__=='__main__':unittest.main()
