import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_partial_restart import CELLS,NAME,register,authorize,compare_available
from shaping_external_recovery import verify_attempt


class PartialRestartTests(unittest.TestCase):
    def fixture(self,study):
        (study/'PLAN.json').write_text('{}');rows=[]
        for cell,count in CELLS.items():
            old=study/'runs'/cell/'attempt-001';(old/'checkpoints').mkdir(parents=True)
            run=dict(cell=dict(id=cell),command=['frozen'],build_sha256='frozen-build')
            (old/'RUN.json').write_bytes(encoded(run))
            (old/'RESULT.json').write_bytes(encoded(dict(stage='interrupted')))
            (old/'SHAPING.json').write_text('bank');(old/'keep.txt').write_text('features')
            for e in range(count):(old/'checkpoints'/f'ckpt_epoch{e:03}.bin').write_text(str(e))
            rows.append(dict(cell=run['cell'],saved_epochs=count,original_epochs=240,new_epochs=240,target_host='i134',
                new_attempt='attempt-002',original_command=run['command'],frozen_build_sha256='frozen-build',
                original_run_sha256=sha(old/'RUN.json'),original_result_sha256=sha(old/'RESULT.json'),
                original_bank_sha256=sha(old/'SHAPING.json'),checkpoint_sha256={p.name:sha(p) for p in (old/'checkpoints').iterdir()}))
        proposal=study/'proposal.json';proposal.write_bytes(encoded(dict(scope=rows)))
        approval=study/'approval.json';approval.write_bytes(encoded(dict(answer='Restart only these two',proposal_sha256=sha(proposal),
            target_host='i134',old_attempts_preserved=True,original_binary_seeds_data_and_banks=True,compare_all_available_checkpoints=True)))
        return proposal,approval

    def test_exact_two_cell_scope_binary_and_replayed_prefix(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study=Path(d);proposal,approval=self.fixture(study);register(study,proposal,approval)
            for cell,count in CELLS.items():
                row,pin=authorize(study,cell,'attempt-002')
                run=dict(cell=row['cell'],build_sha256='frozen-build',recovery_authorization_sha256=pin,recovery_completed_epochs_replayed=count)
                verify_attempt(study,run,'attempt-002')
                run['recovery_completed_epochs_replayed']=0
                with self.assertRaises(ValueError):verify_attempt(study,run,'attempt-002')
                with self.assertRaises(ValueError):authorize(study,cell,'attempt-003')

    def test_approval_or_original_drift_refuses_before_registration(self):
        for kind in ('approval','checkpoint'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                study=Path(d);proposal,approval=self.fixture(study)
                if kind=='approval':approval.write_bytes(encoded(dict(json.loads(approval.read_text()),answer='no')))
                else:(study/'runs'/next(iter(CELLS))/'attempt-001/checkpoints/ckpt_epoch000.bin').write_text('drift')
                with self.assertRaises(ValueError):register(study,proposal,approval)
                self.assertFalse((study/NAME).exists())

    def test_only_completed_write_receipts_trigger_comparison_and_mismatch_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);old=root/'old';dest=root/'new'
            for p in (old,dest):(p/'checkpoints').mkdir(parents=True)
            name='ckpt_epoch000.bin';(old/'checkpoints'/name).write_text('old');(dest/'checkpoints'/name).write_text('different')
            (dest/'train.log').write_text('')
            row=dict(checkpoint_sha256={name:sha(old/'checkpoints'/name)})
            self.assertFalse(compare_available(old,dest,row,Path('converter'))['compared'])
            (dest/'train.log').write_text('  checkpoint dump: '+json.dumps(dict(path=str(dest/'checkpoints'/name)))+'\n')
            def strip(command,**kwargs):Path(command[command.index('--out')+1]).write_bytes(Path(command[command.index('--in')+1]).read_bytes())
            with patch('shaping_partial_restart.subprocess.run',side_effect=strip) as native:
                result=compare_available(old,dest,row,Path('converter'));compare_available(old,dest,row,Path('converter'))
            self.assertFalse(result['compared'][name]['equal']);self.assertEqual(native.call_count,2)
            self.assertEqual((old/'checkpoints'/name).read_text(),'old')


if __name__=='__main__':unittest.main()
