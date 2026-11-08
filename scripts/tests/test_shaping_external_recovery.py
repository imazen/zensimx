import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_external_recovery import CELLS,INSTRUCTION,CAUSE,NAME,register,authorize_attempt,verify_attempt,recover


class ExternalRecoveryTests(unittest.TestCase):
    def fixture(self,root):
        (root/'PLAN.json').write_text('{}');direction=root/'DIRECTION.json'
        direction.write_bytes(encoded(dict(user_instruction=INSTRUCTION,conditional_resumption_authorized=True)))
        for cell in CELLS:
            run=root/'runs'/cell/'attempt-001';(run/'checkpoints').mkdir(parents=True)
            value=dict(cell=dict(id=cell),execution_host='r5600g',build_sha256='original-build')
            (run/'RUN.json').write_bytes(encoded(value))
            (run/'RESULT.json').write_bytes(encoded(dict(value,stage='interrupted',reason=CAUSE,saved_checkpoints=0)))
            (run/'train.log').write_text('loaded inputs; no completed epoch')
            (run/'SHAPING.json').write_text('original bank');(run/'keep.txt').write_text('original features')
        return direction

    def test_authorization_only_accepts_fixed_external_zero_epoch_cases(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);direction=self.fixture(root)
            with patch('shaping_external_recovery.socket.gethostname',return_value='r5600g'),patch('shaping_external_recovery.subprocess.check_output',return_value='yes\n'):
                register(root,direction)
            row,pin=authorize_attempt(root,CELLS[0],'attempt-002')
            run=dict(cell=row['cell'],build_sha256='original-build',recovery_authorization_sha256=pin,recovery_completed_epochs_replayed=0)
            verify_attempt(root,run,'attempt-002')
            with self.assertRaisesRegex(ValueError,'differs'):authorize_attempt(root,CELLS[0],'attempt-003')
            with self.assertRaisesRegex(ValueError,'differs'):authorize_attempt(root,'other','attempt-002')
            run['build_sha256']='corrected-build'
            with self.assertRaisesRegex(ValueError,'explicit authorization'):verify_attempt(root,run,'attempt-002')

    def test_partial_checkpoint_or_unfixed_lifecycle_refuses_registration(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);direction=self.fixture(root)
            with patch('shaping_external_recovery.socket.gethostname',return_value='r5600g'),patch('shaping_external_recovery.subprocess.check_output',return_value='no\n'):
                with self.assertRaisesRegex(ValueError,'lingering'):register(root,direction)
            self.assertFalse((root/NAME).exists())
            (root/'runs'/CELLS[0]/'attempt-001/checkpoints/ckpt_epoch000.bin').write_text('partial evidence')
            with patch('shaping_external_recovery.socket.gethostname',return_value='r5600g'),patch('shaping_external_recovery.subprocess.check_output',return_value='yes\n'):
                with self.assertRaisesRegex(ValueError,'zero-epoch'):register(root,direction)
            self.assertFalse((root/NAME).exists())

    def test_missing_authorization_keeps_later_attempt_gate(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            with self.assertRaisesRegex(ValueError,'explicit reconciliation'):authorize_attempt(Path(d),CELLS[0],'attempt-002')

    def test_initial_boundary_retry_preserves_command_inputs_and_old_attempt(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            base=Path(d);root=base/'study';root.mkdir();direction=self.fixture(root)
            (root/'PLAN.json').write_bytes(encoded(dict(minimum_ssd_free_bytes=0)))
            binaries=base/'study-bin';binaries.mkdir();binary=binaries/'zensim_mlp_train';binary.write_text('frozen executable')
            build=binaries/'BUILD.json';build.write_bytes(encoded(dict(binary_sha256={'zensim_mlp_train':sha(binary)})))
            for cell in CELLS:
                run=root/'runs'/cell/'attempt-001'
                command=['run-heavy','--mem','2G','--jobs','1','--',str(binary),'--lr','0.001','--init-seed','1001107','--out',str(run/'best.bin'),'--keep-features',str(run/'keep.txt'),'--dump-checkpoints-dir',str(run/'checkpoints')]
                identity=json.loads((run/'RUN.json').read_text());identity.update(command=command,build_sha256=sha(build),epochs=120)
                (run/'RUN.json').write_bytes(encoded(identity));(run/'SHAPING.provenance.json').write_text('{}')
            with patch('shaping_external_recovery.socket.gethostname',return_value='r5600g'),patch('shaping_external_recovery.subprocess.check_output',return_value='yes\n'):
                register(root,direction)
            old=root/'runs'/CELLS[0]/'attempt-001';before=(old/'RUN.json').read_bytes()
            def launch(command,**kwargs):
                path=Path(command[command.index('--dump-checkpoints-dir')+1])
                for epoch in range(120):(path/f'ckpt_epoch{epoch:03}.bin').write_text(str(epoch))
                self.assertEqual(kwargs['env']['ZENSIM_INPUT_SHAPING_SHA256'],sha(old/'SHAPING.json'))
                return SimpleNamespace(pid=12345,wait=lambda:0)
            with patch('shaping_external_recovery.socket.gethostname',return_value='r5600g'),patch('shaping_external_recovery.subprocess.check_output',return_value='yes\n'),patch('shaping_external_recovery.subprocess.Popen',side_effect=launch) as native,patch('shaping_reduced_scope.admit_cell'),patch('shaping_rescue_dispatch.reserve_worker',return_value={'can_start':True}),patch('shaping_rescue_dispatch.bind_worker'),patch('shaping_rescue_dispatch.release_worker') as release:
                result=recover(SimpleNamespace(study=root,cell=CELLS[0]))
            self.assertEqual(result['stage'],'trained');self.assertEqual(result['recovery_completed_epochs_replayed'],0)
            self.assertEqual((old/'RUN.json').read_bytes(),before);release.assert_called_once_with(root,CELLS[0])
            argv=native.call_args.args[0];original=json.loads(before)['command']
            for flag in ('--out','--keep-features','--dump-checkpoints-dir'):original[original.index(flag)+1]=argv[argv.index(flag)+1]
            self.assertEqual(argv,original)


if __name__=='__main__':unittest.main()
