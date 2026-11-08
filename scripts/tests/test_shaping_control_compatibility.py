import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_control_compatibility as control
from shaping_completion import REPO,encoded,sha


class CompatibilityTests(unittest.TestCase):
    def command(self,root):
        return ['run-heavy','--mem','1G','--jobs','1','--','/old/zensim_mlp_train',
                '--epochs','240','--pairs-per-epoch','50000','--init-seed','1101','--sample-seed','101',
                '--out','original-best','--dump-checkpoints-dir','original-checkpoints',
                '--derived-inputs',str(REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json')]

    def test_control_command_changes_only_binary_memory_and_output_paths(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);command=self.command(root);original=dict(command=command,_source=str(root))
            before=copy.deepcopy(original);result=control.command_for(original,root/'bin',root/'out')
            self.assertEqual(original,before)
            expected=list(command)
            expected[6]=str(root/'bin/zensim_mlp_train')
            for key,value in [('--mem','2G'),('--out',str(root/'out/best.bin')),('--dump-checkpoints-dir',str(root/'out/checkpoints'))]:expected[expected.index(key)+1]=value
            self.assertEqual(result,expected)
            original['command'][command.index('--epochs')+1]='120'
            with self.assertRaisesRegex(ValueError,'full control exposure'):control.command_for(original,root/'bin',root/'out')

    def fixture(self,root):
        study=root/'study';source=study/'runs/cell/attempt-001';models=source/'checkpoints';models.mkdir(parents=True)
        (study/'PLAN.json').write_bytes(encoded(dict(minimum_ssd_free_bytes=0)))
        original=dict(cell=dict(id='cell',family='relu'),command=self.command(root))
        (source/'RUN.json').write_bytes(encoded(original));(source/'RESULT.json').write_text('{}')
        for epoch in range(240):(models/f'ckpt_epoch{epoch:03}.bin').write_bytes(bytes([epoch]))
        binaries=root/'bin';binaries.mkdir();tool=binaries/'zensim_mlp_train';tool.write_text('frozen test tool')
        (binaries/'BUILD.json').write_bytes(encoded(dict(binary_sha256={'zensim_mlp_train':sha(tool)})))
        approval=root/'APPROVAL.json';approval.write_text('{}')
        record=dict(schema='corrected-runtime-control-compatibility-v1',outer_results_read=False,
            plan_sha256=sha(study/'PLAN.json'),build_sha256=sha(binaries/'BUILD.json'),driver_sha256=sha(control.__file__),
            approval_path=str(approval),approval_sha256=sha(approval),cases={'relu':dict(cell=original['cell'],run=str(source),
                source_file_sha256={str(source/'RUN.json'):sha(source/'RUN.json')},
                original_final_sha256=sha(models/'ckpt_epoch239.bin'))})
        registration=root/'REGISTRATION.json';registration.write_bytes(encoded(record))
        args=SimpleNamespace(study=study,binary_dir=binaries,registration=registration,family='relu',output=root/'new-control')
        return args,source,record

    def test_native_result_and_all_epoch_comparison_are_required_without_replaying(self):
        for equal in (True,False):
            with self.subTest(equal=equal),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);args,source,record=self.fixture(root)
                child=SimpleNamespace(pid=123,wait=lambda:0)
                with patch.object(control,'BUILD_SHA256',record['build_sha256']),patch('shaping_reduced_scope.admit_compute'), \
                     patch('shaping_rescue_dispatch.reserve_worker',return_value={'can_start':True}) as claim, \
                     patch('shaping_rescue_dispatch.bind_worker'),patch('shaping_rescue_dispatch.release_worker') as release, \
                     patch.object(control.subprocess,'Popen',return_value=child) as native, \
                     patch.object(control,'compare_checkpoints',return_value=dict(all_equal=equal)) as compare:
                    if equal:self.assertEqual(control.run(args)['stage'],'qualified')
                    else:
                        with self.assertRaisesRegex(ValueError,'trajectory differs'):control.run(args)
                    compare.assert_called_once()
                    self.assertEqual(compare.call_args.args[-2:],(240,('zentrain.repro',)))
                    self.assertEqual(native.call_count,1);release.assert_called_once()
                    env=native.call_args.kwargs['env'];self.assertEqual(env['OMP_NUM_THREADS'],'1')
                    self.assertEqual(env['ZENSIM_TRAIN_SNAPSHOT_ROOT'],str(args.study.parent/'shared-training-snapshots'))
                    with self.assertRaisesRegex(ValueError,'never replay'):control.run(args)
                    self.assertEqual(claim.call_count,1);self.assertEqual(native.call_count,1)
                result=json.loads((args.output/'RESULT.json').read_text())
                self.assertEqual(result['all_240_checkpoints_equal'],equal)

    def test_missing_original_checkpoint_cannot_start_a_compatibility_control(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            args,source,record=self.fixture(Path(directory));(source/'checkpoints/ckpt_epoch000.bin').unlink()
            with patch.object(control,'BUILD_SHA256',record['build_sha256']),patch('shaping_reduced_scope.admit_compute'), \
                 patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'restore every original'):control.run(args)
                claim.assert_not_called();self.assertFalse(args.output.exists())


if __name__=='__main__':unittest.main()
