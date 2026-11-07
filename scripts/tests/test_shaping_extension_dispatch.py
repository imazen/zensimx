import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_extension_dispatch import dispatch,existing


class ExtensionDispatchTests(unittest.TestCase):
    def test_unregistered_extensions_cannot_start(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            args=SimpleNamespace(study=Path(directory),outer=0,workers=2,session='first',watch_registration=False)
            with patch('shaping_extension_dispatch.subprocess.Popen') as launch:
                with self.assertRaisesRegex(ValueError,'registration required'):dispatch(args)
                launch.assert_not_called()

    def test_final_checkpoint_and_exposure_must_match_and_failures_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}')
            cell=dict(id='c',phase_epochs=[240,240],total_pair_draws=24000000)
            run=root/'runs/c/attempt-001';(run/'fit/checkpoints').mkdir(parents=True)
            checkpoint=run/'fit/checkpoints/ckpt_epoch239.bin';checkpoint.write_bytes(b'final phase checkpoint')
            identity=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),build_sha256='build',process_role='extension_controller',
                epochs=240,training_exposure_pair_draws=24000000,resolution=dict(cell=cell,registration_sha256='registered'))
            (run/'RUN.json').write_bytes(encoded(identity));result=dict(identity,stage='trained',exit_code=0,bake_sha256=sha(checkpoint))
            (run/'RESULT.json').write_bytes(encoded(result))
            self.assertEqual(existing(root,cell,'registered','build'),'trained')
            with self.assertRaisesRegex(ValueError,'identity differs'):existing(root,cell,'changed','build')
            result.update(stage='failed',exit_code=1);(run/'RESULT.json').write_bytes(encoded(result));before=(run/'RESULT.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'fit failed'):existing(root,cell,'registered','build')
            self.assertEqual(before,(run/'RESULT.json').read_bytes())
            result.update(stage='trained',exit_code=0);(run/'RESULT.json').write_bytes(encoded(result));checkpoint.write_bytes(b'wrong checkpoint')
            with self.assertRaisesRegex(ValueError,'checkpoint drift'):existing(root,cell,'registered','build')

    def test_empty_verified_extension_matrix_finishes_without_training(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);path=root/'extensions/outer-0/REGISTRATION.json';path.parent.mkdir(parents=True);path.write_text('{}')
            binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            args=SimpleNamespace(study=root,outer=0,workers=2,session='first',watch_registration=False,binary_dir=binary)
            with patch('shaping_extension_dispatch.admitted',return_value=(dict(cells=[]),path)),patch('shaping_extension_dispatch.subprocess.Popen') as launch:
                dispatch(args);launch.assert_not_called()
            status=json.loads((root/'dispatch/extensions/outer-0/first/STATUS.json').read_text())
            self.assertEqual(status['stage'],'training_finished');self.assertEqual(status['completed'],[])

    def test_queue_preserves_phase_exposure_and_two_worker_limit(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}');path=root/'extensions/outer-0/REGISTRATION.json';path.parent.mkdir(parents=True);path.write_text('{}')
            binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            cells=[dict(id=str(i),phase_epochs=phases,total_pair_draws=sum(phases)*50000) for i,phases in enumerate(([480],[240,240],[240]))]
            live=0;peak=0;started=[]
            class Process:
                def __init__(process,command,**kwargs):
                    nonlocal live,peak
                    process.cell=next(c for c in cells if c['id']==command[command.index('--cell')+1]);started.append(process.cell['id'])
                    process.returncode=None;process.pid=1000+len(started);live+=1;peak=max(peak,live)
                def poll(process):
                    nonlocal live
                    if process.returncode is not None:return process.returncode
                    cell=process.cell;run=root/'runs'/cell['id']/'attempt-001';(run/'fit/checkpoints').mkdir(parents=True)
                    epochs=cell['phase_epochs'][-1];checkpoint=run/'fit/checkpoints'/f'ckpt_epoch{epochs-1:03}.bin';checkpoint.write_bytes(b'queue final checkpoint')
                    identity=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),build_sha256=sha(binary/'BUILD.json'),process_role='extension_controller',
                        epochs=epochs,training_exposure_pair_draws=cell['total_pair_draws'],resolution=dict(cell=cell,registration_sha256=sha(path)))
                    (run/'RUN.json').write_bytes(encoded(identity));(run/'RESULT.json').write_bytes(encoded(dict(identity,stage='trained',exit_code=0,bake_sha256=sha(checkpoint))))
                    live-=1;process.returncode=0;return 0
            args=SimpleNamespace(study=root,outer=0,workers=2,session='first',watch_registration=False,binary_dir=binary,neutral_qualification=root/'neutral')
            with patch('shaping_extension_dispatch.admitted',return_value=(dict(cells=cells),path)),patch('shaping_extension_dispatch.subprocess.Popen',side_effect=Process),patch('shaping_extension_dispatch.time.sleep'):
                dispatch(args)
            self.assertEqual(peak,2);self.assertEqual(live,0);self.assertEqual(started,['0','1','2'])
            status=json.loads((root/'dispatch/extensions/outer-0/first/STATUS.json').read_text());self.assertEqual(status['stage'],'training_finished')


if __name__=='__main__':unittest.main()
