import sys
from pathlib import Path
import unittest,json,tempfile,hashlib
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_rescue_qualification import human_command,qualify_on_worker
from shaping_completion import encoded,sha

class HumanQualificationTests(unittest.TestCase):
    def test_worker_uses_unique_human_admission_key_and_existing_human_owner(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);run=root/'runs'/('a'*20)/'attempt-001';run.mkdir(parents=True)
            (run/'RUN.json').write_bytes(encoded(dict(cell=dict(id='a'*20))))
            binaries=root/'binaries';binaries.mkdir();(binaries/'BUILD.json').write_text('{}')
            (root/'PLAN.json').write_text(json.dumps(dict(minimum_ssd_free_bytes=30*1024**3)))
            key=hashlib.sha256(encoded(dict(source_cell='a'*20,human_only_cell='human',build_sha256=sha(binaries/'BUILD.json')))).hexdigest()[:20]
            args=SimpleNamespace(study=root,run=run,binary_dir=binaries,human_only_cell='human',output=root/key)
            with patch('shaping_reduced_scope.admit_compute'),patch('shaping_rescue_dispatch.reserve_worker',return_value=dict(can_start=True)) as reserve, \
                    patch('shaping_rescue_dispatch.bind_worker'),patch('shaping_rescue_dispatch.release_worker') as release, \
                    patch('shaping_storage.reservation',return_value=nullcontext()),patch('shaping_rescue_qualification.qualify_human',return_value='qualified') as human, \
                    patch('shaping_rescue_qualification.qualify') as original:
                self.assertEqual(qualify_on_worker(args),'qualified');human.assert_called_once_with(args);original.assert_not_called()
                reserve.assert_called_once_with(root,key);release.assert_called_once_with(root,key)
                args.output=root/'wrong-output'
                with self.assertRaisesRegex(ValueError,'unique admission'):qualify_on_worker(args)
                self.assertEqual(reserve.call_count,1)

    def test_replace_teacher_and_preserve_scientific_flags(self):
        groups=[(n,'/data/'+n+'.parquet',1.,0.,'withinref,rank') for n in
                ('human_kadid','human_tid2013','human_konfig','human_cid22_a25')]
        source=['run-heavy','--mem','1G','--','old/zensim_mlp_train','--group','cid22:old',
                '--epochs','240','--mse-weight','1','--out','old/best.bin',
                '--dump-checkpoints-dir','old/ckpt','--init-seed','1101','--sample-seed','101',
                '--lr','0.0003','--l2','0.00001','--pairs-per-epoch','50000']
        out=human_command(source,groups,Path('/bin'),Path('/out'))
        self.assertNotIn('cid22:old',out)
        self.assertEqual(out.count('--group'),4)
        for flag,value in [('--epochs','1'),('--mse-weight','0'),('--init-seed','1101'),
                           ('--sample-seed','101'),('--lr','0.0003'),('--l2','0.00001'),('--pairs-per-epoch','50000')]:
            self.assertEqual(out[out.index(flag)+1],value)
        self.assertIn('cid22:old',source)
        with self.assertRaisesRegex(ValueError,'four human'):
            human_command(source,groups+[('cid22','/teacher',1.,0.,'withinref,both')],Path('/bin'),Path('/out'))
        bad=groups.copy();bad[0]=(*bad[0][:3],1.,bad[0][4])
        with self.assertRaisesRegex(ValueError,'TRAIN-only'):
            human_command(source,bad,Path('/bin'),Path('/out'))

if __name__=='__main__':unittest.main()
