import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_rescue_qualification import human_command

class HumanQualificationTests(unittest.TestCase):
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
