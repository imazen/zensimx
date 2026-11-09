"""Preserve the complete corrected phase and every non-routing native argument."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_gradient_pretrained as owner


class PretrainedCommandTests(unittest.TestCase):
    def command(self):
        return ['run-heavy','--mem','2G','--jobs','1','--','/old/zensim_mlp_train',
            '--epochs','120','--pairs-per-epoch','50000','--init-seed','1001101',
            '--sample-seed','1000101','--lr','0.001','--l2','0.0',
            '--target-column','human_score','--out','/old/best.bin','--dump-checkpoints-dir','/old/checkpoints']

    def invoke(self,command):
        with patch.object(owner,'portable_command',side_effect=lambda values,source:list(values)):
            return owner.command_for({'command':command},Path('/source'),Path('/candidate'),Path('/new'))

    def test_only_native_and_output_routing_change(self):
        original=self.command();actual=self.invoke(original)
        expected=original[:];expected[6]='/candidate/zensim_mlp_train'
        expected[expected.index('--out')+1]='/new/best.bin';expected[expected.index('--dump-checkpoints-dir')+1]='/new/checkpoints'
        self.assertEqual(actual,expected);self.assertEqual(original,self.command())

    def test_shortened_or_extended_phase_and_changed_pair_budget_refuse(self):
        for flag,value in (('--epochs','119'),('--epochs','240'),('--pairs-per-epoch','49999')):
            command=self.command();command[command.index(flag)+1]=value
            with self.subTest(flag=flag,value=value),self.assertRaisesRegex(ValueError,'complete corrected pretrained exposure'):
                self.invoke(command)

    def test_duplicate_native_or_output_arguments_refuse(self):
        for extra in (['/extra/zensim_mlp_train'],['--out','/another/best.bin'],['--epochs','120']):
            with self.subTest(extra=extra),self.assertRaises(ValueError):self.invoke(self.command()+extra)


if __name__=='__main__':unittest.main()
