"""Replay packaging accepts only the exact native command and bounded scope."""
import shlex,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_stable_pairs_archive import prediction_command


class StableReplayPackagingTests(unittest.TestCase):
    def test_command_scope_and_argument_shape_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);log=root/'predict.log'
            command=['/frozen/bin/bake_dial_refit','predict','--bake',str(root/'dense.bin'),
                     '--corpus','/registered/inner.parquet','--score-units','--out',str(root/'pairs.tsv')]
            log.write_text('$ '+shlex.join(command)+'\nnative log\n');self.assertEqual(prediction_command(log,root),command)
            for changed in ([*command,'--extra'],['wrong',*command[1:2]],
                            [*command[:3],str(root/'../outside.bin'),*command[4:]],
                            [*command[:8],'/outside/pairs.tsv']):
                log.write_text('$ '+shlex.join(changed)+'\n')
                with self.assertRaises(ValueError):prediction_command(log,root)
            log.write_text('not a command\n')
            with self.assertRaises(ValueError):prediction_command(log,root)


if __name__=='__main__':unittest.main()
