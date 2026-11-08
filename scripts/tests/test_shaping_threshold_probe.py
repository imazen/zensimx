import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_threshold_probe import CELLS,command_for


class ThresholdProbeTests(unittest.TestCase):
    def test_only_budget_binary_and_fresh_destinations_change(self):
        command=['heavy','--','old-trainer','--epochs','120','--out','old-best','--dump-checkpoints-dir','old-checkpoints',
                 '--pairs-per-epoch','50000','--init-seed','1001107','--keep-features','original-keep']
        original=dict(cell=dict(id=CELLS[0],partition='inner-0-0',family='separate_pretrained_frozen',setting=3),command=command)
        before=copy.deepcopy(original)
        changed=command_for(original,Path('/qualified/trainer'),Path('/fresh'))
        self.assertEqual(original,before)
        self.assertEqual([(a,b) for a,b in zip(command,changed) if a!=b],
                         [('old-trainer','/qualified/trainer'),('120','2'),('old-best','/fresh/best.bin'),
                          ('old-checkpoints','/fresh/checkpoints')])
        original['cell']['partition']='outer-0'
        with self.assertRaises(ValueError):command_for(original,Path('/qualified/trainer'),Path('/fresh'))


if __name__=='__main__':unittest.main()
