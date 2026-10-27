"""The full-fit launcher cannot turn a rejected or changed screen into a fit."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"byconstruction"))
import mlic_vote_full as full


class FullVoteAdmission(unittest.TestCase):
    def selection(self):
        arms={a:{m:dict(per_seed=[0.8]*3) for m in ("old","new","balanced")} for a in ("Mlic","MlicVote")}
        arms["MlicVote"]["balanced"]["per_seed"]=[0.81]*3
        return dict(schema="mlic-native-vote-TRAIN-selection-v1",rule_commit="b2ee920aa351",eval_labels_used=False,
            arms=arms,mixtures=dict(mixture_global={f"MlicVote_{f}_s{s}":dict(inversions=0)
                for f in full.FOLDS for s in range(3)}),**full.neff_decision(arms,[0]*21,"MlicVote"))

    def test_only_exact_registered_passing_decision_can_advance(self):
        selection=self.selection();full.admitted_selection(selection)
        for key,value in (("eval_labels_used",True),("rule_commit","other"),("production_qualified",True),
                          ("finalist","Mlic")):
            changed=copy.deepcopy(selection);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):full.admitted_selection(changed)
        for metric in ("old","balanced"):
            changed=copy.deepcopy(selection)
            changed["arms"]["MlicVote"][metric]["per_seed"]=[0.79]*3
            changed.update(full.neff_decision(changed["arms"],[0]*21,"MlicVote"))
            with self.assertRaisesRegex(ValueError,"did not pass"):full.admitted_selection(changed)
        selection["mixtures"]["mixture_global"]["MlicVote_kadid_s0"]["inversions"]=1
        with self.assertRaisesRegex(ValueError,"differs"):full.admitted_selection(selection)

    def test_full_command_has_fixed_seeds_strict_training_and_existing_packer(self):
        for seed in range(3):
            command=full.full_command(Path("source"),Path("root"),Path("dest"),"13,14",seed)
            self.assertEqual(command[command.index("--seed-index")+1],str(seed))
            self.assertEqual(command[command.index("--spec")+1],full.SPEC)
            self.assertIn("--strict-admission",command)
            self.assertIn("--train-only",command)
            self.assertIn("--pack-production",command)
            self.assertIn("--e33-output-stage",command)
            self.assertNotIn("--local-smoke-budget",command)
            self.assertEqual(command[2],"source/scripts/rev4_featpot/v2_confirm_fit.py")
        with self.assertRaises(ValueError):full.full_command(Path("s"),Path("r"),Path("d"),"13",3)


if __name__=="__main__":unittest.main()
