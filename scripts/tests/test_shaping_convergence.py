import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_convergence import diagnostics


class ConvergenceTests(unittest.TestCase):
    def test_same_cycle_pair_and_loss_trigger(self):
        scores={49:.8,99:.8,119:.8,149:.8,199:.8,219:.803,239:.9}
        result=diagnostics([1.]*20+[.98]*20,scores,{k:4 for k in scores})
        self.assertEqual(result['same_cycle_epochs'],[119,219])
        self.assertEqual(len(result['reasons']),2)

    def test_one_coefficient_and_threshold_uncertainty_are_not_convergence(self):
        scores={49:.8,99:.8,119:.8}
        result=diagnostics([1.]*40,scores,{49:1,99:3,119:3})
        self.assertEqual(result['reasons'],['one_output_coefficient'])
        self.assertTrue(diagnostics([1.]*20+[.99]*20,scores,{k:3 for k in scores})['investigation_required'])
        self.assertFalse(diagnostics([1.]*40,scores,{k:3 for k in scores})['investigation_required'])

    def test_missing_and_nonfinite_losses_refused(self):
        for losses in ([1.]*39,[float('nan')]*40):
            with self.assertRaises(ValueError):diagnostics(losses,{49:.8,99:.8},{49:3,99:3})


if __name__=='__main__':unittest.main()
