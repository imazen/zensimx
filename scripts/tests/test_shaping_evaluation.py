import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_evaluation import admitted_epoch, evaluation_epoch


class EvaluationAdmissionTests(unittest.TestCase):
    def test_diagnostics_do_not_change_final_selection(self):
        self.assertEqual(evaluation_epoch(240,None),239)
        self.assertEqual(evaluation_epoch(120,None),119)
        self.assertEqual(evaluation_epoch(240,49),49)
        for budget,epoch in [(120,149),(240,0),(240,238),(240,240)]:
            with self.assertRaises(ValueError):evaluation_epoch(budget,epoch)

    def test_final_scheduled_epoch_is_required(self):
        cell={'id':'x','stage':'inner','family':'relu'}
        result={'stage':'trained','exit_code':0}
        self.assertEqual(admitted_epoch({'cell':cell,'epochs':240},result,[cell]),240)
        for epochs in (1,119,120,239):
            with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':epochs},result,[cell])

    def test_outer_pilot_unregistered_and_failed_runs_are_refused(self):
        for stage in ('outer','resource_pilot','full_fit'):
            cell={'id':'x','stage':stage,'family':'relu'}
            with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':240},{'stage':'trained','exit_code':0},[cell])
        cell={'id':'x','stage':'inner','family':'relu'}
        with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':240},{'stage':'trained','exit_code':0},[])
        for stage,code in [('running',0),('failed',1),('trained',1)]:
            with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':240},{'stage':stage,'exit_code':code},[cell])


if __name__=='__main__':unittest.main()
