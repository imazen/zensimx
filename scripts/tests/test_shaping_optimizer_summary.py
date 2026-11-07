import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_optimizer_summary import summarize


class OptimizerSummaryTests(unittest.TestCase):
    def row(self):
        return dict(epoch=0,last_units=2,last_unit_active_counts=[0,4],output_amplitudes=[0.,-.2],
            output_zero_weights=1,sampled_responses=4,first_activation_values=8,first_zero_activations=3,
            mean_squared_first_activation_gradient=.1,mean_squared_regularized_output_gradient=.2,output_projection_hits=7)
    def text(self,row):return 'TRAINING_DIAGNOSTICS '+json.dumps(row)
    def test_sampled_collapse_can_coexist_with_nonzero_gradient(self):
        r=summarize(self.text(self.row()),1)
        self.assertEqual(r['epochs_with_at_most_one_output'],[0])
        self.assertEqual(r['final']['sampled_active_last_units'],1)
        self.assertEqual(r['final']['mean_squared_first_activation_gradient'],.1)
        self.assertTrue(r['sampled_activity_is_not_global_deadness'])
    def test_incomplete_duplicate_nonfinite_and_inconsistent_traces_refused(self):
        text=self.text(self.row())
        for value,epochs in [(text,2),(text+'\n'+text,2)]:
            with self.assertRaisesRegex(ValueError,'complete ordered'):summarize(value,epochs)
        for key,value in [('output_zero_weights',0),('last_unit_active_counts',[5,0]),('output_amplitudes',[float('nan'),.2])]:
            row=self.row();row[key]=value
            with self.assertRaises(ValueError):summarize(self.text(row),1)


if __name__=='__main__':unittest.main()
