import json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_prediction_evidence import save,read
from shaping_completion import encoded,sha


class PredictionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.root=Path(self.temp.name)
        self.cell=dict(id='cell',stage='outer',family='relu')
        self.report=dict(cell=self.cell,plan_sha256='plan',admission_sha256='admission',outer_results_read=True,outer_test={})
        save(self.root,self.report,np.array([1,3,5]),np.array([90.,80.,70.]))
        self.commit_report()
    def tearDown(self):self.temp.cleanup()
    def commit_report(self):
        (self.root/'REPORT.json').write_bytes(encoded(self.report))
        (self.root/'COMPLETE.json').write_bytes(encoded(dict(cell='cell',report_sha256=sha(self.root/'REPORT.json'))))
    def read(self):return read(self.root,self.cell,'plan','admission',[1,3,5])
    def test_bound_table_roundtrip(self):
        report,values,pins=self.read();np.testing.assert_array_equal(values,[90.,80.,70.]);self.assertEqual(len(pins),3)
    def test_changed_table_and_unpinned_legacy_report_are_refused(self):
        save(self.root,{},np.array([1,3,5]),np.array([89.,80.,70.]))
        with self.assertRaisesRegex(ValueError,'not bound'):self.read()
        self.report.pop('predictions_sha256');self.commit_report()
        with self.assertRaisesRegex(ValueError,'not bound'):self.read()
    def test_report_scope_and_admission_drift_are_refused(self):
        for key,value in [('outer_results_read',False),('admission_sha256','wrong'),('plan_sha256','wrong'),('exposed_development',{})]:
            original=self.report.copy();self.report[key]=value;self.commit_report()
            with self.assertRaisesRegex(ValueError,'scope'):self.read()
            self.report=original
    def test_wrong_membership_and_nonfinite_predictions_are_refused(self):
        for ids,values in [([1,3,4],[90.,80.,70.]),([1,3,5],[90.,float('nan'),70.])]:
            save(self.root,self.report,np.array(ids),np.array(values));self.commit_report()
            with self.assertRaisesRegex(ValueError,'coverage'):self.read()
    def test_report_mutation_is_not_a_completed_evaluation(self):
        (self.root/'REPORT.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'identity'):self.read()


if __name__=='__main__':unittest.main()
