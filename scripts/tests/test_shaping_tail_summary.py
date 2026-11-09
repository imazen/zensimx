import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_tail_summary import paired_summary,read_panel,POPULATIONS


class TailSummaryTests(unittest.TestCase):
    def test_incomplete_metrics_have_no_mean_and_zero_is_measured(self):
        result=paired_summary([(0.,0.),(.2,.1),(.1,.2)])
        self.assertEqual((result['lower'],result['equal'],result['higher']),(1,1,1))
        self.assertAlmostEqual(result['delta'],0.)
        result=paired_summary([(0.,0.),(None,.1)])
        self.assertEqual(result['finite'],1);self.assertIsNone(result['delta']);self.assertIsNone(result['shaped'])

    def test_native_identity_exposure_and_denominators_are_checked(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp:
            root=Path(temp);path=root/'details/native/panel.json';path.parent.mkdir(parents=True)
            cell=dict(id='cell');record=dict(cell=cell,details=dict(report_sha256='report'))
            index=dict(plan_sha256='plan',outer_results_read=False,
                cells=dict(cell=dict(cell=cell,report_sha256='report',native_url='details/native/panel.json')))
            labels=[p+'|'+r for p in POPULATIONS for r in ('train','validation')]
            value=dict(schema='zensim-native-detailed-panel-v1',cell=cell,report_sha256='report',
                outer_results_read=False,input_sha256='input',panel_sha256='binary',
                statistics=dict(full={k:{'n':60,'n_dropped':0,'or':0.} for k in labels},
                    scatter={k:dict(n=60,status='NOT_MEASURED') for k in labels}))
            path.write_text(json.dumps(value));metrics,_=read_panel(root,index,record,'plan')
            self.assertIsNone(metrics[('train','kadid')]['shape_p99'])
            self.assertEqual(metrics[('train','kadid')]['outlier_ratio'],0.)
            for mode in ('cell','outer','report','dropped','n'):
                changed=json.loads(json.dumps(value))
                if mode=='cell':changed['cell']={}
                if mode=='outer':changed['outer_results_read']=True
                if mode=='report':changed['report_sha256']='other'
                if mode=='dropped':changed['statistics']['full'][labels[0]]['n_dropped']=1
                if mode=='n':changed['statistics']['scatter'][labels[0]]['n']=59
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):read_panel(root,index,record,'plan')


if __name__=='__main__':unittest.main()
