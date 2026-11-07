from pathlib import Path
import sys,unittest,json,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_rescue import controls,selected_base
from shaping_completion import matrix,encoded,sha
from shaping_selection import HUMANS,aggregate

class RescueTests(unittest.TestCase):
    def test_selection_rechecks_full_pairing_and_report_hashes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}');cells=matrix();reports={};pins={}
            for c in cells:
                if c['stage']!='inner' or c['family']!='relu' or not c['partition'].startswith('inner-0-'):continue
                run=root/'runs'/c['id'];run.mkdir(parents=True)
                score=.7+.01*c['setting']
                report=dict(cell=c,outer_results_read=False,inner_validation=dict(summary={p:dict(pooled=score,within=score) for p in HUMANS},selection_score=score))
                path=run/'REPORT.json';path.write_bytes(encoded(report));reports[c['id']]=report;pins[str(path)]=sha(path)
                conv=run/'CONVERGENCE.json';conv.write_bytes(encoded(dict(cell=c,outer_results_read=False,report_sha256={str(path):sha(path)})));pins[str(conv)]=sha(conv)
            record=aggregate(cells,reports,'relu',0);record.update(plan_sha256=sha(root/'PLAN.json'),evidence_sha256=pins)
            dest=root/'selection/outer-0';dest.mkdir(parents=True);path=dest/'relu-base.json';path.write_bytes(encoded(record))
            self.assertEqual(selected_base(root,'relu',0,cells)[0],3)
            report_path=next(Path(p) for p in pins if Path(p).name=='REPORT.json')
            report_path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'evidence drift'):selected_base(root,'relu',0,cells)
            record['evidence_sha256'].pop(str(report_path));path.write_bytes(encoded(record))
            with self.assertRaisesRegex(ValueError,'missing or extra paired'):selected_base(root,'relu',0,cells)

    def test_controls_preserve_objective_and_sampling(self):
        for mode,scale,schedule in [('init_half',.5,'cosine'),('init_double',2.,'cosine'),('constant_lr',1.,'constant'),('consolidate_lr',1.,'consolidate')]:
            a=controls(mode,True,groups=[dict(name='cid22',families=[0,0])]);b=controls(mode)
            self.assertEqual(a['init_scale'],scale);self.assertEqual(a['bias_scale'],scale)
            self.assertEqual(a['schedule'],schedule);self.assertEqual(b['schedule'],schedule)
            self.assertEqual(b['init_scale'],scale);self.assertEqual(a['teacher_mass'],1.)
            self.assertEqual(a['order'],'random');self.assertEqual(a['dropout'],0.);self.assertEqual(a['adamw'],0.)
        with self.assertRaises(ValueError):controls('extra_search')
        with self.assertRaisesRegex(ValueError,'exact admitted'):controls('constant_lr',True)

if __name__=='__main__':unittest.main()
