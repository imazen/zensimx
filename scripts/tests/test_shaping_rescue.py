from pathlib import Path
import sys,unittest,json,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_rescue import controls,selected_base,rewrite_command
from shaping_completion import matrix,encoded,sha
from shaping_selection import HUMANS,aggregate
from shaping_rescue_selection import aggregate as aggregate_rescues

class RescueTests(unittest.TestCase):
    def test_human_only_controls_require_explicit_teacher_exclusion(self):
        groups=[dict(name='human_kadid',families=[0,1])]
        with self.assertRaisesRegex(ValueError,'one teacher'):controls('constant_lr',True,groups=groups)
        result=controls('constant_lr',True,groups=groups,human_only=True)
        self.assertTrue(result['human_only']);self.assertEqual(result['groups'],groups)
        self.assertEqual(result['teacher_mass'],1.)
        with self.assertRaisesRegex(ValueError,'zero teachers'):
            controls('constant_lr',True,groups=groups+[dict(name='cid22',families=[0])],human_only=True)
        self.assertNotIn('human_only',controls('constant_lr',True,groups=[dict(name='cid22',families=[0])]))

    def test_rescue_selection_requires_all_thirty_paired_reports(self):
        cells=matrix();reports={}
        for cell in cells:
            if cell['family']!='relu' or cell['partition'] not in ('inner-0-0','inner-0-1'):continue
            if not (cell['stage']=='rescue' or (cell['stage']=='inner' and cell['setting']==2)):continue
            score=.81 if cell.get('rescue')=='constant_lr' else .8
            reports[cell['id']]=dict(cell=cell,outer_results_read=False,inner_validation=dict(
                summary={p:dict(pooled=score,within=score) for p in HUMANS},selection_score=score))
        result=aggregate_rescues(cells,reports,'relu',0,2)
        self.assertEqual(result['required_reports'],30);self.assertEqual(result['selected_choice'],'constant_lr')
        missing=reports.copy();missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):aggregate_rescues(cells,missing,'relu',0,2)
        for report in reports.values():
            score=.8+5e-13 if report['cell'].get('rescue')=='init_half' else .8
            report['inner_validation']=dict(summary={p:dict(pooled=score,within=score) for p in HUMANS},selection_score=score)
        self.assertEqual(aggregate_rescues(cells,reports,'relu',0,2)['selected_choice'],'base')
        next(iter(reports.values())).update(reexport_receipt='different',evaluation_build_sha256='runtime')
        with self.assertRaises(ValueError):aggregate_rescues(cells,reports,'relu',0,2)

    def test_phase_command_preserves_seed_objective_and_inputs(self):
        command=['run-heavy','--mem','1G','--jobs','1','--','/old/zensim_mlp_train',
                 '--epochs','240','--out','old.bin','--dump-checkpoints-dir','old-checkpoints',
                 '--init-seed','1101','--sample-seed','101','--l2','0.00001','--group','frozen:1:0:rank']
        got=rewrite_command(command,Path('/native'),Path('/outputs/phase'),120)
        for flag in ('--jobs','--init-seed','--sample-seed','--l2','--group'):
            self.assertEqual(got[got.index(flag)+1],command[command.index(flag)+1])
        self.assertEqual(got[got.index('--epochs')+1],'120')
        self.assertIn('/native/zensim_mlp_train',got)
        self.assertEqual(command[command.index('--epochs')+1],'240')
        for bad in (command+['--epochs','120'],command+['/another/zensim_mlp_train']):
            with self.assertRaises(ValueError):rewrite_command(bad,Path('/native'),Path('/output'),120)
        with self.assertRaises(ValueError):rewrite_command(command,Path('/native'),Path('/output'),119)

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
