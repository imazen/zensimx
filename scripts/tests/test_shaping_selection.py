import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_selection import HUMANS,aggregate,human_score,choose,INELIGIBILITY_POLICY
from shaping_completion import matrix


class SelectionTests(unittest.TestCase):
    def test_teacher_never_changes_human_selection(self):
        report={'outer_results_read':False,'inner_validation':{'summary':{p:{'pooled':.8,'within':.6} for p in HUMANS},'selection_score':.7}}
        report['inner_validation']['summary']['cid22_ssim2']={'pooled':-1.,'within':-1.}
        a=human_score(report)
        report['inner_validation']['summary']['cid22_ssim2']={'pooled':1.,'within':1.}
        self.assertEqual(a,human_score(report))
        report['inner_validation']['selection_score']=.8
        with self.assertRaises(ValueError):human_score(report)

    def test_missing_seed_and_outer_exposure_refuse_selection(self):
        cells=matrix();reports={}
        for c in cells:
            if c['stage']=='inner' and c['family']=='extra_trees' and c['partition'].startswith('inner-0-'):
                reports[c['id']]={'cell':c,'outer_results_read':False,'inner_validation':{'summary':{p:{'pooled':.8,'within':.8} for p in HUMANS},'selection_score':.8}}
        result=aggregate(cells,reports,'extra_trees',0)
        self.assertEqual(result['required_reports'],24);self.assertEqual(result['selected_setting'],0)
        missing=copy.deepcopy(reports);missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):aggregate(cells,missing,'extra_trees',0)
        mixed=copy.deepcopy(reports)
        next(iter(mixed.values())).update(reexport_receipt='receipt',evaluation_build_sha256='other')
        with self.assertRaises(ValueError):aggregate(cells,mixed,'extra_trees',0)
        next(iter(reports.values()))['outer_results_read']=True
        with self.assertRaises(ValueError):aggregate(cells,reports,'extra_trees',0)

    def test_softplus_requires_uniform_repaired_paired_reports(self):
        cells=matrix();reports={}
        for c in cells:
            if c['stage']=='inner' and c['family']=='softplus' and c['partition'].startswith('inner-0-'):
                reports[c['id']]={'cell':c,'outer_results_read':False,'reexport_receipt':'receipt','evaluation_build_sha256':'runtime','inner_validation':{'summary':{p:{'pooled':.8,'within':.8} for p in HUMANS},'selection_score':.8}}
        self.assertEqual(aggregate(cells,reports,'softplus',0)['required_reports'],24)
        first=next(iter(reports.values()));first['evaluation_build_sha256']='different'
        with self.assertRaisesRegex(ValueError,'mixed evaluation'):aggregate(cells,reports,'softplus',0)
        first['evaluation_build_sha256']='runtime';del first['reexport_receipt']
        with self.assertRaisesRegex(ValueError,'mixed evaluation'):aggregate(cells,reports,'softplus',0)

    def test_undefined_fit_excludes_entire_setting_without_dropping_pairs(self):
        cells=matrix();reports={}
        for c in cells:
            if c['stage']=='inner' and c['family']=='extra_trees' and c['partition'].startswith('inner-0-'):
                score=.9 if c['setting']==0 else .8
                reports[c['id']]={'cell':c,'outer_results_read':False,'inner_validation':{'summary':{p:{'pooled':score,'within':score} for p in HUMANS},'selection_score':score}}
        first=next(r for r in reports.values() if r['cell']['setting']==0)
        first['inner_validation']['summary']['kadid']['pooled']=None
        with self.assertRaisesRegex(ValueError,'undefined human'):aggregate(cells,reports,'extra_trees',0)
        result=aggregate(cells,reports,'extra_trees',0,INELIGIBILITY_POLICY)
        self.assertEqual(result['selected_setting'],1)
        self.assertIsNone(result['setting_scores'][0])
        self.assertEqual(result['required_reports'],24)
        self.assertEqual(len(result['reported_cells']),24)
        self.assertEqual(result['ineligible_settings']['0']['undefined_cells'],[first['cell']['id']])
        first['outer_results_read']=True
        with self.assertRaisesRegex(ValueError,'outer-exposed'):aggregate(cells,reports,'extra_trees',0,INELIGIBILITY_POLICY)
        first['outer_results_read']=False
        for r in reports.values():r['inner_validation']['summary']['kadid']['pooled']=None
        with self.assertRaisesRegex(ValueError,'no fully defined'):aggregate(cells,reports,'extra_trees',0,INELIGIBILITY_POLICY)

    def test_registered_tie_rule(self):
        self.assertEqual(choose({0:.8,1:.8+5e-13,2:.7,3:.6}),0)
        self.assertEqual(choose({0:.8,1:.8+2e-12,2:.7,3:.6}),1)
        self.assertEqual(choose({0:.8,1:.8+5e-13,2:.8+1.2e-12,3:.6}),1)


if __name__=='__main__':unittest.main()
