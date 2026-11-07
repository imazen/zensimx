import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import FAMILIES,NEURAL,matrix,encoded,sha
from shaping_outer_admission import build,register,resolve

class OuterAdmissionTests(unittest.TestCase):
    def base(self,root,family,outer,cells):
        expected=[c for c in cells if c['stage']=='inner' and c['family']==family and c['partition'].startswith(f'inner-{outer}-')]
        return dict(selected_setting=0,setting_scores={i:.8 for i in range(4)},reported_cells=[c['id'] for c in expected]),dict(representation='original'),{}

    def candidate(self,root,cell,representation):
        return {},dict(cell=cell,investigation_required=False,reasons=[],outer_results_read=False),{}

    def test_complete_evidence_admits_without_opening_outer_and_preserves_receipt(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}');(root/'selection/outer-0').mkdir(parents=True)
            with patch('shaping_outer_admission.load_registration',return_value=({},matrix(),{})), \
                 patch('shaping_outer_admission.base_decision',side_effect=self.base), \
                 patch('shaping_outer_admission.verified_candidate',side_effect=self.candidate):
                result=register(root,0)
                self.assertTrue(result['outer_evaluation_admitted']);self.assertFalse(result['outer_results_read'])
                self.assertEqual(set(result['schedules']),set(FAMILIES));self.assertEqual(len(result['fresh_controls']),3)
                self.assertTrue(all(v['status']=='registered_triggers_absent' for v in result['qualifications'].values()))
                saved,path=resolve(root,0);self.assertEqual(saved,result)
                with self.assertRaises(FileExistsError):register(root,0)
                saved['schedules']['relu']['setting']=3;path.write_bytes(encoded(saved))
                with self.assertRaisesRegex(ValueError,'cannot be reproduced'):resolve(root,0)

    def test_mixed_repaired_evaluators_cannot_admit(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}')
            def base(root,family,outer,cells):
                actual,record,pins=self.base(root,family,outer,cells)
                if family=='relu':record['representation']='research-relu-max-pairs-v1'
                return actual,record,pins
            def candidate(root,cell,representation):
                row,diagnostic,pins=self.candidate(root,cell,representation)
                if cell['family']=='relu':row.update(reexport_receipt='qualified',evaluation_build_sha256=str(cell['seed']))
                return row,diagnostic,pins
            with patch('shaping_outer_admission.load_registration',return_value=({},matrix(),{})), \
                 patch('shaping_outer_admission.base_decision',side_effect=base), \
                 patch('shaping_outer_admission.verified_candidate',side_effect=candidate):
                with self.assertRaisesRegex(ValueError,'mixed candidate evaluation'):build(root,0)

    def test_missing_pair_or_unregistered_investigation_never_admits(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}')
            def pending(*args):raise ValueError('missing paired report')
            with patch('shaping_outer_admission.load_registration',return_value=({},matrix(),{})), \
                 patch('shaping_outer_admission.base_decision',side_effect=self.base), \
                 patch('shaping_outer_admission.verified_candidate',side_effect=pending):
                with self.assertRaisesRegex(ValueError,'missing paired'):register(root,0)
            def improving(root,cell,representation):
                row,diagnostic,pins=self.candidate(root,cell,representation)
                diagnostic.update(investigation_required=True,reasons=['one_output_coefficient']);return row,diagnostic,pins
            with patch('shaping_outer_admission.load_registration',return_value=({},matrix(),{})), \
                 patch('shaping_outer_admission.base_decision',side_effect=self.base), \
                 patch('shaping_outer_admission.verified_candidate',side_effect=improving):
                with self.assertRaisesRegex(ValueError,'equal-opportunity rescues'):register(root,0)
            self.assertFalse(list(root.rglob('OUTER_ADMISSION.json')))

if __name__=='__main__':unittest.main()
