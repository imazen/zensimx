import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import FAMILIES,NEURAL,matrix,encoded,sha
from shaping_full_fit_admission import register,resolve


class FullFitAdmissionTests(unittest.TestCase):
    def fixtures(self,root):
        (root/'PLAN.json').write_text('{}');admissions={}
        for outer in range(3):
            directory=root/'selection'/f'outer-{outer}';directory.mkdir(parents=True)
            pins={};schedules={}
            for family in FAMILIES:
                base=directory/(family+'-base.json');base.write_bytes(encoded({'setting_scores':{str(i):.8+i*.01 for i in range(4)}}))
                pins[str(base)]=sha(base)
                phases=([120,120] if '_pretrained_' in family else [240]) if family in NEURAL else []
                schedules[family]=dict(setting=1,optimizer_choice='base',phase_epochs=phases)
            path=directory/'OUTER_ADMISSION.json';path.write_text('{}')
            admissions[outer]=(dict(decision_sha256=pins,schedules=schedules,
                evaluation_representations={f:dict(representation='original',evaluation_build_sha256=None) for f in NEURAL}),path)
        return admissions

    def test_complete_three_fold_evidence_binds_receipt_and_refuses_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);admissions=self.fixtures(root)
            with patch('shaping_full_fit_admission.load_registration',return_value=({},matrix(),{})),\
                 patch('shaping_full_fit_admission.resolve_outer',side_effect=lambda root,outer:admissions[outer]):
                record=register(root);saved,path=resolve(root);self.assertEqual(saved,record)
                self.assertEqual(record['schedules']['relu']['optimizer_choice'],'base')
                self.assertFalse(record['development_results_read'])
                with self.assertRaises(FileExistsError):register(root)
                base=root/'selection/outer-1/relu-base.json';base.write_text('{}')
                with self.assertRaisesRegex(ValueError,'not admitted'):resolve(root)

    def test_missing_admission_creates_no_full_fit_plan(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            with patch('shaping_full_fit_admission.load_registration',return_value=({},matrix(),{})),\
                 patch('shaping_full_fit_admission.resolve_outer',side_effect=ValueError('unfinished paired rescues')):
                with self.assertRaisesRegex(ValueError,'unfinished'):register(root)
            self.assertFalse((root/'full_fit').exists())


if __name__=='__main__':unittest.main()
