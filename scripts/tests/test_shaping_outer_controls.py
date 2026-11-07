import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,sha
from shaping_controls import selected_cells,run


class OuterControlTests(unittest.TestCase):
    def test_outer_controls_use_inner_selected_setting_and_original_paired_cells(self):
        cells=matrix()
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);path=root/'admission';path.write_text('{}')
            for family,expected in [('extra_trees',{0,1,2}),('shaped_additive',{None})]:
                schedule=dict(setting=2,optimizer_choice='base',phase_epochs=[])
                admission=dict(outer_evaluation_admitted=True,schedules={family:schedule})
                with patch('shaping_outer_admission.resolve',return_value=(admission,path)):
                    chosen,context=selected_cells(root,'outer-0',family,cells)
                    self.assertEqual({c['seed'] for c in chosen},expected)
                    self.assertTrue(all(c in cells and c['setting']=='selected' for c in chosen))
                    self.assertEqual(context,dict(admission_sha256=sha(path),selected_setting=2))
                    with self.assertRaisesRegex(ValueError,'seed coverage'):
                        selected_cells(root,'outer-0',family,[c for c in cells if c!=chosen[0]])
                    schedule['phase_epochs']=[240]
                    with self.assertRaisesRegex(ValueError,'schedule differs'):selected_cells(root,'outer-0',family,cells)

    def test_missing_admission_refuses_before_fit_or_response_reads(self):
        args=SimpleNamespace(study=Path('/unused'),partition='outer-0',tree=True)
        with patch('shaping_controls.load_registration',return_value=({},matrix(),{})),\
             patch('shaping_outer_admission.resolve',side_effect=ValueError('incomplete inner evidence')),\
             patch('shaping_controls.fit') as fit:
            with self.assertRaisesRegex(ValueError,'incomplete inner'):run(args)
            fit.assert_not_called()

    def test_inner_search_keeps_all_four_settings(self):
        chosen,context=selected_cells(Path('/unused'),'inner-0-0','extra_trees',matrix())
        self.assertIsNone(context);self.assertEqual(len(chosen),12)
        self.assertEqual({c['setting'] for c in chosen},{0,1,2,3})


if __name__=='__main__':unittest.main()
