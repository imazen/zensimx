import json,sys,tempfile,unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,sha
from shaping_controls import selected_cells,run


class OuterControlTests(unittest.TestCase):
    def test_variant_control_receives_reduced_membership_and_selected_setting(self):
        cell=next(c for c in matrix() if c['stage']=='human_only' and c['family']=='extra_trees')
        membership=dict(train=[1,3],test=[9])
        variant=dict(cell=cell,resolved_family='extra_trees',schedule={'setting':2},
                     admission_sha256='pinned',groups_path='/prepared/variant-human/GROUPS.json')
        args=SimpleNamespace(study=Path('/unused'),partition=cell['partition'],tree=True,cell=cell['id'],attempt=1)
        with patch('shaping_controls.load_registration',return_value=({'minimum_ssd_free_bytes':30*1024**3},matrix(),{})),\
             patch('shaping_variant_execution.resolve',return_value=variant),\
             patch('shaping_variant_inputs.load_variant',return_value=({},cell,'view',membership)),\
             patch('shaping_storage.reservation',return_value=nullcontext()) as reserve,\
             patch('shaping_controls.fit') as fit:
            run(args)
            supplied=fit.call_args.args
            self.assertEqual(supplied[2],{cell['partition']:membership})
            self.assertEqual(supplied[4],[cell]);self.assertEqual(supplied[5]['selected_setting'],2)
            self.assertIn('variant-human',str(reserve.call_args.args[1]))
            fit.reset_mock();args.tree=False
            with self.assertRaisesRegex(ValueError,'family or partition'):run(args)
            fit.assert_not_called()

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

class FullFitControlTests(unittest.TestCase):
    def test_full_fit_controls_retain_all_settings_and_isolate_caches(self):
        from shaping_controls import full_fit_cells,control_cache_key
        cells=matrix()
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);path=root/'admission';path.write_text('{}')
            for family,count in [('extra_trees',12),('shaped_additive',4)]:
                admission=dict(selection_cells=[c for c in cells if c['stage']=='full_fit_selection'],
                    finalist_roles={family:family},schedules={family:dict(optimizer_choice='base',phase_epochs=[])})
                with patch('shaping_full_fit_admission.resolve',return_value=(admission,path)):
                    chosen,context=full_fit_cells(root,'outer-1',family,cells)
                    self.assertEqual(len(chosen),count);self.assertEqual({c['setting'] for c in chosen},{0,1,2,3})
                    self.assertEqual(context['selection_scope'],'full_fit_only')
                    self.assertNotEqual(control_cache_key('outer-1',context),control_cache_key('outer-1',{}))
                    args=SimpleNamespace(study=root,partition='outer-1',tree=family=='extra_trees',full_fit_selection=True,attempt=1)
                    with patch('shaping_controls.load_registration',return_value=({'minimum_ssd_free_bytes':0},cells,{})),patch('shaping_storage.reservation',return_value=nullcontext()),patch('shaping_controls.fit') as fit:
                        run(args);self.assertEqual(fit.call_args.args[4],chosen)
                    admission['selection_cells'].remove(chosen[0])
                    with self.assertRaisesRegex(ValueError,'coverage'):full_fit_cells(root,'outer-1',family,cells)

class FinalControlTests(unittest.TestCase):
    def test_final_controls_use_verified_settings_and_keep_dev_out_of_training(self):
        from shaping_controls import final_refit_cells,control_cache_key
        cells=matrix()
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);path=root/'admission';path.write_text('{}')
            partition=dict(train=[1,2],test=[],descriptive_development=[3])
            for family in ['extra_trees','shaped_additive']:
                cell=next(c for c in cells if c['stage']=='full_fit' and c['family']==family)
                resolved=({},cells,{'full-fit':partition},{'full_fit_selection_sha256':{'selection':'pin'}},path,cell,family,dict(setting=3,optimizer_choice='base',phase_epochs=[]))
                with patch('shaping_full_fit_execution.refit_cell',return_value=resolved):
                    chosen,context=final_refit_cells(root,'full-fit',family,cells)
                    self.assertEqual(context['selected_setting'],3);self.assertTrue(context['descriptive'])
                    self.assertTrue(control_cache_key('full-fit',context).startswith('full-fit-refit-'))
                    args=SimpleNamespace(study=root,partition='full-fit',tree=family=='extra_trees',full_fit_refit=True,attempt=1)
                    with patch('shaping_controls.load_registration',return_value=({'minimum_ssd_free_bytes':0},cells,{'full-fit':partition})),patch('shaping_storage.reservation',return_value=nullcontext()),patch('shaping_controls.fit') as fit:
                        run(args)
                        membership=fit.call_args.args[2]['full-fit']
                        self.assertEqual(membership['train'],[1,2]);self.assertEqual(membership['test'],[3])
                        self.assertEqual(partition['test'],[])
                        self.assertEqual(fit.call_args.args[5]['selected_setting'],3)
