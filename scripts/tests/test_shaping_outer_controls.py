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

class OuterControlWatchTests(unittest.TestCase):
    def fixture(self,root):
        binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
        (root/'PLAN.json').write_text('{}')
        args=SimpleNamespace(study=root,binary_dir=binary,partition='outer-0',attempt=1,
            tree=True,cell=None,full_fit_selection=False,full_fit_refit=False)
        admission=root/'selection/outer-0/OUTER_ADMISSION.json'
        admission.parent.mkdir(parents=True)
        chosen=[c for c in matrix() if c['stage']=='outer' and c['partition']=='outer-0' and c['family']=='extra_trees']
        return args,admission,chosen

    def test_waits_without_reading_responses_then_runs_exact_queue_once(self):
        from shaping_controls import watch_outer
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);args,admission,chosen=self.fixture(root)
            def admit(seconds):
                self.assertEqual(seconds,30);fit.assert_not_called();load.assert_not_called()
                admission.write_text('{}')
            with patch('shaping_controls.time.sleep',side_effect=admit),patch('shaping_controls.run') as fit,\
                 patch('shaping_controls.load_registration',return_value=({},matrix(),{})) as load,\
                 patch('shaping_controls.selected_cells',return_value=(chosen,dict(admission_sha256='gate',selected_setting=2))):
                watch_outer(args);fit.assert_called_once_with(args)
                with self.assertRaisesRegex(ValueError,'reconciliation'):watch_outer(args)
                self.assertEqual(fit.call_count,1)
            status=json.loads((root/'dispatch/outer-controls/outer-0/extra_trees/STATUS.json').read_text())
            self.assertEqual(status['stage'],'complete');self.assertEqual(status['required_cells'],chosen)

    def test_changed_input_or_invalid_admission_never_launches(self):
        from shaping_controls import watch_outer
        for drift in (True,False):
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);args,admission,chosen=self.fixture(root)
                def admit(_):
                    admission.write_text('{}')
                    if drift:(root/'PLAN.json').write_text('{"changed":true}')
                with patch('shaping_controls.time.sleep',side_effect=admit),patch('shaping_controls.run') as fit,\
                     patch('shaping_controls.load_registration',return_value=({},matrix(),{})),\
                     patch('shaping_controls.selected_cells',side_effect=ValueError('incomplete scientific evidence')):
                    with self.assertRaisesRegex(ValueError,'changed|incomplete'):watch_outer(args)
                    fit.assert_not_called()
                status=json.loads((root/'dispatch/outer-controls/outer-0/extra_trees/STATUS.json').read_text())
                self.assertEqual(status['stage'],'failed')

    def test_existing_attempt_is_preserved_before_training(self):
        from shaping_controls import watch_outer
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);args,admission,chosen=self.fixture(root);admission.write_text('{}')
            prior=root/'runs'/chosen[0]['id'];prior.mkdir(parents=True)
            (prior/'evidence').write_text('preserve')
            with patch('shaping_controls.run') as fit,patch('shaping_controls.load_registration',return_value=({},matrix(),{})),\
                 patch('shaping_controls.selected_cells',return_value=(chosen,{})):
                with self.assertRaisesRegex(ValueError,'attempts require reconciliation'):watch_outer(args)
                fit.assert_not_called()
            self.assertEqual((prior/'evidence').read_text(),'preserve')
