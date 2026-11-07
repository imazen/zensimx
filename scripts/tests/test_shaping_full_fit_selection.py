import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,encoded,sha
from shaping_full_fit_selection import register,resolve,candidate
from shaping_selection import HUMANS

class FullFitSelectionTests(unittest.TestCase):
    def fixture(self,root):
        def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(encoded(d))
        write(root/'PLAN.json',{});cells=matrix();chosen=[c for c in cells if c['stage']=='full_fit_selection' and c['family']=='shaped_additive']
        admission=dict(selection_cells=chosen,finalist_roles={'shaped_additive':'shaped_additive'})
        path=root/'full_fit/PLAN.json';write(path,admission)
        for cell in chosen:
            directory=root/'runs'/cell['id']/'attempt-001'
            model=directory/'native/model.bin';model.parent.mkdir(parents=True);model.write_bytes(cell['id'].encode())
            run=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),resolved_family='shaped_additive',selection_scope='full_fit_only',development_results_read=False,admission_sha256=sha(path))
            write(directory/'RUN.json',run);write(directory/'RESULT.json',dict(run,stage='complete',exit_code=0))
            value=[.8,.81,.81,.79][cell['setting']]
            row=dict(run,qualification=dict(model_sha256=sha(model),identity=100,damage_increase_violations=0),full_fit_validation=dict(selection_score=value,summary={p:dict(pooled=value,within=value) for p in HUMANS}))
            report=directory/'evaluation/REPORT.json';write(report,row);write(report.parent/'COMPLETE.json',dict(report_sha256=sha(report)))
        return cells,admission,path,chosen

    def test_verified_search_reproduces_and_model_drift_invalidates_decision(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells,admission,path,chosen=self.fixture(root)
            with patch('shaping_full_fit_selection.load_registration',return_value=({'input_pins':{}},cells,{})),patch('shaping_full_fit_selection.resolve_admission',return_value=(admission,path)):
                record=register(root,'shaped_additive');self.assertEqual(record['selected_setting'],1)
                self.assertEqual(resolve(root,'shaped_additive')[0],record)
                self.assertEqual(len(record['reported_cells']),12)
                with self.assertRaises(FileExistsError):register(root,'shaped_additive')
                (root/'runs'/chosen[-1]['id']/'attempt-001/native/model.bin').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'evidence drift'):resolve(root,'shaped_additive')

    def test_missing_and_wrong_scope_reports_cannot_write_selection(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells,admission,path,chosen=self.fixture(root)
            cell=chosen[0];report=root/'runs'/cell['id']/'attempt-001/evaluation/REPORT.json'
            row=json.loads(report.read_text());row['selection_scope']='nested';report.write_bytes(encoded(row))
            (report.parent/'COMPLETE.json').write_bytes(encoded(dict(report_sha256=sha(report))))
            with self.assertRaisesRegex(ValueError,'scope'):candidate(root,cell,'shaped_additive',admission,path)
            (report.parent/'COMPLETE.json').unlink()
            with patch('shaping_full_fit_selection.load_registration',return_value=({'input_pins':{}},cells,{})),patch('shaping_full_fit_selection.resolve_admission',return_value=(admission,path)):
                with self.assertRaisesRegex(ValueError,'completed'):register(root,'shaped_additive')
                self.assertFalse((root/'full_fit/selection').exists())

    def test_native_selection_binds_final_checkpoint_and_training_diagnostics(self):
        from shaping_convergence import training_report
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(encoded(d))
            write(root/'PLAN.json',{})
            cell=next(c for c in matrix() if c['stage']=='full_fit_selection' and c['family']=='best_analytic')
            schedule=dict(phase_epochs=[240],optimizer_choice='base',total_pair_draws=12000000)
            admission=dict(schedules={'softplus':schedule},evaluation_representations={'softplus':dict(representation='original',evaluation_build_sha256=None)})
            path=root/'full_fit/PLAN.json';write(path,admission)
            run_dir=root/'runs'/cell['id']/'attempt-001';bake=run_dir/'fit/checkpoints/ckpt_epoch239.bin'
            bake.parent.mkdir(parents=True);bake.write_bytes(b'final model')
            phase=dict(epochs=240,stage='trained',exit_code=0,bake_sha256=sha(bake))
            write(run_dir/'fit/RESULT.json',phase)
            (run_dir/'fit/train.log').write_text(''.join(f'epoch {i} | loss=1.0000 | end\n' for i in range(240)))
            run=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),resolved_family='softplus',selection_scope='full_fit_only',development_results_read=False,
                admission_sha256=sha(path),schedule=dict(schedule,setting=cell['setting']),epochs=240,training_exposure_pair_draws=12000000,file_sha256={})
            write(run_dir/'RUN.json',run);write(run_dir/'RESULT.json',dict(run,stage='trained',exit_code=0,bake_sha256=sha(bake),phases=[phase]))
            report=run_dir/'evaluation/REPORT.json'
            write(report,dict(run,checkpoint_epoch=239,checkpoint_sha256=sha(bake),active_output_coefficients=4,
                feature_gate=dict(finite=True,identity=100,damage_increase_violations=0)))
            write(report.parent/'COMPLETE.json',dict(report_sha256=sha(report)))
            diagnostic=training_report(run_dir,report);write(run_dir/'TRAIN_CONVERGENCE.json',diagnostic)
            _,classification,pins=candidate(root,cell,'softplus',admission,path)
            self.assertEqual(classification['conclusion'],'inconclusive_training_only');self.assertIn(str(bake),pins)
            diagnostic['conclusion']='converged';write(run_dir/'TRAIN_CONVERGENCE.json',diagnostic)
            with self.assertRaisesRegex(ValueError,'cannot reproduce'):candidate(root,cell,'softplus',admission,path)

if __name__=='__main__':unittest.main()
