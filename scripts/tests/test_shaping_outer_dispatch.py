from pathlib import Path
import copy,json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,FAMILIES,encoded,sha
from shaping_outer_plan import plan,finalist_roles
from shaping_outer_execution import resolve_cell
from shaping_outer_dispatch import selected_cells,existing,dispatch


class OuterDispatchTests(unittest.TestCase):
    def test_reduced_outer_and_variant_cells_keep_all_paired_seeds(self):
        from shaping_outer_dispatch import variant_cells
        decisions={f:dict(family=f,outer=0,outer_results_read=False,setting=0,
                         optimizer_choice='base',inner_score=.8) for f in FAMILIES}
        roles=finalist_roles(decisions)
        admission=plan(decisions,[],0,self.cells,active_families=roles.values(),frozen_roles=roles)
        admission.update(outer_evaluation_admitted=True,reduced_scope_sha256='approved')
        selected=selected_cells(self.cells,admission)
        self.assertEqual(sum(c['stage']=='outer' for c in selected),12)
        self.assertEqual(sum(c['stage']=='fresh_control_outer' for c in selected),3)
        neural=variant_cells(self.cells,admission)
        controls=variant_cells(self.cells,admission,True)
        self.assertEqual((len(neural),len(controls)),(48,16))
        self.assertEqual({c['stage'] for c in neural+controls},{'source_curve','shuffle','human_only'})
        self.assertEqual({c['fraction'] for c in neural+controls if c['stage']=='source_curve'},{.25,.5})
        self.assertEqual({c['order_seed'] for c in neural+controls if c['stage']=='source_curve'},{20261107})
        self.assertEqual({c['permutation'] for c in neural+controls if c['stage']=='shuffle'},{'population'})
        self.assertEqual({c['permutation_seed'] for c in neural+controls if c['stage']=='shuffle'},{20261107})
        with self.assertRaisesRegex(ValueError,'complete registered variant'):
            variant_cells([c for c in self.cells if c['id']!=neural[0]['id']],admission)

    def setUp(self):
        self.cells=matrix()
        decisions={f:dict(family=f,outer=0,outer_results_read=False,setting=0,optimizer_choice='base',inner_score=.8) for f in FAMILIES}
        self.admission=plan(decisions,['relu','product_pretrained_joint'],0,self.cells)
        self.admission['outer_evaluation_admitted']=True

    def test_full_pairing_and_exact_extra_fresh_controls(self):
        selected=selected_cells(self.cells,self.admission)
        self.assertEqual(sum(c['stage']=='outer' for c in selected),30)
        self.assertEqual([c for c in selected if c['stage']!='outer'],
                         sorted([v['cell'] for v in self.admission['fresh_controls']],key=lambda c:(c['seed'],c['family'],c['id'])))
        incomplete=[c for c in self.cells if c['id']!=selected[0]['id']]
        with self.assertRaises(ValueError):selected_cells(incomplete,self.admission)
        duplicate=copy.deepcopy(self.admission);duplicate['fresh_controls'].append(duplicate['fresh_controls'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):selected_cells(self.cells,duplicate)
        self.admission['outer_evaluation_admitted']=False
        with self.assertRaisesRegex(ValueError,'not admitted'):selected_cells(self.cells,self.admission)

    def fixture(self,root):
        (root/'PLAN.json').write_text('{}')
        cell=next(c for c in selected_cells(self.cells,self.admission) if c['stage']=='outer')
        run=root/'runs'/cell['id']/'attempt-001';(run/'fit/checkpoints').mkdir(parents=True)
        schedule=resolve_cell(cell,self.admission)
        identity=dict(cell=cell,schedule=schedule,admission_sha256='admitted',build_sha256='build',
                      plan_sha256=sha(root/'PLAN.json'),epochs=schedule['phase_epochs'][-1],process_role='outer_controller')
        checkpoint=run/'fit/checkpoints'/f"ckpt_epoch{identity['epochs']-1:03}.bin";checkpoint.write_bytes(b'final scheduled model')
        (run/'RUN.json').write_bytes(encoded(identity))
        result=dict(identity,stage='trained',exit_code=0,bake_sha256=sha(checkpoint));(run/'RESULT.json').write_bytes(encoded(result))
        return cell,run,checkpoint,result

    def test_terminal_resume_requires_exact_model_and_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cell,run,checkpoint,result=self.fixture(root)
            self.assertEqual(existing(root,cell,self.admission,'admitted','build'),'trained')
            with self.assertRaisesRegex(ValueError,'identity differs'):existing(root,cell,self.admission,'other','build')
            checkpoint.write_bytes(b'changed model')
            with self.assertRaisesRegex(ValueError,'checkpoint drift'):existing(root,cell,self.admission,'admitted','build')

    def test_failed_and_ambiguous_attempts_are_preserved_and_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cell,run,checkpoint,result=self.fixture(root)
            result.update(stage='failed',exit_code=1);(run/'RESULT.json').write_bytes(encoded(result))
            before=(run/'RESULT.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'fit failed'):existing(root,cell,self.admission,'admitted','build')
            self.assertEqual((run/'RESULT.json').read_bytes(),before)
            (run.parent/'attempt-002').mkdir()
            with self.assertRaisesRegex(ValueError,'explicit reconciliation'):existing(root,cell,self.admission,'admitted','build')

    def test_incomplete_admission_cannot_prepare_or_launch(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            args=SimpleNamespace(study=Path(directory),outer=0,workers=2,session='first',watch_admission=False)
            with patch('shaping_outer_dispatch.prepare') as prepare,patch('shaping_outer_dispatch.subprocess.Popen') as launch:
                with self.assertRaisesRegex(ValueError,'not yet complete'):dispatch(args)
                prepare.assert_not_called();launch.assert_not_called()

    def test_queue_runs_each_registered_cell_once_with_two_worker_limit(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}');binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            admission_path=root/'selection/outer-0/OUTER_ADMISSION.json';admission_path.parent.mkdir(parents=True);admission_path.write_bytes(encoded(self.admission))
            cells={c['id']:c for c in selected_cells(self.cells,self.admission)};started=[];live=0;peak=0
            admission=self.admission
            class Process:
                def __init__(process,command,**kwargs):
                    nonlocal live,peak
                    cell_id=command[command.index('--cell')+1];started.append(cell_id)
                    process.cell=cells[cell_id];process.pid=1000+len(started);process.returncode=None
                    live+=1;peak=max(peak,live)
                def poll(process):
                    nonlocal live
                    if process.returncode is not None:return process.returncode
                    cell=process.cell;schedule=resolve_cell(cell,admission)
                    run=root/'runs'/cell['id']/'attempt-001';(run/'fit/checkpoints').mkdir(parents=True)
                    identity=dict(cell=cell,schedule=schedule,admission_sha256=sha(admission_path),build_sha256=sha(binary/'BUILD.json'),
                                  plan_sha256=sha(root/'PLAN.json'),epochs=schedule['phase_epochs'][-1],process_role='outer_controller')
                    checkpoint=run/'fit/checkpoints'/f"ckpt_epoch{identity['epochs']-1:03}.bin";checkpoint.write_bytes(b'queue test final model')
                    (run/'RUN.json').write_bytes(encoded(identity));(run/'RESULT.json').write_bytes(encoded(dict(identity,stage='trained',exit_code=0,bake_sha256=sha(checkpoint))))
                    live-=1;process.returncode=0;return 0
            args=SimpleNamespace(study=root,outer=0,workers=2,session='first',watch_admission=False,binary_dir=binary,neutral_qualification=root/'neutral')
            with patch('shaping_outer_dispatch.resolve',return_value=(self.admission,admission_path)),patch('shaping_outer_dispatch.load_registration',return_value=({},self.cells,{})),patch('shaping_outer_dispatch.prepare'),patch('shaping_outer_dispatch.subprocess.Popen',side_effect=Process),patch('shaping_outer_dispatch.time.sleep'):
                dispatch(args)
            self.assertEqual(peak,2);self.assertEqual(live,0)
            self.assertEqual(len(started),len(cells));self.assertEqual(set(started),set(cells))
            status=json.loads((root/'dispatch/outer/outer-0/first/STATUS.json').read_text())
            self.assertEqual(status['stage'],'training_finished');self.assertEqual(set(status['completed']),set(cells))


if __name__=='__main__':unittest.main()

class VariantDispatchTests(OuterDispatchTests):
    def test_variant_matrix_is_complete_and_only_inner_resolved_neural_roles(self):
        from shaping_outer_dispatch import variant_cells
        selected=variant_cells(self.cells,self.admission)
        self.assertEqual(len(selected),168)
        self.assertEqual({c['stage'] for c in selected},{'human_only','source_curve','row_curve','shuffle'})
        self.assertEqual({c['seed'] for c in selected},{0,1,2})
        with self.assertRaisesRegex(ValueError,'complete registered'):
            variant_cells([c for c in self.cells if c!=selected[0]],self.admission)
        with self.assertRaisesRegex(ValueError,'complete registered'):
            variant_cells(self.cells+[selected[0]],self.admission)
        self.admission['finalist_roles']['best_product']='extra_trees'
        with self.assertRaisesRegex(ValueError,'nonneural'):variant_cells(self.cells,self.admission)

    def test_variant_resume_checks_resolution_model_and_failed_evidence(self):
        from shaping_outer_dispatch import variant_cells,existing_variant
        cell=variant_cells(self.cells,self.admission)[0]
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}')
            run=root/'runs'/cell['id']/'attempt-001';(run/'fit/checkpoints').mkdir(parents=True)
            resolution=dict(cell=cell,admission_sha256='gate',schedule=dict(phase_epochs=[240]))
            identity=dict(cell=cell,resolution=resolution,build_sha256='build',plan_sha256=sha(root/'PLAN.json'),
                          epochs=240,process_role='variant_controller')
            checkpoint=run/'fit/checkpoints/ckpt_epoch239.bin';checkpoint.write_bytes(b'final variant')
            (run/'RUN.json').write_bytes(encoded(identity));result=dict(identity,stage='trained',exit_code=0,bake_sha256=sha(checkpoint))
            (run/'RESULT.json').write_bytes(encoded(result))
            with patch('shaping_variant_execution.resolve',return_value=resolution):
                self.assertEqual(existing_variant(root,cell,self.admission,'gate','build'),'trained')
                with self.assertRaisesRegex(ValueError,'admission differs'):existing_variant(root,cell,self.admission,'other','build')
                result['stage']='failed';(run/'RESULT.json').write_bytes(encoded(result));before=(run/'RESULT.json').read_bytes()
                with self.assertRaisesRegex(ValueError,'fit failed'):existing_variant(root,cell,self.admission,'gate','build')
                self.assertEqual((run/'RESULT.json').read_bytes(),before)
                result['stage']='trained';(run/'RESULT.json').write_bytes(encoded(result));checkpoint.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError,'checkpoint drift'):existing_variant(root,cell,self.admission,'gate','build')

    def test_missing_variant_admission_never_prepares_or_launches(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            args=SimpleNamespace(study=Path(directory),outer=0,workers=2,session='first',watch_admission=False,
                                 variants=True,human_qualification=Path(directory)/'human')
            with patch('shaping_variant_execution.resolve') as resolve_variant,patch('shaping_outer_dispatch.subprocess.Popen') as launch:
                with self.assertRaisesRegex(ValueError,'not yet complete'):dispatch(args)
                resolve_variant.assert_not_called();launch.assert_not_called()

    def test_complete_variant_queue_uses_variant_owner_and_two_slots(self):
        from shaping_outer_dispatch import variant_cells
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}');binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            path=root/'selection/outer-0/OUTER_ADMISSION.json';path.parent.mkdir(parents=True);path.write_bytes(encoded(self.admission))
            chosen={c['id']:c for c in variant_cells(self.cells,self.admission)};started=[];live=0;peak=0
            def resolution(study,cell_id):
                return dict(cell=chosen[cell_id],admission_sha256=sha(path),schedule=dict(phase_epochs=[240]))
            class Process:
                def __init__(process,command,**kwargs):
                    nonlocal live,peak
                    self.assertIn('scripts/rev4_featpot/shaping_variant_execution.py',command)
                    self.assertIn('--human-qualification',command)
                    process.cell=chosen[command[command.index('--cell')+1]]
                    started.append(process.cell['id']);process.pid=1000+len(started);process.returncode=None
                    live+=1;peak=max(peak,live)
                def poll(process):
                    nonlocal live
                    if process.returncode is not None:return process.returncode
                    cell=process.cell;run=root/'runs'/cell['id']/'attempt-001';(run/'fit/checkpoints').mkdir(parents=True)
                    identity=dict(cell=cell,resolution=resolution(root,cell['id']),build_sha256=sha(binary/'BUILD.json'),
                        plan_sha256=sha(root/'PLAN.json'),epochs=240,process_role='variant_controller')
                    ckpt=run/'fit/checkpoints/ckpt_epoch239.bin';ckpt.write_bytes(b'variant final')
                    (run/'RUN.json').write_bytes(encoded(identity));(run/'RESULT.json').write_bytes(encoded(dict(identity,stage='trained',exit_code=0,bake_sha256=sha(ckpt))))
                    live-=1;process.returncode=0;return 0
            args=SimpleNamespace(study=root,outer=0,workers=2,session='first',watch_admission=False,
                binary_dir=binary,neutral_qualification=root/'neutral',variants=True,human_qualification=root/'human')
            with patch('shaping_outer_dispatch.resolve',return_value=(self.admission,path)),\
                 patch('shaping_outer_dispatch.load_registration',return_value=({},self.cells,{})),\
                 patch('shaping_variant_execution.resolve',side_effect=resolution),\
                 patch('shaping_outer_dispatch.prepare') as prepare,\
                 patch('shaping_outer_dispatch.subprocess.Popen',side_effect=Process),patch('shaping_outer_dispatch.time.sleep'):
                dispatch(args);prepare.assert_not_called()
            self.assertEqual(peak,2);self.assertEqual(live,0);self.assertEqual(len(started),168)
            self.assertEqual(set(started),set(chosen))
            status=json.loads((root/'dispatch/variants-neural/outer-0/first/STATUS.json').read_text())
            self.assertEqual(status['stage'],'training_finished')

class ControlVariantDispatchTests(unittest.TestCase):
    def fixture(self,root):
        cells=matrix();decisions={f:dict(family=f,outer=0,outer_results_read=False,setting=0,optimizer_choice='base',inner_score=.8) for f in FAMILIES}
        admission=plan(decisions,[],0,cells);admission['outer_evaluation_admitted']=True
        (root/'PLAN.json').write_text('{}');binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
        path=root/'selection/outer-0/OUTER_ADMISSION.json';path.parent.mkdir(parents=True);path.write_bytes(encoded(admission))
        return cells,admission,binary,path

    def test_full_control_queue_and_bound_evidence(self):
        from shaping_outer_dispatch import variant_cells,existing_variant
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells,admission,binary,path=self.fixture(root)
            selected=variant_cells(cells,admission,True);self.assertEqual(len(selected),56)
            self.assertEqual(sum(c['family']=='shaped_additive' for c in selected),14)
            with self.assertRaisesRegex(ValueError,'complete registered'):
                variant_cells([c for c in cells if c!=selected[0]],admission,True)
            byid={c['id']:c for c in selected};started=[];live=0;peak=0
            def resolution(study,cell_id):
                return dict(cell=byid[cell_id],resolved_family=byid[cell_id]['family'],admission_sha256=sha(path),schedule=dict(setting=2,phase_epochs=[]))
            class Process:
                def __init__(process,command,**kwargs):
                    nonlocal live,peak
                    self.assertIn('scripts/rev4_featpot/shaping_controls.py',command)
                    process.cell=byid[command[command.index('--cell')+1]]
                    self.assertIn('--tree' if process.cell['family']=='extra_trees' else '--linear',command)
                    started.append(process.cell['id']);process.pid=1000+len(started);process.returncode=None
                    live+=1;peak=max(peak,live)
                def poll(process):
                    nonlocal live
                    if process.returncode is not None:return process.returncode
                    cell=process.cell;run=root/'runs'/cell['id']/'attempt-001';evaluation=run/'evaluation';evaluation.mkdir(parents=True)
                    identity=dict(cell=cell,variant=resolution(root,cell['id']),build_sha256=sha(binary/'BUILD.json'),
                        plan_sha256=sha(root/'PLAN.json'),process_role='control_worker',selected_setting=2,admission_sha256=sha(path))
                    (run/'RUN.json').write_bytes(encoded(identity));(run/'RESULT.json').write_bytes(encoded(dict(identity,stage='complete',exit_code=0)))
                    (evaluation/'predictions.npz').write_bytes(b'bound predictions fixture')
                    report=dict(identity,outer_results_read=True,outer_test={},predictions_sha256=sha(evaluation/'predictions.npz'))
                    (evaluation/'REPORT.json').write_bytes(encoded(report));(evaluation/'COMPLETE.json').write_bytes(encoded(dict(cell=cell['id'],report_sha256=sha(evaluation/'REPORT.json'))))
                    live-=1;process.returncode=0;return 0
            args=SimpleNamespace(study=root,outer=0,workers=2,session='first',watch_admission=False,
                binary_dir=binary,neutral_qualification=root/'unused',variant_controls=True)
            with patch('shaping_outer_dispatch.resolve',return_value=(admission,path)),\
                 patch('shaping_outer_dispatch.load_registration',return_value=({},cells,{})),\
                 patch('shaping_variant_execution.resolve',side_effect=resolution),\
                 patch('shaping_outer_dispatch.prepare') as prepare,\
                 patch('shaping_outer_dispatch.subprocess.Popen',side_effect=Process),patch('shaping_outer_dispatch.time.sleep'):
                dispatch(args);prepare.assert_not_called()
                cell=selected[0];evaluation=root/'runs'/cell['id']/'attempt-001/evaluation'
                (evaluation/'predictions.npz').write_bytes(b'altered predictions')
                with self.assertRaisesRegex(ValueError,'completed evidence differs'):
                    existing_variant(root,cell,admission,sha(path),sha(binary/'BUILD.json'),True)
            self.assertEqual(peak,2);self.assertEqual(live,0);self.assertEqual(len(started),56)
            self.assertEqual(set(started),set(byid))
            status=json.loads((root/'dispatch/variants-controls/outer-0/first/STATUS.json').read_text())
            self.assertEqual(status['stage'],'training_finished')
