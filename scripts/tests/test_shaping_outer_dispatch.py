from pathlib import Path
import copy,json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,FAMILIES,encoded,sha
from shaping_outer_plan import plan
from shaping_outer_execution import resolve_cell
from shaping_outer_dispatch import selected_cells,existing,dispatch


class OuterDispatchTests(unittest.TestCase):
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
