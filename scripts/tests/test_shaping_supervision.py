from pathlib import Path
import sys,tempfile,json,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_supervision import sweep
from shaping_evaluation import NativeFeatureGateError
from shaping_completion import encoded

class SupervisionTests(unittest.TestCase):
    def test_rescue_uses_explicit_optimizer_binary_set(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cell=dict(id='rescue',stage='rescue',family='softplus')
            run=root/'runs/rescue/attempt-001';run.mkdir(parents=True)
            (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',epochs=240)))
            args=SimpleNamespace(study=root,binary_dir=root/'primary',rescue_binary_dir=root/'optimizer')
            with patch('shaping_supervision.load_registration',return_value=({},[cell],{})),patch('shaping_supervision.evaluate') as evaluate,patch('shaping_supervision.report'):
                self.assertEqual(sweep(args),1)
                self.assertTrue(all(c.args[0].binary_dir==root/'optimizer' for c in evaluate.call_args_list))
                args.rescue_binary_dir=None
                with self.assertRaisesRegex(ValueError,'explicit pinned'):sweep(args)

    def test_supplemental_watch_keeps_original_failure_and_isolates_new_failure(self):
        from shaping_reexport_supervision import sweep as supplemental
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells=[dict(id=str(i),stage='inner',family='relu') for i in range(2)]
            for cell in cells:
                run=root/'runs'/cell['id']/'attempt-001';run.mkdir(parents=True)
                (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',epochs=240)))
                (run/'EVALUATION_FAILURES.json').write_text('original evidence')
            seen=[]
            def export(args):
                seen.append((args.run.parent.name,args.checkpoint))
                dest=root/'reexport_max_pairs'/args.run.parent.name/args.run.name
                dest.mkdir(parents=True,exist_ok=True)
                if args.run.parent.name=='0' and args.checkpoint==119:
                    dest=dest/'epoch-119';dest.mkdir()
                    (dest/'FAILED_GATE.json').write_text('{}')
                    raise NativeFeatureGateError('unchanged gate')
            with patch('shaping_reexport_supervision.load_registration',return_value=({},cells,{})),patch('shaping_reexport_supervision.reexport',side_effect=export),patch('shaping_reexport_supervision.report') as report:
                self.assertEqual(supplemental(SimpleNamespace(study=root,binary_dir=root/'bin',inspector=root/'inspect')),1)
                report.assert_called_once_with(root/'runs/1/attempt-001',root/'reexport_max_pairs/1/attempt-001')
            self.assertIn(('1',239),seen)
            self.assertEqual((root/'runs/0/attempt-001/EVALUATION_FAILURES.json').read_text(),'original evidence')

    def test_failed_gate_remains_failed_while_other_cells_finish(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells=[dict(id=str(i),stage='inner',family='relu') for i in range(2)]
            for cell in cells:
                run=root/'runs'/cell['id']/'attempt-001';run.mkdir(parents=True)
                (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',epochs=240)))
            seen=[]
            def evaluate(args):
                seen.append((args.run.parent.name,args.checkpoint))
                if args.run.parent.name=='0' and args.checkpoint==119:
                    dest=args.run/'curve_evaluation/epoch-119';dest.mkdir(parents=True)
                    (dest/'FAILED_GATE.json').write_bytes(encoded(dict(damage_increase_violations=1)))
                    raise NativeFeatureGateError('measured failure')
            with patch('shaping_supervision.load_registration',return_value=({},cells,{})),patch('shaping_supervision.evaluate',side_effect=evaluate),patch('shaping_supervision.report') as report:
                self.assertEqual(sweep(SimpleNamespace(study=root,binary_dir=root/'bin')),1)
                report.assert_called_once_with(root/'runs/1/attempt-001')
            self.assertIn(('0',239),seen);self.assertIn(('1',239),seen)
            failure=json.loads((root/'runs/0/attempt-001/EVALUATION_FAILURES.json').read_text())
            self.assertFalse(failure['scientifically_complete']);self.assertTrue(failure['requires_investigation'])
            self.assertEqual(failure['failures'][0]['checkpoint'],119)

if __name__=='__main__':unittest.main()

class LaterSupervisionTests(unittest.TestCase):
    def test_later_stage_scores_only_final_epoch_and_records_training_diagnostics(self):
        from shaping_supervision import later_sweep
        from shaping_completion import sha
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            cells=[dict(id='later',stage='full_fit_selection',family='best_analytic'),dict(id='inner',stage='inner',family='softplus')]
            for cell in cells:
                run=root/'runs'/cell['id']/'attempt-001';run.mkdir(parents=True)
                (run/'RUN.json').write_bytes(encoded(dict(cell=cell,build_sha256=sha(binary/'BUILD.json'))))
                (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',epochs=480)))
            context=dict(full_fit=True,resolved_family='softplus',admission={'evaluation_representations':{'softplus':{'representation':'original'}}})
            args=SimpleNamespace(study=root,binary_dir=binary)
            with patch('shaping_supervision.load_registration',return_value=({},cells,{})),patch('shaping_evaluation.outer_context',return_value=context),patch('shaping_supervision.evaluate') as evaluate,patch('shaping_convergence.save_training_report') as diagnose:
                self.assertEqual(later_sweep(args),1)
                evaluate.assert_called_once();self.assertIsNone(evaluate.call_args.args[0].checkpoint)
                self.assertEqual(evaluate.call_args.args[0].run.parent.name,'later')
                diagnose.assert_called_once()

    def test_later_gate_failure_is_preserved_and_does_not_stop_peer(self):
        from shaping_supervision import later_sweep
        from shaping_completion import sha
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            cells=[dict(id=str(i),stage='outer',family='softplus') for i in range(2)]
            for cell in cells:
                run=root/'runs'/cell['id']/'attempt-001';run.mkdir(parents=True)
                (run/'RUN.json').write_bytes(encoded(dict(cell=cell,build_sha256=sha(binary/'BUILD.json'))))
                (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',epochs=240)))
            context=dict(admission={'evaluation_representations':{'softplus':{'representation':'original'}}})
            def evaluate(args):
                if args.run.parent.name=='0':
                    dest=args.run/'evaluation';dest.mkdir();(dest/'FAILED_GATE.json').write_text('{}');raise NativeFeatureGateError('gate')
            with patch('shaping_supervision.load_registration',return_value=({},cells,{})),patch('shaping_evaluation.outer_context',return_value=context),patch('shaping_supervision.evaluate',side_effect=evaluate):
                self.assertEqual(later_sweep(SimpleNamespace(study=root,binary_dir=binary)),1)
                self.assertTrue((root/'runs/0/attempt-001/LATER_EVALUATION_FAILURE.json').exists())

class StageIsolationTests(unittest.TestCase):
    def test_inner_watchers_ignore_conditionally_registered_outer_cells(self):
        from shaping_reexport_supervision import sweep as supplemental
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);run=root/'runs/conditional/attempt-001';run.mkdir(parents=True)
            cell=dict(id='conditional',stage='fresh_control_outer_matched',family='relu')
            (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',epochs=240)))
            args=SimpleNamespace(study=root,binary_dir=root/'bin',inspector=root/'inspect')
            with patch('shaping_supervision.load_registration',return_value=({},[],{})),patch('shaping_supervision.evaluate') as evaluate:
                self.assertEqual(sweep(args),0);evaluate.assert_not_called()
            with patch('shaping_reexport_supervision.load_registration',return_value=({},[],{})),patch('shaping_reexport_supervision.reexport') as export:
                self.assertEqual(supplemental(args),0);export.assert_not_called()

class FailureEvidenceTests(unittest.TestCase):
    def test_repeat_observation_does_not_reopen_and_conflict_preserves_original(self):
        from shaping_supervision import preserve_failure
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path=Path(directory)/'FAILURE.json';record={'scientifically_complete':False,'gate':'original'}
            preserve_failure(path,record);original=path.read_bytes();before=path.stat()
            preserve_failure(path,record);after=path.stat()
            self.assertEqual((before.st_ino,before.st_mtime_ns),(after.st_ino,after.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'existing failure evidence differs'):
                preserve_failure(path,dict(record,gate='changed'))
            self.assertEqual(path.read_bytes(),original)
