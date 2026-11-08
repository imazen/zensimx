from pathlib import Path
import json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_bootstrap import ready_cells,watch_sweep,complete_output


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))

class BootstrapWatchTests(unittest.TestCase):
    def test_reduced_queue_waits_for_admissions_and_refuses_old_exhaustive_registry(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);registry=root/'registry'
            entries=[dict(file=f'comparison-{i:03}.json',spec=dict(role='relu',condition=dict(stage='human_only'))) for i in range(24)]
            write(registry/'MATCHED_REGISTRATION.json',dict(comparisons=entries))
            args=SimpleNamespace(study=root/'study',matched_registry=registry,draws=root/'draws',binary_dir=root/'bin',output=root/'out',archive_destination=root/'archive')
            with patch('shaping_execution.load_registration',return_value=({},[],{})),patch('shaping_bootstrap_plan.load_comparison'),patch('shaping_reduced_scope.coverage',return_value={'registration_sha256':'approved'}),patch('shaping_bootstrap.subprocess.run') as execute,patch('builtins.print'):
                status=watch_sweep(args)
                self.assertEqual(status['required'],25);self.assertEqual(len(status['waiting']),25)
                execute.assert_not_called()
                write(registry/'MATCHED_REGISTRATION.json',dict(comparisons=entries+[dict(file=f'comparison-{i:03}.json',spec={}) for i in range(24,88)]))
                with self.assertRaisesRegex(ValueError,'queue required'):watch_sweep(args)
                execute.assert_not_called()

    def test_metadata_waits_for_all_fits_and_repaired_panels(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study=Path(d);cell=dict(id='cell',stage='outer',partition='outer-0',family='relu')
            admissions={0:dict(evaluation_representations={'relu':dict(representation='research-relu-max-pairs-v1')})}
            self.assertFalse(ready_cells(study,[cell],admissions))
            run=study/'runs/cell/attempt-001';write(run/'RESULT.json',dict(cell=cell,stage='trained',exit_code=0))
            write(run/'evaluation/COMPLETE.json',{})
            self.assertFalse(ready_cells(study,[cell],admissions))
            write(study/'reexport_max_pairs/cell/attempt-001/final/COMPLETE.json',{})
            self.assertTrue(ready_cells(study,[cell],admissions))
            (study/'runs/cell/attempt-002').mkdir();self.assertFalse(ready_cells(study,[cell],admissions))
            write(study/'runs/cell/attempt-002/RESULT.json',dict(cell=cell,stage='trained',exit_code=0))
            write(study/'reexport_max_pairs/cell/attempt-002/final/COMPLETE.json',{})
            with self.assertRaisesRegex(ValueError,'ambiguous'):ready_cells(study,[cell],admissions)

    def test_complete_output_checks_summary_and_sample_hashes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);self.assertFalse(complete_output(root))
            (root/'samples.npz').write_bytes(b'samples');write(root/'SUMMARY.json',dict(samples_sha256=sha(root/'samples.npz')))
            write(root/'COMPLETE.json',dict(summary_sha256=sha(root/'SUMMARY.json')));self.assertTrue(complete_output(root))
            (root/'samples.npz').write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError,'output drift'):complete_output(root)

    def test_complete_queue_waits_then_preserves_failure_and_continues(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study=root/'study';registry=root/'registry'
            entries=[dict(file=f'comparison-{i:03}.json',spec=dict(fresh_family='test')) for i in range(88)]
            write(registry/'MATCHED_REGISTRATION.json',dict(comparisons=entries))
            args=SimpleNamespace(study=study,matched_registry=registry,draws=root/'draws',binary_dir=root/'bin',output=root/'out',archive_destination=root/'archive')
            cell=dict(id='cell')
            def child(command,**kwargs):
                if '--cold-readback' in command:
                    output=Path(command[command.index('--root')+1]);dest=Path(command[command.index('--destination')+1]);dest.mkdir(parents=True)
                    (output/'SHA256SUMS').write_text('manifest');(dest/'SHA256SUMS').write_text('manifest')
                    write(output/'MIRROR_VERIFIED.json',dict(destination=str(dest),all_artifact_sha256_verified=True))
                    return SimpleNamespace(returncode=0)
                output=Path(command[command.index('--output')+1]);output.mkdir()
                if output.name=='primary':
                    write(output/'FAILURE.json',dict(reason='retained failure'));return SimpleNamespace(returncode=1)
                (output/'samples.npz').write_bytes(b'samples')
                write(output/'SUMMARY.json',dict(samples_sha256=sha(output/'samples.npz')))
                write(output/'COMPLETE.json',dict(summary_sha256=sha(output/'SUMMARY.json')))
                return SimpleNamespace(returncode=0)
            with patch('shaping_execution.load_registration',return_value=({},[],{})),patch('shaping_bootstrap_plan.load_comparison'),patch('shaping_outer_admission.resolve',return_value=({},None)) as admission,patch('shaping_bootstrap.primary_cells',return_value=[cell]),patch('shaping_outer_evidence.matched_cells',return_value=([dict(candidate=cell,control=cell)],(0,1,2))),patch('shaping_bootstrap.ready_cells',return_value=True),patch('shaping_bootstrap.subprocess.run',side_effect=child) as execute:
                status=watch_sweep(args);self.assertEqual(len(status['waiting']),89);execute.assert_not_called();admission.assert_not_called()
                for o in range(3):write(study/'selection'/f'outer-{o}'/'OUTER_ADMISSION.json',{})
                status=watch_sweep(args);self.assertEqual(len(status['completed']),88);self.assertEqual(len(status['archived']),88);self.assertEqual(status['failed'],['primary']);self.assertEqual(execute.call_count,177)
                status=watch_sweep(args);self.assertEqual(status['failed'],['primary']);self.assertEqual(execute.call_count,177)
                self.assertTrue((args.output/'primary/FAILURE.json').exists())
                (args.archive_destination/'comparison-000/SHA256SUMS').write_text('changed')
                with self.assertRaisesRegex(ValueError,'archive readback incomplete'):watch_sweep(args)
                self.assertEqual(execute.call_count,177)
