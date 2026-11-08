import json,sys,tempfile,unittest,subprocess
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_worker_watch import pending,cycle,watch


class WorkerWatchTests(unittest.TestCase):
    def test_missing_transfer_tool_fails_before_claiming_or_publishing_status(self):
        with patch('shaping_worker_watch.shutil.which',side_effect=lambda name:None if name=='just' else '/usr/bin/'+name):
            with self.assertRaisesRegex(ValueError,'missing from PATH: just'):
                watch(SimpleNamespace())

    def status(self):
        return dict(checked_utc=datetime.now(timezone.utc).isoformat(),unexpectedly_stopped=[],live=1,
                    runs=[dict(cell='done',terminal='trained',run='/remote/runs/done/attempt-001'),
                          dict(cell='live',terminal=None,run='/remote/runs/live/attempt-001')])
    def test_only_terminal_unimported_fits_and_fresh_observations(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);status=self.status()
            self.assertEqual([r['cell'] for r in pending(root,status)],['done'])
            (root/'runs/done/attempt-001').mkdir(parents=True)
            self.assertEqual(pending(root,status),[])
            status['checked_utc']=(datetime.now(timezone.utc)-timedelta(seconds=121)).isoformat()
            with self.assertRaisesRegex(ValueError,'fresh'):pending(root,status)
            status=self.status();status['unexpectedly_stopped']=['live']
            with self.assertRaisesRegex(ValueError,'reconciliation'):pending(root,status)
    def test_empty_cycle_cannot_start_transfer(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            args=SimpleNamespace(study=Path(directory));status=self.status();status['runs']=status['runs'][1:]
            with patch('shaping_worker_watch.collect_wave') as transfer:
                self.assertIsNone(cycle(args,status,1));transfer.assert_not_called()
    def test_failed_collection_is_not_archived_or_retried(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);args=SimpleNamespace(study=root,output=root,host='i265',source=root,destination=root)
            with patch('shaping_worker_watch.collect_wave',side_effect=ValueError('checksum mismatch')) as transfer,patch('shaping_worker_watch.subprocess.run') as archive:
                with self.assertRaisesRegex(ValueError,'checksum'):cycle(args,self.status(),1)
                transfer.assert_called_once();archive.assert_not_called()
            self.assertTrue((root/'wave-000001-observation.json').exists())

    def test_observation_timeout_does_not_report_training_stopped(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);args=SimpleNamespace(study=root,output=root/'watch',host='i265',source=root,status_source=root,watch=False)
            with patch('shaping_worker_watch.observe',side_effect=subprocess.TimeoutExpired('ssh',45)),patch('shaping_worker_watch.collect_wave') as collect:
                with self.assertRaises(subprocess.TimeoutExpired):watch(args)
                collect.assert_not_called()
            status=json.loads((args.output/'STATUS.json').read_text())
            self.assertEqual(status['stage'],'observation_unavailable_training_state_unknown')
            self.assertEqual(status['completed_waves'],[])


if __name__=='__main__':unittest.main()
