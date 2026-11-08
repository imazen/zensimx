import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_rescue_worker as worker
from shaping_completion import encoded,sha


class RescueWorkerTests(unittest.TestCase):
    def test_missing_declaration_relocates_by_hash_but_existing_drift_refuses(self):
        from shaping_rescue_execution import portable_command
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);command=['trainer','--derived-inputs',str(root/'old-declaration.json')]
            changed=portable_command(command,root)
            self.assertNotEqual(changed,command)
            self.assertEqual(sha(changed[-1]),worker.DECLARATION_SHA256)
            Path(command[-1]).write_text('drift')
            with self.assertRaisesRegex(ValueError,'native declaration drift'):portable_command(command,root)

    def test_worker_reproduces_gate_before_claiming_or_fitting(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);request=root/'REQUEST.json'
            request.write_bytes(encoded(dict(study=str(root),cell=dict(id='new'),resolution=dict(setting=0))))
            with patch.object(worker,'verify'),patch('shaping_rescue.resolve',return_value=dict(setting=1)), \
                 patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'reproduce'):worker.work(request)
                claim.assert_not_called()

    def test_read_capture_restores_path_open_on_failure(self):
        original=Path.open
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path=Path(directory)/'input.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'gate failed'):
                with worker.reads() as observed:
                    path.read_text();path.read_bytes();raise ValueError('gate failed')
            self.assertEqual(observed,{path});self.assertIs(Path.open,original)

    def fixture(self,directory):
        root=Path(directory);data=root/'data';data.mkdir();source=root/'export';source.mkdir()
        a=data/'input.json';a.write_text('pinned');b=source/'policy.md';b.write_text('policy')
        record=dict(schema='shaping-rescue-worker-v1',host='r3500',driver_sha256=sha(worker.__file__),
                    outer_results_read=False,source_root='/old/export',files={
                        str(a):dict(kind='data',name=str(a),bytes=a.stat().st_size,sha256=sha(a)),
                        '/old/export/policy.md':dict(kind='source',name='policy.md',bytes=b.stat().st_size,sha256=sha(b))})
        return record,data,source

    def test_existing_byte_drift_refuses_and_missing_source_is_never_skipped(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            record,data,source=self.fixture(directory)
            with patch.object(worker,'REPO',source),patch.object(worker,'DATA_ROOTS',(data,)),patch.object(worker.socket,'gethostname',return_value='r3500'):
                self.assertEqual(worker.verify(record),[])
                (data/'input.json').write_text('drift')
                with self.assertRaisesRegex(ValueError,'drift'):worker.verify(record,allow_missing=True)
                (data/'input.json').unlink()
                self.assertEqual(worker.verify(record,allow_missing=True),[str(data/'input.json')])
                (source/'policy.md').unlink()
                with self.assertRaises(FileNotFoundError):worker.verify(record,allow_missing=True)

    def test_wrong_host_outer_exposure_and_symlinks_refuse(self):
        for kind in ('host','outer','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                record,data,source=self.fixture(directory)
                if kind=='host':record['host']='dev'
                if kind=='outer':record['outer_results_read']=True
                if kind=='symlink':
                    path=data/'input.json';path.unlink();path.symlink_to(source/'policy.md')
                with patch.object(worker,'REPO',source),patch.object(worker,'DATA_ROOTS',(data,)),patch.object(worker.socket,'gethostname',return_value=record['host']):
                    with self.assertRaises(ValueError):worker.verify(record,allow_missing=True)


if __name__=='__main__':unittest.main()
