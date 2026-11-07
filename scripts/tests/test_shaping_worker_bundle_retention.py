import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import sha
from shaping_worker_bundle_retention import plan,REMOTE

class BundleRetentionTests(unittest.TestCase):
    def fixture(self,root):
        cell='a'*20;name=cell+'-attempt-001.tar';study=root/'study';archive=root/'archive'
        bundle=archive/('i270-'+name[:-4])/name;bundle.parent.mkdir(parents=True);bundle.write_bytes(b'complete archived bundle')
        producer=dict(cell={'id':cell},attempt='attempt-001',bundle_sha256=sha(bundle),assignment_sha256='assignment')
        sidecar=Path(str(bundle)+'.json');sidecar.write_text(json.dumps(producer))
        verified=dict(bundle=str(bundle),bundle_bytes=bundle.stat().st_size,bundle_sha256=sha(bundle),assignment_sha256='assignment',producer_receipt_sha256=sha(sidecar))
        Path(str(bundle)+'.verified.json').write_text(json.dumps(verified))
        collection=study/'runs'/cell/'attempt-001/WORKER_COLLECTION.json';collection.parent.mkdir(parents=True)
        collection.write_text(json.dumps(dict(source_host='i270',bundle_sha256=sha(bundle),assignment_sha256='assignment')))
        collection.with_name('RESULT.json').write_text('{"stage":"trained","exit_code":0}')
        worker=root/'worker';transport=worker/'worker-bundles'/name;transport.parent.mkdir(parents=True);transport.write_bytes(bundle.read_bytes());Path(str(transport)+'.json').write_bytes(sidecar.read_bytes())
        original=worker/'study/runs'/cell/'attempt-001/important.bin';original.parent.mkdir(parents=True);original.write_bytes(b'original')
        rows=plan(study,archive,'i270',{name:bundle.stat().st_size})
        return study,archive,bundle,worker,transport,original,rows

    def apply(self,worker,rows,busy=False):
        payload=dict(mode='apply',root=str(worker),receipt='retention.json',rows=rows)
        with patch.object(sys,'argv',['remote',json.dumps(payload)]),patch('subprocess.run',return_value=SimpleNamespace(returncode=0 if busy else 1,stdout='123\n' if busy else '',stderr='')),contextlib.redirect_stdout(io.StringIO()):
            exec(REMOTE,{})

    def test_success_removes_transport_only_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study,archive,bundle,worker,transport,original,rows=self.fixture(Path(directory))
            self.apply(worker,rows)
            self.assertFalse(transport.exists());self.assertTrue(original.exists());self.assertTrue(bundle.exists())
            self.assertTrue(Path(str(transport)+'.json').exists());self.assertEqual(json.loads((worker/'retention.json').read_text())['stage'],'duplicates_removed')
            with self.assertRaisesRegex(ValueError,'existing'):self.apply(worker,rows)

    def test_corrupt_durable_worker_and_busy_files_refuse(self):
        for mode in ['durable','worker','busy']:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                study,archive,bundle,worker,transport,original,rows=self.fixture(Path(directory))
                if mode=='durable':
                    bundle.write_bytes(b'changed archived bundle!')
                    with self.assertRaises(ValueError):plan(study,archive,'i270',{bundle.name:bundle.stat().st_size})
                else:
                    if mode=='worker':transport.write_bytes(b'changed')
                    with self.assertRaises(ValueError):self.apply(worker,rows,busy=mode=='busy')
                self.assertTrue(transport.exists());self.assertTrue(original.exists());self.assertFalse((worker/'retention.json').exists())

if __name__=='__main__':unittest.main()
