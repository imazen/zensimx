import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_queue_transfer as transfer
from shaping_completion import encoded,sha


class QueueTransferTests(unittest.TestCase):
    def test_real_queue_handoff_preserves_unstopped_child_and_request_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root,plan,requests=self.fixture(directory);script=root/'shaping_rescue_worker.py'
            script.write_text("import subprocess,sys\nfrom pathlib import Path\np=subprocess.Popen(['sleep','30'])\nPath(sys.argv[-1]).with_name('CHILD_PID').write_text(str(p.pid))\np.wait()\n")
            manifest=root/'WAVE.json';process=subprocess.Popen([sys.executable,str(script),'wave','--request',str(manifest)])
            child=None
            try:
                deadline=time.monotonic()+3
                while not (root/'CHILD_PID').exists() and time.monotonic()<deadline:time.sleep(.01)
                child=int((root/'CHILD_PID').read_text())
                status=root/'WAVE_STATUS.json';value=json.loads(status.read_text());value['pid']=process.pid;status.write_bytes(encoded(value))
                before=sha(requests[2]);rename=Path.rename
                def check(path,target):
                    parent_state=(Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
                    child_state=(Path('/proc')/str(child)/'stat').read_text().rsplit(')',1)[1].split()[0]
                    self.assertEqual(parent_state,'T');self.assertNotIn(child_state,('T','t','Z'))
                    return rename(path,target)
                with patch.object(Path,'rename',new=check):record=transfer.transfer(plan)
                self.assertFalse(Path(requests[2]).exists());self.assertEqual(sha(record['files'][requests[2]]['preserved_path']),before)
                self.assertTrue(all(Path(p).exists() for p in requests[:2]))
                self.assertNotEqual((Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()[0],'T')
            finally:
                if child is not None:os.kill(child,signal.SIGTERM)
                if process.poll() is None:process.wait(timeout=3)

    def fixture(self,directory):
        root=Path(directory)
        for name in ('cpu-admission','storage-admission'):(root/name).mkdir()
        (root/'cpu-admission/CLAIMS.json').write_text('{}')
        requests=[]
        for cell in ('active','waiting','queued'):
            directory=root/cell;directory.mkdir();path=directory/'REQUEST.json'
            path.write_bytes(encoded(dict(study=str(root),cell=dict(id=cell))));requests.append(str(path))
        manifest=root/'WAVE.json';manifest.write_bytes(encoded(dict(workers=2,requests=requests,request_sha256={p:sha(p) for p in requests})))
        (root/'WAVE_STATUS.json').write_bytes(encoded(dict(pid=123,completed={},failed={},manifest_sha256=sha(manifest))))
        plan=root/'PLAN.json';plan.write_bytes(encoded(dict(wave=str(manifest),wave_sha256=sha(manifest),requests=requests[2:],output=str(root/'TRANSFER.json'))))
        return root,plan,requests

    def test_active_and_waiting_slots_never_transfer(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root,plan,requests=self.fixture(directory);value=json.loads(plan.read_text());value['requests']=[requests[1]];plan.write_bytes(encoded(value))
            with self.assertRaisesRegex(ValueError,'exclude every initial'):transfer.transfer(plan)
            self.assertTrue(all(Path(p).exists() for p in requests))

    def test_pid_mismatch_never_signals_or_transfers(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root,plan,requests=self.fixture(directory)
            # A recycled PID cannot authorize signals; an absent PID also fails.
            with patch.object(transfer.os,'kill') as signal:
                with self.assertRaises((FileNotFoundError,ValueError)):transfer.transfer(plan)
                signal.assert_not_called()
            self.assertTrue(all(Path(p).exists() for p in requests))


if __name__=='__main__':unittest.main()
