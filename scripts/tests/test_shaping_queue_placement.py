"""Real queue suspension leaves a running child productive and recoverable."""
import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import sha
import shaping_queue_placement as placement


class QueuePlacementTests(unittest.TestCase):
    def test_child_keeps_progressing_while_parent_admission_is_frozen(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root=Path(name);study=root/'study';study.mkdir()
            for part in ('cpu-admission','storage-admission'):(study/part).mkdir()
            request=root/'REQUEST.json';request.write_text(json.dumps(dict(study=str(study),cell=dict(id='fixture'))))
            manifest=root/'MANIFEST.json';manifest.write_text(json.dumps(dict(requests=[str(request)],request_sha256={str(request):sha(request)})))
            child=root/'child.py';child.write_text('import time\nfrom pathlib import Path\np=Path('+repr(str(root/'TICK'))+')\nwhile True:\n p.write_text(str(time.monotonic_ns()))\n time.sleep(.02)\n')
            worker=root/'shaping_rescue_worker.py';worker.write_text('''import os,subprocess,sys,time,json
from pathlib import Path
child=subprocess.Popen([sys.executable,'''+repr(str(child))+'''])
Path('''+repr(str(root/'CHILD'))+''').write_text(str(child.pid))
Path('''+repr(str(root/'WAVE_STATUS.json'))+''').write_text(json.dumps(dict(pid=os.getpid(),manifest_sha256='''+repr(sha(manifest))+''')))
time.sleep(60)
''')
            process=subprocess.Popen([sys.executable,str(worker),'wave','--request',str(manifest)]);pid=None;record=None
            try:
                for _ in range(200):
                    if (root/'TICK').exists() and (root/'WAVE_STATUS.json').exists():break
                    time.sleep(.01)
                else:self.fail('fixture child did not start')
                pid=int((root/'CHILD').read_text());record=placement.freeze(study,root/'FREEZE.json')
                self.assertEqual(record['queues'][0]['pid'],process.pid)
                self.assertEqual(record['queues'][0]['children'],[str(pid)])
                before=(root/'TICK').read_text();time.sleep(.1)
                self.assertNotEqual(before,(root/'TICK').read_text())
                self.assertEqual(placement.process_identity(Path('/proc'),process.pid)[0],'T')
                with self.assertRaisesRegex(ValueError,'preserve existing'):placement.freeze(study,root/'FREEZE.json')
                placement.resume(record);record=None;self.assertIsNone(process.poll())
            finally:
                if record:placement.resume(record)
                if pid:
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                process.terminate();process.wait(timeout=3)

    def test_moved_condition_refuses_before_creating_an_attempt(self):
        from shaping_execution import _run_native
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root=Path(name);(root/'worker_assignments').mkdir();(root/'PLAN.json').write_text('{}')
            binaries=root/'bin';binaries.mkdir();(binaries/'BUILD.json').write_text('{}')
            cell=dict(id='moved')
            (root/'worker_assignments/moved.json').write_text(json.dumps(dict(cells=['moved'],host='other-worker',
                plan_sha256=sha(root/'PLAN.json'),build_sha256=sha(binaries/'BUILD.json'))))
            with self.assertRaisesRegex(ValueError,'another worker'):_run_native(root,binaries,cell,{},1,120,'2G')
            self.assertFalse((root/'runs').exists())


if __name__=='__main__':unittest.main()
