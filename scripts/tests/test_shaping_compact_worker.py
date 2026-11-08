"""Exercise subprocess gate isolation and real idle-process handoff boundaries."""
import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_compact_worker as compact


class CompactWorkerTests(unittest.TestCase):
    def fixture(self,root,active_child=False):
        source=root/'source';scripts=source/'scripts/rev4_featpot';scripts.mkdir(parents=True)
        (scripts/'shaping_rescue_worker.py').write_text('''import json,subprocess,sys,time\nfrom pathlib import Path\ndef verify(record):\n if not record['approved']:raise ValueError('fixture admission refused')\nif __name__=='__main__':\n wave=Path(sys.argv[sys.argv.index('--request')+1]);ready=wave.parent/'READY'\n if '''+repr(active_child)+''':\n  child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);ready.write_text(str(child.pid))\n else:ready.write_text('idle')\n time.sleep(60)\n''')
        (scripts/'shaping_corrected_worker.py').write_text("def reproduce(record):return record['actual_resolution'],record['native_command']\n")
        study=root/'study';study.mkdir()
        for name in ('cpu-admission','storage-admission'):(study/name).mkdir()
        (study/'cpu-admission/CLAIMS.json').write_text('{}')
        record=dict(study=str(study),cell=dict(id='cell'),execution_kind='corrected_inner',approved=True,
                    resolution={'setting':2},actual_resolution={'setting':2},native_command=['unchanged-native-command'])
        request=root/'REQUEST.json';request.write_text(json.dumps(record))
        wave=root/'WAVE.json';wave.write_text(json.dumps(dict(workers=1,requests=[str(request)],request_sha256={str(request):compact.sha(request)})))
        return source,study,request,wave,record

    def command(self,mode,source,**options):
        return [sys.executable,'-B',compact.__file__,mode,'--source',str(source),
                *[v for key,value in options.items() for v in ('--'+key,str(value))]]

    def test_fresh_process_runs_original_gate_without_starting_native_command(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);source,study,request,wave,record=self.fixture(root)
            result=subprocess.run(self.command('preflight',source,request=request),capture_output=True,text=True,check=True)
            value=json.loads(result.stdout);self.assertEqual(value['resolution'],record['resolution'])
            self.assertEqual(value['command'],record['native_command']);self.assertFalse((study/'runs').exists())
            record['actual_resolution']={'setting':3};request.write_text(json.dumps(record))
            result=subprocess.run(self.command('preflight',source,request=request),capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('original evidence gate',result.stderr)
            self.assertFalse((study/'runs').exists())

    def test_existing_attempt_and_failed_preflight_cannot_reach_cpu_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            source,study,request,wave,record=self.fixture(Path(directory));frozen=SimpleNamespace(verify=lambda record:None)
            with patch.object(compact,'install',return_value=frozen),patch.object(compact,'preflight',side_effect=ValueError('gate refused')):
                with self.assertRaisesRegex(ValueError,'gate refused'):compact.work(source,request)
                self.assertEqual(json.loads((study/'cpu-admission/CLAIMS.json').read_text()),{})
                (study/'runs/cell').mkdir(parents=True)
                with self.assertRaisesRegex(ValueError,'never replay'):compact.work(source,request)
            self.assertFalse(request.with_name('worker.log').exists())

    def test_wave_manifest_drift_and_duplicate_requests_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            source,study,request,wave,record=self.fixture(Path(directory));value=json.loads(wave.read_text())
            value['requests'].append(str(request));wave.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'distinct'):compact.manifest(wave)
            value['requests'].pop();wave.write_text(json.dumps(value));request.write_text('changed')
            with self.assertRaisesRegex(ValueError,'request drift'):compact.manifest(wave)

    def test_only_an_idle_wave_can_retire_and_active_children_are_preserved(self):
        for active in (False,True):
            with self.subTest(active_child=active),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);source,study,request,wave,record=self.fixture(root,active)
                process=subprocess.Popen([sys.executable,str(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),'wave','--request',str(wave)])
                child=None
                try:
                    for _ in range(100):
                        if (root/'READY').exists():break
                        time.sleep(.01)
                    else:self.fail('fixture process did not become ready')
                    if active:child=int((root/'READY').read_text())
                    wave.with_name('WAVE_STATUS.json').write_text(json.dumps(dict(pid=process.pid,manifest_sha256=compact.sha(wave),completed={},failed={})))
                    result=subprocess.run(self.command('handoff',source,wave=wave,output=root/'handoff'),capture_output=True,text=True)
                    if active:
                        self.assertNotEqual(result.returncode,0);self.assertIn('active children',result.stderr)
                        self.assertIsNone(process.poll());os.kill(child,0)
                    else:
                        self.assertEqual(result.returncode,0,result.stderr);process.wait(timeout=3)
                        receipt=json.loads((root/'handoff/HANDOFF.json').read_text())
                        self.assertEqual(receipt['stage'],'retired_idle_wave');self.assertEqual(receipt['native_fits_interrupted'],0)
                    self.assertFalse((study/'runs').exists())
                finally:
                    if child:
                        try:os.kill(child,signal.SIGTERM)
                        except ProcessLookupError:pass
                    if process.poll() is None:process.terminate()
                    process.wait(timeout=3)


    def test_idle_freeze_retains_the_verified_queue_for_placement_transition(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);source,study,request,wave,record=self.fixture(root)
            process=subprocess.Popen([sys.executable,str(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),'wave','--request',str(wave)])
            try:
                for _ in range(100):
                    if (root/'READY').exists():break
                    time.sleep(.01)
                else:self.fail('fixture process did not become ready')
                wave.with_name('WAVE_STATUS.json').write_text(json.dumps(dict(pid=process.pid,manifest_sha256=compact.sha(wave),completed={},failed={})))
                subprocess.run([*self.command('handoff',source,wave=wave,output=root/'freeze'),'--freeze-only'],capture_output=True,text=True,check=True)
                receipt=json.loads((root/'freeze/HANDOFF.json').read_text());self.assertEqual(receipt['stage'],'frozen_idle_wave')
                self.assertEqual((Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()[0],'T')
                self.assertIsNone(process.poll());self.assertFalse((study/'runs').exists())
            finally:
                if process.poll() is None:
                    process.terminate();os.kill(process.pid,signal.SIGCONT)
                process.wait(timeout=3)


if __name__=='__main__':unittest.main()
