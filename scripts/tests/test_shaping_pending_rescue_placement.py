import json,os,signal,socket,subprocess,sys,tempfile,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import sha
from shaping_pending_rescue_placement import retire,relocated_extension


class PendingTests(unittest.TestCase):
    def test_extension_relocation_keeps_every_sealed_field_except_host(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            source=Path(tmp);worker=source/'scripts/rev4_featpot/shaping_rescue_worker.py';worker.parent.mkdir(parents=True);worker.write_text('# sealed source\n')
            old=dict(host='r7900x',execution_kind='extension',cell=dict(id='a'*20,stage='exposure_extension'),
                     source_root=str(source),driver_sha256=sha(worker),files={'kept':dict(sha256='original')},
                     build_sha256='original-native',resolution=dict(seed=2,phase_epochs=[480]))
            new=relocated_extension(old,'r5600g',source)
            self.assertEqual(new,{**old,'host':'r5600g'});self.assertEqual(old['host'],'r7900x')
            for wrong in [dict(old,execution_kind='outer'),dict(old,source_root='elsewhere'),dict(old,driver_sha256='changed')]:
                with self.assertRaises(ValueError):relocated_extension(wrong,'r5600g',source)
            with self.assertRaises(ValueError):relocated_extension(old,'r7900x',source)

    def test_extension_waiter_requires_explicit_stage_and_execution_kind(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
            for name in ('cpu-admission','storage-admission'):(study/name).mkdir()
            worker=root/'shaping_rescue_worker.py';worker.write_text('import time\ntime.sleep(60)\n')
            request=root/'REQUEST.json';request.write_text(json.dumps(dict(study=str(study),host=socket.gethostname(),cell=dict(id='c'*20,stage='exposure_extension'),execution_kind='extension',driver_sha256=sha(worker),plan_sha256=sha(study/'PLAN.json'))))
            child=subprocess.Popen([sys.executable,str(worker),'work','--request',str(request)])
            try:
                time.sleep(.03)
                with self.assertRaisesRegex(ValueError,'no unclaimed'):retire(study,1,root/'wrong-stage')
                self.assertIsNone(child.poll())
                result=retire(study,1,root/'correct-stage',stage='exposure_extension')
                self.assertEqual(child.wait(timeout=3),-signal.SIGTERM)
                self.assertEqual(result['native_processes_signalled'],[])
            finally:
                if child.poll() is None:child.terminate()
                child.wait(timeout=3)

    def test_claimed_controller_is_preserved_and_only_unstarted_waiter_retires(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
            for n in ['cpu-admission','storage-admission']:(study/n).mkdir()
            claimed='a'*20;waiting='b'*20
            (study/'cpu-admission/CLAIMS.json').write_text(json.dumps({claimed:{'owner':'fixture'}}))
            worker=root/'shaping_rescue_worker.py';worker.write_text('import time\ntime.sleep(60)\n')
            children=[]
            try:
                for cid in [claimed,waiting]:
                    path=root/cid/'REQUEST.json';path.parent.mkdir()
                    path.write_text(json.dumps(dict(study=str(study),host=socket.gethostname(),cell=dict(id=cid,stage='rescue'),
                        driver_sha256=sha(worker),plan_sha256=sha(study/'PLAN.json'))))
                    children.append(subprocess.Popen([sys.executable,str(worker),'work','--request',str(path)]))
                time.sleep(.03)
                result=retire(study,1,root/'transition')
                self.assertEqual([r['cell']['id'] for r in result['rows']],[waiting])
                self.assertTrue(result['rows'][0]['retired_before_fit'])
                self.assertIsNone(children[0].poll());self.assertEqual(children[1].wait(timeout=3),-signal.SIGTERM)
                self.assertFalse((study/'runs').exists())
                self.assertEqual(result['native_processes_signalled'],[])
            finally:
                for child in children:
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=3)


if __name__=='__main__':unittest.main()
