import json,os,signal,socket,subprocess,sys,tempfile,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import sha
from shaping_pending_rescue_placement import retire


class PendingTests(unittest.TestCase):
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
