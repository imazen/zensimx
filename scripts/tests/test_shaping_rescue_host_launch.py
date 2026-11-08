import json,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_rescue_host_launch as owner


class LaunchTests(unittest.TestCase):
    def test_transfer_failure_starts_no_fit_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch);source=root/'source';driver=source/'scripts/rev4_featpot/shaping_rescue_worker.py'
            driver.parent.mkdir(parents=True);driver.write_text('frozen worker source')
            wave=root/'wave';wave.mkdir();jobs=[]
            for i in range(2):
                request=root/f'request-{i}.json';request.write_text(json.dumps({'study':str(root/'study')}))
                jobs.append(dict(host='i270',cell={'id':str(i)},request=str(request),request_sha256=owner.sha(request)))
            (wave/'REGISTRATION.json').write_text(json.dumps(dict(stage='registered',source_driver_sha256=owner.sha(driver),jobs=jobs)))
            calls=[]
            def fail_second(command,**kwargs):
                calls.append(command)
                if len(calls)==2:raise subprocess.CalledProcessError(1,command)
            out=root/'launch'
            with patch.object(owner.subprocess,'run',side_effect=fail_second):
                with self.assertRaises(subprocess.CalledProcessError):owner.launch([wave],source,'i270',out)
            self.assertTrue((out/'transfer.log').exists())
            self.assertFalse((out/'INPUT_VERIFIED.json').exists())
            self.assertFalse(list(out.glob('*-LAUNCH.json')))
            self.assertFalse(any(c[0]=='ssh' for c in calls))


if __name__=='__main__':unittest.main()
