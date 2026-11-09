import json,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_rescue_host_launch as owner


class LaunchTests(unittest.TestCase):
    def test_canonical_assignment_is_checked_and_copied_before_verification(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);assignment=root/'study/downstream_assignments/cell.json';assignment.parent.mkdir(parents=True)
            assignment.write_text(json.dumps(dict(host='i270',cells=['cell'])))
            job=dict(cell={'id':'cell'},assignment=str(assignment),assignment_sha256=owner.sha(assignment))
            out=root/'out';out.mkdir()
            calls=[]
            def run(argv,**kwargs):calls.append((argv,kwargs));return None
            with patch.object(owner.subprocess,'run',side_effect=run):owner.install_assignments([job],'i270',out)
            self.assertEqual([c[0][0] for c in calls],['ssh','rsync','ssh'])
            self.assertIn(str(assignment),calls[0][1]['input'])
            self.assertIn('--ignore-existing',calls[1][0])
            self.assertTrue((out/'CANONICAL_ASSIGNMENTS_VERIFIED.json').exists())
            assignment.write_text('changed')
            with patch.object(owner.subprocess,'run') as remote:
                with self.assertRaisesRegex(ValueError,'canonical assignment differs'):
                    owner.install_assignments([job],'i270',out)
                remote.assert_not_called()

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
