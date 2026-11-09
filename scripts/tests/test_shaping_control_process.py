import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_execution import transported_control_live
from shaping_completion import sha


class ControlProcessTests(unittest.TestCase):
    def fixture(self,root):
        study=root/'study';directory=study/'runs/cell/attempt-001';request=root/'REQUEST.json'
        run=dict(cell=dict(id='cell'),execution_host='r3500',plan_sha256='plan',build_sha256='build')
        record=dict(execution_kind='control',study=str(study),result_output=str(directory),host='r3500',
                    cell=run['cell'],plan_sha256='plan',build_sha256='build')
        request.write_text(json.dumps(record));request.with_name('STATUS.json').write_text(json.dumps(dict(request_sha256=sha(request))))
        argv=['python3','-B','/export/shaping_control_worker.py','fit','--request',str(request)]
        return study,directory,request,run,record,argv

    def test_transported_process_has_the_exact_admitted_identity(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,directory,request,run,record,argv=self.fixture(Path(d))
            self.assertTrue(transported_control_live(study,run,directory,argv))
            for key in record:
                with self.subTest(key=key):
                    request.write_text(json.dumps(dict(record,**{key:'different'})))
                    request.with_name('STATUS.json').write_text(json.dumps(dict(request_sha256=sha(request))))
                    self.assertFalse(transported_control_live(study,run,directory,argv))
            request.write_text(json.dumps(record));request.with_name('STATUS.json').write_text(json.dumps(dict(request_sha256='drift')))
            self.assertFalse(transported_control_live(study,run,directory,argv))

    def test_unrelated_cli_or_missing_request_does_not_claim_liveness(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,directory,request,run,record,argv=self.fixture(Path(d))
            self.assertFalse(transported_control_live(study,run,directory,[*argv,'--request',str(request)]))
            other=list(argv);other[3]='pack';self.assertFalse(transported_control_live(study,run,directory,other))
            request.unlink();self.assertFalse(transported_control_live(study,run,directory,argv))


if __name__=='__main__':unittest.main()
