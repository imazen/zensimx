import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_archive_supervision import failed_eligible
from shaping_completion import encoded,sha


class FailedArchiveTests(unittest.TestCase):
    def fixture(self,base,kind):
        study=base/'study';directory=study/'runs/cell';run=directory/'attempt-001';run.mkdir(parents=True)
        (study/'PLAN.json').write_text('{}');cell=dict(id='cell',stage='inner',family='relu')
        def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(encoded(d))
        write(run/'RUN.json',dict(cell=cell,plan_sha256=sha(study/'PLAN.json')))
        bake=run/'checkpoints/ckpt_epoch119.bin';bake.parent.mkdir();bake.write_bytes(b'model')
        write(run/'RESULT.json',dict(cell=cell,stage='trained',exit_code=0,pid=99999999,epochs=120,bake_sha256=sha(bake)))
        for name in ['evaluation','curve_evaluation/epoch-049','curve_evaluation/epoch-099','curve_evaluation/epoch-119']:
            panel=run/name;write(panel/'REPORT.json',{'cell':cell})
            write(panel/('COMPLETE.json' if name=='evaluation' else 'DIAGNOSTIC_COMPLETE.json'),dict(report_sha256=sha(panel/'REPORT.json')))
        failure=dict(cell=cell,scientifically_complete=False,outer_results_read=False)
        if kind=='gate':
            gate=run/'curve_evaluation/epoch-049/FAILED_GATE.json';write(gate,{'finite':False})
            failure['failures']=[dict(gate=str(gate),gate_sha256=sha(gate))];name='EVALUATION_FAILURES.json'
        else:
            failure['evidence_sha256']={str(run/'RESULT.json'):sha(run/'RESULT.json')};name='CONVERGENCE_FAILURE.json'
        write(run/name,failure)
        return study,directory,run,cell

    def test_gate_and_convergence_failures_preserved_without_qualification(self):
        for kind in ['gate','convergence']:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp:
                study,directory,run,cell=self.fixture(Path(temp),kind)
                self.assertTrue(failed_eligible(study,directory,[cell]))
                self.assertFalse((run/'CONVERGENCE.json').exists())
                (run/'curve_evaluation/epoch-099/DIAGNOSTIC_COMPLETE.json').unlink()
                self.assertFalse(failed_eligible(study,directory,[cell]))

    def test_changed_evidence_and_models_refused(self):
        for mode in ['gate','report','model','registration']:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp:
                study,directory,run,cell=self.fixture(Path(temp),'gate')
                if mode=='gate':(run/'curve_evaluation/epoch-049/FAILED_GATE.json').write_text('{}')
                if mode=='report':(run/'evaluation/REPORT.json').write_text('{}')
                if mode=='model':(run/'checkpoints/ckpt_epoch119.bin').write_bytes(b'changed')
                with self.assertRaises(ValueError):failed_eligible(study,directory,[] if mode=='registration' else [cell])

    def test_ambiguous_attempts_are_not_archived(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp:
            study,directory,run,cell=self.fixture(Path(temp),'convergence')
            (directory/'attempt-002').mkdir()
            self.assertFalse(failed_eligible(study,directory,[cell]))

    def test_rescue_checkpoint_uses_declared_fit_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp:
            study,directory,run,cell=self.fixture(Path(temp),'gate')
            identity=json.loads((run/'RUN.json').read_text());identity['fit_directory']='fit'
            (run/'RUN.json').write_bytes(encoded(identity));(run/'fit').mkdir()
            (run/'checkpoints').rename(run/'fit/checkpoints')
            self.assertTrue(failed_eligible(study,directory,[cell]))

    def test_extension_uses_complete_conditional_registration(self):
        for drift in (None,'cell','identity','result','unregistered'):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp:
                study,directory,run,cell=self.fixture(Path(temp),'gate')
                cell['stage']='exposure_extension'
                resolution=dict(cell=cell,schedule=dict(phase_epochs=[120]))
                for name in ('RUN.json','RESULT.json','EVALUATION_FAILURES.json'):
                    path=run/name;row=json.loads(path.read_text());row['cell']=cell
                    if name in ('RUN.json','RESULT.json'):
                        row['resolution']={} if drift==('identity' if name=='RUN.json' else 'result') else resolution
                    path.write_bytes(encoded(row))
                admitted=json.loads(json.dumps(resolution))
                if drift=='cell':admitted['cell']['family']='softplus'
                with patch('shaping_extension.resolve',return_value=admitted,
                           side_effect=ValueError('not registered') if drift=='unregistered' else None) as resolve:
                    if drift is None:self.assertTrue(failed_eligible(study,directory,[]))
                    else:
                        with self.assertRaises(ValueError):failed_eligible(study,directory,[])
                    resolve.assert_called_once_with(study,cell['id'])


if __name__=='__main__':unittest.main()
