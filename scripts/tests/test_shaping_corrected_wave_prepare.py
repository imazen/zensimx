import json,sys,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_corrected_wave_prepare as owner
from shaping_completion import encoded


class CorrectedWaveTests(unittest.TestCase):
    def fixture(self,d):
        root=Path(d);study=root/'study';source=Path(owner.__file__).resolve().parents[2]
        (study/'corrected_thresholds').mkdir(parents=True)
        for name in ('REGISTRATION.json','ACTIVATION.json'):(study/'corrected_thresholds'/name).write_text('{}')
        binary=root/'binaries';binary.mkdir();(binary/'BUILD.json').write_text('{}')
        cells=[dict(id=f'{i:020x}',family='product_pretrained_joint',partition='inner-0-0',stage='inner') for i in range(5)]
        return study,source,binary,cells

    def gates(self,study,cells):
        stack=ExitStack()
        stack.enter_context(patch('shaping_execution.load_registration',return_value=({},[],{})))
        stack.enter_context(patch.object(owner,'registration',return_value=(dict(plan_sha256='plan'),dict(corrected_pretrained_conditions=[dict(id=c['id']) for c in cells]),cells)))
        return stack

    def test_controllers_have_distinct_status_directories_and_guard_precedes_seal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,source,binary,cells=self.fixture(d);output=Path(d)/'batch'
            def register(command,**kwargs):
                cid=command[command.index('--cell')+1]
                guards=list((study/'worker_assignments').glob('*.json'))
                self.assertEqual(len(guards),1);self.assertIn(cid,json.loads(guards[0].read_text())['cells'])
                job=Path(command[command.index('--output')+1]);job.mkdir(parents=True)
                (job/'REQUEST.json').write_bytes(encoded(dict(cell=cid)))
            with self.gates(study,cells),patch.object(owner.subprocess,'run',side_effect=register):
                result=owner.prepare(study,source,binary,{c['id']:'r3800x' for c in cells},output)
            manifests=[Path(w['manifest']) for w in result['waves']]
            self.assertEqual([w['workers'] for w in result['waves']],[4,1])
            self.assertEqual(len({p.with_name('WAVE_STATUS.json') for p in manifests}),2)

    def test_existing_attempt_is_preserved_before_any_new_placement(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,source,binary,cells=self.fixture(d);output=Path(d)/'batch'
            old=study/'runs'/cells[0]['id'];old.mkdir(parents=True);(old/'RUN.json').write_text('protected')
            with self.gates(study,cells):
                with self.assertRaisesRegex(ValueError,'existing attempts'):
                    owner.prepare(study,source,binary,{cells[0]['id']:'r3800x'},output)
            self.assertFalse(output.exists());self.assertFalse((study/'worker_assignments').exists())
            self.assertEqual((old/'RUN.json').read_text(),'protected')

    def test_unactivated_study_creates_no_wave(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,source,binary,cells=self.fixture(d);output=Path(d)/'batch'
            (study/'corrected_thresholds/ACTIVATION.json').unlink()
            with self.gates(study,cells):
                with self.assertRaisesRegex(ValueError,'activation required'):
                    owner.prepare(study,source,binary,{cells[0]['id']:'r3800x'},output)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
