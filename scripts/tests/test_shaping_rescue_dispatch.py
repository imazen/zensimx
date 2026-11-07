from pathlib import Path
import sys,tempfile,unittest,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,sha
from shaping_rescue_dispatch import base_ready,completed_runs


class DispatchTests(unittest.TestCase):
    def test_resume_preserves_exposure_and_refuses_failed_or_changed_models(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}')
            cell=dict(id='cell',family='relu');run=root/'runs/cell/attempt-001'
            (run/'fit/checkpoints').mkdir(parents=True)
            checkpoint=run/'fit/checkpoints/ckpt_epoch239.bin';checkpoint.write_bytes(b'model')
            identity=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),build_sha256='build',
                resolution={'setting':1},process_role='rescue_controller',epochs=240,
                pairs_per_epoch=50000,training_exposure_pair_draws=12000000,fit_directory='fit')
            result=dict(identity,stage='trained',exit_code=0,bake_sha256=sha(checkpoint))
            (run/'RUN.json').write_text(json.dumps(identity));(run/'RESULT.json').write_text(json.dumps(result))
            with patch('shaping_rescue_dispatch.resolve',return_value={'setting':1}):
                done,pins=completed_runs(root,[cell],'build');self.assertEqual(done,{'cell'});self.assertEqual(len(pins),3)
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint drift'):completed_runs(root,[cell],'build')
                checkpoint.write_bytes(b'model');result['stage']='failed'
                (run/'RESULT.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'failed rescue'):completed_runs(root,[cell],'build')
                result['stage']='trained';result['epochs']=120
                (run/'RESULT.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'exposure differs'):completed_runs(root,[cell],'build')
                (run/'RESULT.json').unlink()
                with self.assertRaises(FileNotFoundError):completed_runs(root,[cell],'build')

    def test_all_paired_reports_and_diagnostics_are_required(self):
        cells=matrix()
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d)
            selected=[c for c in cells if c['stage']=='inner' and c['family']=='relu' and c['partition'] in ('inner-0-0','inner-0-1')]
            for c in selected:
                p=root/'reexport_max_pairs'/c['id']/'attempt-001';(p/'final').mkdir(parents=True)
                (p/'final/COMPLETE.json').write_text('{}');(p/'CONVERGENCE.json').write_text('{}')
            self.assertTrue(base_ready(root,cells,'relu',0))
            (p/'CONVERGENCE.json').unlink();self.assertFalse(base_ready(root,cells,'relu',0))
            (p/'CONVERGENCE.json').write_text('{}')
            (p/'final/COMPLETE.json').unlink();self.assertFalse(base_ready(root,cells,'relu',0))
            # These are presence checks only. Native selection separately hashes
            # and validates reports; empty fixtures never qualify a selection.


if __name__=='__main__':unittest.main()
