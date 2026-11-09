import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_archive_supervision import rescue_eligible
from shaping_completion import encoded,sha


class RescueArchiveTests(unittest.TestCase):
    def fixture(self,root,pretrained=False,extension=False):
        (root/'PLAN.json').write_text('{}');directory=root/'runs/cell';run=directory/'attempt-001'
        run.mkdir(parents=True);cell=dict(id='cell',stage='exposure_extension' if extension else 'rescue',family='product_pretrained_frozen' if pretrained else 'relu')
        epochs=(240 if pretrained else 480) if extension else 120 if pretrained else 240
        if extension:cell.update(phase_epochs=[epochs,epochs] if pretrained else [epochs],total_pair_draws=24000000)
        resolution=dict(cell=cell,registered=True)
        identity=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),build_sha256='build',epochs=epochs,
            pairs_per_epoch=50000,training_exposure_pair_draws=24000000 if extension else 12000000,fit_directory='fit',
            process_role='extension_controller' if extension else 'rescue_controller',pid=999999999,resolution=resolution)
        phases=[]
        for name in (['pretraining','fit'] if pretrained else ['fit']):
            p=run/name;(p/'checkpoints').mkdir(parents=True);model=p/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin';model.write_bytes(name.encode())
            manifest=p/'OPTIMIZER.json';manifest.write_text('{}')
            phase=dict(stage='trained',exit_code=0,epochs=epochs,pairs_per_epoch=50000,
                       bake_sha256=sha(model),manifest_sha256={str(manifest):sha(manifest)})
            (p/'RESULT.json').write_bytes(encoded(phase));phases.append(phase)
        (run/'RUN.json').write_bytes(encoded(identity));(run/'RESULT.json').write_bytes(encoded(dict(identity,stage='trained',exit_code=0,phases=phases)))
        (run/'CONVERGENCE.json').write_text('{}');(run/'evaluation').mkdir();(run/'evaluation/COMPLETE.json').write_text('{}')
        return directory,run,cell,resolution

    def test_complete_single_and_two_phase_rescues_use_existing_candidate_gate(self):
        for pretrained in (False,True):
            with self.subTest(pretrained=pretrained), tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);directory,run,cell,resolution=self.fixture(root,pretrained)
                with patch('shaping_rescue.resolve',return_value=resolution),patch('shaping_rescue_selection.verified_candidate') as verify:
                    self.assertTrue(rescue_eligible(root,directory,[cell]));verify.assert_called_once_with(root,cell)
                    model=run/'fit/checkpoints'/('ckpt_epoch119.bin' if pretrained else 'ckpt_epoch239.bin');model.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError,'phase model drift'):rescue_eligible(root,directory,[cell])

    def test_qualified_canonical_reports_allow_retention_with_original_failures_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);directory,run,cell,resolution=self.fixture(root,True)
            canonical=root/'reexport_max_pairs/cell/attempt-001';(canonical/'final').mkdir(parents=True)
            (canonical/'CONVERGENCE.json').write_text('{}');(canonical/'final/COMPLETE.json').write_text('{}')
            (run/'CONVERGENCE.json').unlink();(run/'evaluation/COMPLETE.json').unlink()
            failed=run/'EVALUATION_FAILURES.json';failed.write_text('preserved raw order failure')
            with patch('shaping_completion.max_pair_families',return_value=(cell['family'],)), \
                 patch('shaping_rescue.resolve',return_value=resolution),patch('shaping_rescue_selection.verified_candidate') as verify:
                self.assertTrue(rescue_eligible(root,directory,[cell]))
                verify.assert_called_once_with(root,cell,'research-relu-max-pairs-v1')
            self.assertEqual(failed.read_text(),'preserved raw order failure')

    def test_registered_extensions_preserve_continuous_and_pretrained_exposure(self):
        for pretrained in (False,True):
            with self.subTest(pretrained=pretrained),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);directory,run,cell,resolution=self.fixture(root,pretrained,True)
                with patch('shaping_extension.resolve',return_value=resolution),patch('shaping_rescue_selection.verified_candidate'):
                    self.assertTrue(rescue_eligible(root,directory,[]))
                    result=json.loads((run/'RESULT.json').read_text());result['training_exposure_pair_draws']-=50000
                    (run/'RESULT.json').write_text(json.dumps(result))
                    with self.assertRaisesRegex(ValueError,'identity or exposure differs'):rescue_eligible(root,directory,[])

    def test_failed_or_unevaluated_runs_remain_with_their_existing_owner(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);directory,run,cell,_=self.fixture(root)
            (run/'EVALUATION_FAILURES.json').write_text('{}')
            with patch('shaping_rescue_selection.verified_candidate') as verify:
                self.assertFalse(rescue_eligible(root,directory,[cell]));verify.assert_not_called()
            (run/'EVALUATION_FAILURES.json').unlink();(run/'evaluation/COMPLETE.json').unlink()
            self.assertFalse(rescue_eligible(root,directory,[cell]))

    def test_incomplete_pretraining_and_resolution_drift_refuse_archival(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);directory,run,cell,resolution=self.fixture(root,True)
            with patch('shaping_rescue.resolve',return_value={'different':True}):
                with self.assertRaisesRegex(ValueError,'resolution differs'):rescue_eligible(root,directory,[cell])
            result=json.loads((run/'RESULT.json').read_text());result['phases']=result['phases'][1:];(run/'RESULT.json').write_bytes(encoded(result))
            with patch('shaping_rescue.resolve',return_value=resolution):
                with self.assertRaisesRegex(ValueError,'phase exposure differs'):rescue_eligible(root,directory,[cell])


if __name__=='__main__':unittest.main()
