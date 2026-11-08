import contextlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,sha
from shaping_stable_export_watch import required_scope
from shaping_stable_export_prepare import rescue_inputs


class RescueScopeTests(unittest.TestCase):
    def test_only_complete_shortlisted_paired_rescues_are_required(self):
        cells=matrix();families={'product_joint_identity','separate_joint_identity'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def shortlist(study,outer):
                p=root/f'{outer}.json';p.write_text('{}')
                selected=[c for c in cells if c['stage']=='rescue' and c['family'] in families
                          and c['partition'].startswith(f'inner-{outer}-') and c['rescue'] in ('init_half','constant_lr')]
                return dict(roles=dict(product='product_joint_identity',separate='separate_joint_identity'),cells=selected),p
            with patch('shaping_reduced_scope.shortlist',side_effect=shortlist):
                required,pins=required_scope(root,cells,'rescue')
            self.assertEqual(len(required),72);self.assertEqual(len(pins),3)
            self.assertTrue(all(c['stage']=='rescue' for c in cells if c['id'] in required))

    def test_both_phase_receipts_and_checkpoints_are_pinned_and_drift_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';phases=[]
            for name in ['pretraining','fit']:
                d=run/name;(d/'checkpoints').mkdir(parents=True)
                for f in ['RUN.json','train.log','HEAD.json']:(d/f).write_text('{}')
                model=d/'checkpoints/ckpt_epoch119.bin';model.write_bytes(b'fixture-model')
                phase=dict(stage='trained',exit_code=0,epochs=120,pairs_per_epoch=50000,bake_sha256=sha(model),
                           manifest_sha256={str(d/'HEAD.json'):sha(d/'HEAD.json')},source_file_sha256={})
                (d/'RESULT.json').write_text(json.dumps(phase));phases.append(phase)
            identity=dict(cell={'id':'a'},resolution={'fixed':True})
            @contextlib.contextmanager
            def reads():yield set()
            with patch('shaping_rescue_worker.reads',reads),patch('shaping_rescue.resolve',return_value={'fixed':True}):
                paths=set();rescue_inputs(root,root/'source',run,identity,dict(phases=phases),paths)
                for name in ['pretraining','fit']:
                    self.assertIn(run/name/'RESULT.json',paths)
                    self.assertIn(run/name/'checkpoints/ckpt_epoch119.bin',paths)
                (run/'pretraining/checkpoints/ckpt_epoch119.bin').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'phase model differs'):
                    rescue_inputs(root,root/'source',run,identity,dict(phases=phases),set())


if __name__=='__main__':unittest.main()
