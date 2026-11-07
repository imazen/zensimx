from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_archive_supervision import later_neural_bundles,later_sweep


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))

class LaterArchiveTests(unittest.TestCase):
    def fixture(self,root,stage='outer',repaired=False):
        study=root/'study';study.mkdir();write(study/'PLAN.json',{})
        cell=dict(id='cell',stage=stage,family='relu');run=study/'runs/cell/attempt-001'
        model=run/'fit/checkpoints/ckpt_epoch119.bin';model.parent.mkdir(parents=True);model.write_bytes(b'model')
        phase=dict(stage='trained',exit_code=0,epochs=120,bake_sha256=sha(model));write(run/'fit/RESULT.json',phase)
        identity=dict(cell=cell,plan_sha256=sha(study/'PLAN.json'),build_sha256='train',epochs=120)
        result=dict(identity,stage='trained',exit_code=0,phases=[phase],bake_sha256=sha(model))
        write(run/'RUN.json',identity);write(run/'RESULT.json',result)
        rep=dict(representation='research-relu-max-pairs-v1' if repaired else 'original',evaluation_build_sha256='eval')
        ctx=dict(admission=dict(evaluation_representations={'relu':rep}),admission_sha256='admit',schedule=dict(phase_epochs=[120]))
        role='outer_test'
        if stage in ('full_fit','full_fit_selection'):
            ctx.update(full_fit=True,descriptive=stage=='full_fit');role='exposed_development' if stage=='full_fit' else 'full_fit_validation'
        panel=study/'reexport_max_pairs/cell/attempt-001/final' if repaired else run/'evaluation'
        panel.mkdir(parents=True);(panel/'predictions.npz').write_bytes(b'opaque prediction bytes')
        report=dict(cell=cell,plan_sha256=sha(study/'PLAN.json'),admission_sha256='admit',checkpoint_epoch=119,
                    outer_results_read=True,checkpoint_sha256=sha(model),predictions_sha256=sha(panel/'predictions.npz'),**{role:{}})
        if repaired:
            bake=study/'reexports/cell/attempt-001/final/repaired.bin';bake.parent.mkdir(parents=True);bake.write_bytes(b'repaired')
            receipt=bake.parent/'RECEIPT.json'
            write(receipt,dict(rep,original_sha256=sha(model),training_build_sha256='train',bake=str(bake),file_sha256={str(bake):sha(bake)}))
            report.update(reexport_receipt=str(receipt),reexport_receipt_sha256=sha(receipt),evaluation_build_sha256='eval',checkpoint_sha256=sha(bake))
        self.save_report(panel,report)
        return study,run,panel,report,ctx

    def save_report(self,panel,report):
        write(panel/'REPORT.json',report);write(panel/'COMPLETE.json',dict(cell='cell',report_sha256=sha(panel/'REPORT.json')))

    def test_original_and_repaired_require_terminal_pinned_evidence(self):
        for repaired in (False,True):
            with self.subTest(repaired=repaired),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                study,run,panel,report,ctx=self.fixture(Path(d),repaired=repaired)
                with patch('shaping_evaluation.outer_context',return_value=ctx) as context:
                    def check():return later_neural_bundles(study,run.parent,[])
                    self.assertEqual(len(check()),3 if repaired else 1)
                    context.side_effect=ValueError('admission drift')
                    with self.assertRaisesRegex(ValueError,'admission drift'):check()
                    context.side_effect=None
                    (panel/'predictions.npz').write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError,'prediction drift'):check()
                    (panel/'predictions.npz').write_bytes(b'opaque prediction bytes')
                    report['inner_validation']={};self.save_report(panel,report)
                    with self.assertRaisesRegex(ValueError,'response role'):check()
                    del report['inner_validation'];self.save_report(panel,report)
                    model=run/'fit/checkpoints/ckpt_epoch119.bin';model.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError,'phase model drift'):check()
                    model.write_bytes(b'model')
                    if repaired:
                        receipt=json.loads(Path(report['reexport_receipt']).read_text());Path(receipt['bake']).write_bytes(b'changed')
                        with self.assertRaisesRegex(ValueError,'conversion artifact drift'):check()
                        Path(receipt['bake']).write_bytes(b'repaired')
                    (panel/'FAILED_GATE.json').write_text('{}');self.assertEqual(check(),[])
                    (panel/'FAILED_GATE.json').unlink();(run.parent/'attempt-002').mkdir();self.assertEqual(check(),[])

    def test_full_fit_waits_for_reproducible_training_only_diagnostics(self):
        for stage in ('full_fit_selection','full_fit'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                study,run,panel,report,ctx=self.fixture(Path(d),stage=stage)
                with patch('shaping_evaluation.outer_context',return_value=ctx),patch('shaping_convergence.training_report',return_value={'evidence':'verified','driver_sha256':'new'}):
                    def check():return later_neural_bundles(study,run.parent,[])
                    self.assertEqual(check(),[])
                    write(run/'TRAIN_CONVERGENCE.json',dict(evidence='verified',driver_sha256='old'))
                    self.assertEqual(len(check()),1)
                    write(run/'TRAIN_CONVERGENCE.json',dict(evidence='changed'))
                    with self.assertRaisesRegex(ValueError,'training diagnostic drift'):check()

    def test_sweep_mirrors_all_repaired_bundles_once(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,run,panel,report,ctx=self.fixture(Path(d),repaired=True)
            args=SimpleNamespace(study=study,destination=Path(d)/'archive')
            def mirror(command,**kwargs):
                self.assertIn('--cold-readback',command)
                source=Path(command[command.index('--root')+1]);dest=Path(command[command.index('--destination')+1])
                dest.mkdir(parents=True);(source/'SHA256SUMS').write_text('manifest');(dest/'SHA256SUMS').write_text('manifest')
                write(source/'MIRROR_VERIFIED.json',dict(destination=str(dest),all_artifact_sha256_verified=True))
            with patch('shaping_archive_supervision.load_registration',return_value=({},[],{})),patch('shaping_evaluation.outer_context',return_value=ctx),patch('shaping_archive_supervision.subprocess.run',side_effect=mirror) as copier:
                (panel/'predictions.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'prediction drift'):later_sweep(args)
                copier.assert_not_called();(panel/'predictions.npz').write_bytes(b'opaque prediction bytes')
                self.assertEqual(later_sweep(args),3)
                self.assertEqual(later_sweep(args),0)
                self.assertEqual(copier.call_count,3)
