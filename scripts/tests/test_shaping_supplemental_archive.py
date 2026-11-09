from pathlib import Path
import json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_archive_supervision import supplemental_sweep

class SupplementalArchiveTests(unittest.TestCase):
    def test_complete_pins_required_before_mirroring(self):
        self.check_complete_pins('inner')

    def test_extension_requires_resolved_registration_and_complete_pins(self):
        self.check_complete_pins('exposure_extension')

    def check_complete_pins(self,stage):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study=Path(d)/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
            cell=dict(id='cell',family='softplus',stage=stage)
            run=study/'runs/cell/attempt-001';root=study/'reexport_max_pairs/cell/attempt-001'
            def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(data))
            write(run/'RUN.json',dict(cell=cell,plan_sha256=sha(study/'PLAN.json'),build_sha256='train'))
            write(run/'RESULT.json',dict(cell=cell,stage='trained',exit_code=0,epochs=120))
            pins={}
            for name,epoch in [('final',119),('epoch-049',49),('epoch-099',99),('epoch-119',119)]:
                original=run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin';original.parent.mkdir(exist_ok=True);original.write_bytes(b'original')
                bake=study/'reexports/cell/attempt-001'/name/'bake.bin';bake.parent.mkdir(parents=True);bake.write_bytes(b'repaired')
                receipt=bake.parent/'RECEIPT.json'
                write(receipt,dict(original_sha256=sha(original),training_build_sha256='train',evaluation_build_sha256='eval',file_sha256={str(bake):sha(bake)},bake=str(bake)))
                report=root/name/'REPORT.json'
                write(report,dict(cell=cell,checkpoint_epoch=epoch,outer_results_read=False,reexport_receipt=str(receipt),reexport_receipt_sha256=sha(receipt),evaluation_build_sha256='eval',checkpoint_sha256=sha(bake)))
                write(report.parent/('COMPLETE.json' if name=='final' else 'DIAGNOSTIC_COMPLETE.json'),dict(report_sha256=sha(report)))
                if name!='final':pins[str(report)]=sha(report)
            c=dict(cell=cell,outer_results_read=False,original_training_run=str(run),report_sha256=pins,evaluation_build_sha256='eval')
            write(root/'CONVERGENCE.json',c)
            args=SimpleNamespace(study=study,destination=Path(d)/'archive')
            def mirror(command,**kwargs):
                source=Path(command[command.index('--root')+1]);dest=Path(command[command.index('--destination')+1]);dest.mkdir(parents=True)
                manifest=''.join(sha(p)+'  '+p.relative_to(source).as_posix()+'\n' for p in sorted(source.rglob('*')) if p.is_file())
                (source/'SHA256SUMS').write_text(manifest);(dest/'SHA256SUMS').write_text(manifest)
                write(source/'MIRROR_VERIFIED.json',dict(all_artifact_sha256_verified=True,destination=str(dest)))
            registered=[] if stage=='exposure_extension' else [cell]
            with patch('shaping_archive_supervision.load_registration',return_value=({},registered,{})),patch('shaping_archive_supervision.subprocess.run',side_effect=mirror) as native,patch('shaping_extension.resolve',return_value={'cell':cell}) as resolve:
                if stage=='exposure_extension':
                    resolve.side_effect=ValueError('extension registration drift')
                    with self.assertRaisesRegex(ValueError,'registration drift'):supplemental_sweep(args)
                    native.assert_not_called()
                    resolve.side_effect=None
                    resolve.return_value={'cell':dict(cell,seed=99)}
                    with self.assertRaisesRegex(ValueError,'extension identity differs'):supplemental_sweep(args)
                    native.assert_not_called()
                    resolve.return_value={'cell':cell}
                saved=dict(pins);pins.pop(next(iter(pins)));write(root/'CONVERGENCE.json',c)
                with self.assertRaisesRegex(ValueError,'incomplete'):supplemental_sweep(args)
                native.assert_not_called()
                pins.update(saved);write(root/'CONVERGENCE.json',c)
                bake.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError,'artifact drift'):supplemental_sweep(args)
                native.assert_not_called();bake.write_bytes(b'repaired')
                self.assertEqual(supplemental_sweep(args),2)
                def metadata_only(path):
                    if Path(path).suffix=='.bin':raise AssertionError('completed native bake was read again')
                    return sha(path)
                with patch('shaping_archive_supervision.sha',side_effect=metadata_only):
                    self.assertEqual(supplemental_sweep(args),0)
                self.assertEqual(native.call_count,2)
                destination=args.destination/'supplemental-bakes-cell';manifest=destination/'SHA256SUMS';original=manifest.read_bytes();manifest.write_text('changed')
                with self.assertRaisesRegex(ValueError,'archive identity differs'):supplemental_sweep(args)
                manifest.write_bytes(original)
                write(root/'CONVERGENCE.json',dict(c,evaluation_build_sha256='changed'))
                with self.assertRaisesRegex(ValueError,'closed supplemental convergence differs'):supplemental_sweep(args)
                native.assert_called();self.assertEqual(native.call_count,2)
