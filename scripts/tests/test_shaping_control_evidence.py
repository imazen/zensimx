import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from scripts.tests.test_shaping_corrected_view import CorrectedViewTests
import shaping_control_evidence as evidence
from shaping_completion import encoded,sha


class ControlEvidenceTests(unittest.TestCase):
    def test_exact_native_inventory_excludes_scratch_and_rejects_changed_bakes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            root=Path(temporary);source=root/'original';source.mkdir();(source/'RESULT.json').write_text('{}')
            run=root/'run';(run/'checkpoints').mkdir(parents=True);case,result=CorrectedViewTests().proof()
            case['run']=str(source);registration=root/'REGISTRATION.json';registration.write_bytes(encoded(dict(cases=dict(relu=case))))
            result.update(registration_sha256=sha(registration),source_result_sha256=sha(source/'RESULT.json'))
            for epoch in range(240):
                path=run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin';path.write_bytes(str(epoch).encode())
                result['trajectory']['compared'][path.name]['input_sha256']['neutral']=sha(path)
            for name in ('RUN.json','TRAINING_RESULT.json','train.log','best.bin','best.bin.spec.json'):(run/name).write_text('{}')
            (run/'RESULT.json').write_bytes(encoded(result));scratch=run/'trajectory/intermediate.bin';scratch.parent.mkdir();scratch.write_bytes(b'not copied')
            inventory=evidence.prepare(registration,'relu',run,root/'inventory')
            self.assertEqual(len(inventory['files']),246);self.assertNotIn('trajectory/intermediate.bin',inventory['files'])
            self.assertTrue(evidence.verify(root/'inventory/INVENTORY.json',run)['all_files_verified'])
            (run/'checkpoints/ckpt_epoch001.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed after comparison'):evidence.prepare(registration,'relu',run,root/'other-inventory')
            with self.assertRaisesRegex(ValueError,'drift'):evidence.verify(root/'inventory/INVENTORY.json',run)


if __name__=='__main__':unittest.main()
