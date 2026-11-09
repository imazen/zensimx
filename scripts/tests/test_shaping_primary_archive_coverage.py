import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_primary_archive_coverage import coverage


class CoverageTests(unittest.TestCase):
    def test_control_batch_binds_model_and_completed_evaluation_to_same_plan(self):
        for family in ('extra_trees','shaped_additive'):
            with self.subTest(family=family),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                study=Path(directory)/'study';run=study/'runs/control/attempt-001';(run/'evaluation').mkdir(parents=True)
                (study/'PLAN.json').write_text('{}');cell=dict(stage='inner',id='control',family=family);plan=sha(study/'PLAN.json')
                (run/'RUN.json').write_bytes(encoded(dict(cell=cell,plan_sha256=plan)))
                (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,plan_sha256=plan,stage='complete',exit_code=0)))
                q=dict(native_tree_gate=dict(model_sha256='b'*64)) if family=='extra_trees' else dict(model_sha256='b'*64)
                report=run/'evaluation/REPORT.json';report.write_bytes(encoded(dict(cell=cell,plan_sha256=plan,qualification=q)))
                marker=report.with_name('COMPLETE.json');marker.write_bytes(encoded(dict(report_sha256=sha(report))))
                pins={p.relative_to(study).as_posix():sha(p) for p in (run/'RUN.json',run/'RESULT.json',report,marker)}
                model=run/('model.opaque-vector.zcth' if family=='extra_trees' else 'native/model.bin');pins[model.relative_to(study).as_posix()]='b'*64
                batch=Path(directory)/'batch';batch.mkdir();(batch/'SHA256SUMS').write_text('manifest');(batch/'MIRROR_VERIFIED.json').write_text('{}')
                with patch('shaping_primary_archive_coverage.archive_inventory',return_value=(Path(directory)/'durable',pins)):
                    result=coverage(study,[cell],[batch]);self.assertEqual(result['covered_cells'],1)
                    changed=json.loads(report.read_text());changed['plan_sha256']='wrong';report.write_bytes(encoded(changed));marker.write_bytes(encoded(dict(report_sha256=sha(report))))
                    with self.assertRaisesRegex(ValueError,'control report identity'):coverage(study,[cell],[batch])

    def test_batch_covers_completed_recovery_without_dropping_interrupted_attempt(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study=Path(directory)/'study';run=study/'runs/cell/attempt-002';run.mkdir(parents=True)
            (study/'PLAN.json').write_text('{}');cell=dict(stage='inner',id='cell',family='relu')
            failed=run.with_name('attempt-001');failed.mkdir();(failed/'RESULT.json').write_bytes(encoded(dict(stage='interrupted',exit_code=137)))
            (run/'RUN.json').write_bytes(encoded(dict(cell=cell,plan_sha256=sha(study/'PLAN.json'))))
            (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,plan_sha256=sha(study/'PLAN.json'),stage='trained',exit_code=0,epochs=240,bake_sha256='a'*64)))
            batch=Path(directory)/'batch';batch.mkdir();(batch/'SHA256SUMS').write_text('manifest');(batch/'MIRROR_VERIFIED.json').write_text('{}')
            pins={p.relative_to(study).as_posix():sha(p) for p in (run/'RUN.json',run/'RESULT.json')}
            pins['runs/cell/attempt-002/checkpoints/ckpt_epoch239.bin']='a'*64
            with patch('shaping_primary_archive_coverage.archive_inventory',return_value=(Path(directory)/'durable',pins)):
                result=coverage(study,[cell],[batch]);self.assertTrue(result['complete']);self.assertEqual(result['batch_only_cells'],1)
                self.assertTrue((failed/'RESULT.json').exists());self.assertEqual(result['individual_archive_cells'],0)
                pins['runs/cell/attempt-002/checkpoints/ckpt_epoch239.bin']='changed'
                with self.assertRaisesRegex(ValueError,'does not bind'):coverage(study,[cell],[batch])
                pins.clear();result=coverage(study,[cell],[batch]);self.assertFalse(result['complete']);self.assertEqual(result['missing_cells'],['cell'])


if __name__=='__main__':unittest.main()
