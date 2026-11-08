import contextlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_rescue_qualification import qualify_on_worker


class QualificationWorkerTests(unittest.TestCase):
    def args(self,root):
        (root/'PLAN.json').write_text(json.dumps(dict(minimum_ssd_free_bytes=30*1024**3)))
        binaries=root/'bin';binaries.mkdir();(binaries/'BUILD.json').write_text('{}')
        return SimpleNamespace(study=root,run=root/'original',binary_dir=binaries,
                               output=root/'qualification',epochs=240,human_only_cell=None)

    def test_host_admission_precedes_claim_or_output(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            args=self.args(Path(directory))
            with patch('shaping_reduced_scope.admit_compute',side_effect=ValueError('host refused')),\
                 patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'host refused'):qualify_on_worker(args)
                claim.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_existing_attempt_is_never_replayed(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            args=self.args(Path(directory));args.output.mkdir()
            with patch('shaping_reduced_scope.admit_compute'),patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'preserve existing'):qualify_on_worker(args)
                claim.assert_not_called()

    def test_capacity_wait_storage_and_release_after_native_failure(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            args=self.args(Path(directory))
            with patch('shaping_reduced_scope.admit_compute'),\
                 patch('shaping_rescue_dispatch.reserve_worker',side_effect=[dict(can_start=False),dict(can_start=True)]) as claim,\
                 patch('shaping_rescue_dispatch.bind_worker') as bind,\
                 patch('shaping_rescue_dispatch.release_worker') as release,\
                 patch('shaping_storage.reservation',return_value=contextlib.nullcontext()) as storage,\
                 patch('shaping_rescue_qualification.time.sleep') as sleep,\
                 patch('shaping_rescue_qualification.qualify',side_effect=RuntimeError('native failure')) as native:
                with self.assertRaisesRegex(RuntimeError,'native failure'):qualify_on_worker(args)
                self.assertEqual(claim.call_count,2);sleep.assert_called_once_with(30)
                key=claim.call_args.args[1]
                bind.assert_called_once();self.assertEqual(bind.call_args.args[:2],(args.study,key))
                storage.assert_called_once_with(args.study,args.output,30*1024**3,bytes_required=1024**3)
                native.assert_called_once_with(args);release.assert_called_once_with(args.study,key)


if __name__=='__main__':unittest.main()
