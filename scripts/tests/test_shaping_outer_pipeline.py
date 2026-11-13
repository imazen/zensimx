from pathlib import Path
import json
import shlex
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import shaping_outer_pipeline as owner


class PipelineTests(unittest.TestCase):
    def test_terminal_control_pack_uses_its_frozen_source(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch)
            request = root / 'REQUEST.json'
            frozen = root / 'frozen-source'
            request.write_text(json.dumps(dict(result_output=str(root / 'result'),
                                              source_root=str(frozen), host='r5600g')))
            args = SimpleNamespace(study=root / 'study', output=root / 'out')
            cell = dict(id='c', family='shaped_additive')
            class PackObserved(Exception):
                pass
            def remote(command, **kwargs):
                if kwargs.get('capture_output'):
                    return SimpleNamespace(stdout=json.dumps(dict(ready=True)))
                actual = shlex.split(command[2])
                self.assertIn(str(frozen / 'scripts/rev4_featpot/shaping_control_worker.py'), actual)
                raise PackObserved()
            with patch.object(owner.subprocess, 'run', side_effect=remote):
                with self.assertRaises(PackObserved):
                    owner.collect_controls(args, [dict(request=str(request), cell=cell)])

    def test_explicit_pool_excludes_withdrawn_hosts_before_observation(self):
        args = SimpleNamespace(worker_hosts=["i134", "r3500", "r3800x", "r5600g"])
        slots = owner.worker_slots(args)
        self.assertEqual(set(slots), set(args.worker_hosts))
        self.assertNotIn("r7900x", slots)
        self.assertNotIn("i270", slots)
        for invalid in ([], ["i134", "i134"], ["unknown"]):
            with self.assertRaisesRegex(ValueError, "distinct known"):
                owner.worker_slots(SimpleNamespace(worker_hosts=invalid))

    def test_busy_hosts_do_not_receive_new_work_ahead_of_empty_hosts(self):
        cells = [dict(id=str(i), seed=i % 3, family='relu') for i in range(12)]
        loads = dict(owner.HOST_SLOTS)
        loads['r7900x'] = 0
        result = owner.placements(cells, 0, loads)
        self.assertEqual([host for _, host in result], ['r7900x'] * 12)
        self.assertEqual(loads['r7900x'], 0)

    def test_spreading_preserves_every_cell_and_uses_each_native_host(self):
        cells = [dict(id=str(i), seed=i % 3, family='relu') for i in range(84)]
        for outer in range(3):
            result = owner.placements(cells, outer)
            self.assertEqual(sorted(c['id'] for c, _ in result), sorted(c['id'] for c in cells))
            counts = {h: sum(host == h for _, host in result) for h in owner.HOST_SLOTS}
            self.assertEqual(counts, owner.HOST_SLOTS)

    def test_no_training_or_preparation_before_complete_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch)
            args = SimpleNamespace(study=root / 'study', output=root / 'out', outer=0)
            with patch('shaping_outer_prepare_worker.register') as register, \
                 patch.object(owner, 'register_fits') as fits, patch.object(owner.time, 'sleep', side_effect=RuntimeError('observed waiting')):
                with self.assertRaisesRegex(RuntimeError, 'observed waiting'):
                    owner.watch(args)
                register.assert_not_called(); fits.assert_not_called()

    def test_uncertain_submission_is_recorded_before_the_remote_call(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch); request = root / 'REQUEST.json'; request.write_text('{}')
            def submit(*args, **kwargs):
                self.assertTrue((root / 'PREPARATION_SUBMITTED.json').exists())
                raise RuntimeError('network uncertainty')
            with patch('shaping_outer_prepare_worker.transfer'), patch.object(owner.subprocess, 'run', side_effect=submit):
                with self.assertRaisesRegex(RuntimeError, 'network uncertainty'):
                    owner.submit_preparation(request, dict(study='/study', outer=0, host='r5600g'), root)


if __name__ == '__main__':
    unittest.main()
