"""Downstream controller admission must cover its actual native child cap."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_extension_execution import command
from shaping_rescue_worker import memory_allowance


class PhaseMemoryTests(unittest.TestCase):
    def test_admitted_schedule_covers_each_native_phase(self):
        # Use the actual native command builder, including its doubled event cap.
        original=['run-heavy','--mem','2G','--jobs','1','--',
            '/native/zensim_mlp_train','--epochs','240','--pairs-per-epoch','50000',
            '--out','/source/best.bin','--dump-checkpoints-dir','/source/checkpoints']
        for kind in ('outer','variant'):
            for phases in ([240],[480],[240,240],[240,480]):
                record=dict(execution_kind=kind,resolution=dict(schedule=dict(phase_epochs=phases)))
                admitted=memory_allowance(record)
                for epochs in phases:
                    argv=command(original,Path('/candidate'),Path('/dest'),epochs)
                    child=int(argv[argv.index('--mem')+1][:-1])*1024**3
                    self.assertGreaterEqual(admitted,child)
                self.assertEqual(admitted,(4 if 480 in phases else 2)*1024**3)

    def test_invalid_schedule_fails_before_resource_admission(self):
        for phases in ([],[0],[True],[960]):
            with self.assertRaisesRegex(ValueError,'phase lengths'):
                memory_allowance(dict(execution_kind='outer',resolution=dict(schedule=dict(phase_epochs=phases))))

    def test_existing_control_and_rescue_bounds_are_preserved(self):
        for kind in ('extension','control'):
            self.assertEqual(memory_allowance(dict(execution_kind=kind)),4*1024**3)
        for kind in (None,'corrected_inner','bootstrap'):
            self.assertEqual(memory_allowance(dict(execution_kind=kind)),2*1024**3)


if __name__=='__main__':unittest.main()
