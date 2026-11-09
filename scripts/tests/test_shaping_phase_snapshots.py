import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_snapshot_retention as owner


class PhaseSnapshotTests(unittest.TestCase):
    def test_completed_first_phase_keeps_second_phase_reserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);study=root/'study';(study/'cpu-admission').mkdir(parents=True)
            (study/'cpu-admission/CLAIMS.json').write_text(json.dumps(dict(old=dict(snapshot_phases=2),new=dict(snapshot_phases=2))))
            store=root/'shared-training-snapshots';store.mkdir()
            for i in range(60):(store/f'{i:064x}').mkdir()
            run=study/'runs/old/attempt-001/pretraining';run.mkdir(parents=True)
            (run/'RUN.json').write_text(json.dumps(dict(cell=dict(id='old'))))
            (run/'SNAPSHOT_BINDING.json').write_text(json.dumps(dict(host=socket.gethostname(),run=str(run),
                run_sha256=owner.sha(run/'RUN.json'),snapshot_directory=str(store/f'{1:064x}'))))
            observed=dict(native_jobs=[dict(cell=dict(id='old'),snapshot_directory=str(store/f'{0:064x}'))])
            with patch.object(owner,'observe',return_value=observed):
                self.assertTrue(owner.capacity(study,'new')['can_start'])
                (store/f'{60:064x}').mkdir()
                self.assertTrue(owner.capacity(study,'new')['can_start'])
                (store/f'{61:064x}').mkdir()
                result=owner.capacity(study,'new');self.assertFalse(result['can_start'])
                self.assertEqual(result['pending_snapshot_slots'],2)
                self.assertEqual(result['pending_cpu_claims'],1)

    def phase(self,root):
        parent=root/'attempt-001';run=parent/'fit';run.mkdir(parents=True)
        cell=dict(stage='outer',family='relu',id='cell')
        (parent/'RUN.json').write_text(json.dumps(dict(cell=cell,build_sha256='native',schedule=dict(phase_epochs=[480]))))
        native=['/native/zensim_mlp_train','--epochs','480','--out',str(run/'best.bin')]
        identity=dict(cell=cell,build_sha256='native',stage='running',epochs=480,snapshot_phase=True,
            parent_run_sha256=owner.sha(parent/'RUN.json'),command=['run-heavy','--mem','4G','--',*native])
        (run/'RUN.json').write_text(json.dumps(identity))
        return parent,run,identity,native

    def test_long_fit_binding_uses_verified_phase_and_actual_native_command(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);parent,run,identity,native=self.phase(root)
            observed=dict(native_jobs=[dict(run=str(parent),snapshot_run=str(run),command=native,
                snapshot_directory=str(root/'shared-training-snapshots'/('a'*64)))])
            with patch.object(owner,'observe',return_value=observed):
                value=owner.bind(root,run,run/'SNAPSHOT_BINDING.json')
            self.assertEqual(value['run'],str(run))
            self.assertEqual(value['native_observation']['run'],str(parent))

    def test_wrong_budget_or_native_process_cannot_bind_a_phase(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);parent,run,identity,native=self.phase(root)
            wrong=copy.deepcopy(identity);wrong['epochs']=240
            with self.assertRaisesRegex(ValueError,'exposure differs'):owner.phase_authority(run,wrong)
            observed=dict(native_jobs=[dict(run=str(parent),snapshot_run=str(run),command=native+['--changed'],
                snapshot_directory=str(root/'shared-training-snapshots'/('a'*64)))])
            with patch.object(owner,'observe',return_value=observed):
                with self.assertRaisesRegex(ValueError,'native command differs'):owner.bind(root,run,run/'SNAPSHOT_BINDING.json')
            self.assertFalse((run/'SNAPSHOT_BINDING.json').exists())
            (parent/'RUN.json').write_text('changed parent')
            with self.assertRaises((ValueError,json.JSONDecodeError)):owner.phase_authority(run,identity)


if __name__=='__main__':unittest.main()
