import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_phase_snapshot_watch as watch
from shaping_completion import encoded,sha


class PhaseSnapshotWatchTests(unittest.TestCase):
    def fixture(self,root):
        study=root/'study';cid='a'*20;run=study/'runs'/cid/'attempt-001';phase=run/'fit';phase.mkdir(parents=True)
        identity=dict(cell=dict(id=cid),stage='running');result=dict(identity,stage='trained',exit_code=0)
        (run/'RUN.json').write_bytes(encoded(identity));(run/'RESULT.json').write_bytes(encoded(result))
        (phase/'RUN.json').write_bytes(encoded(dict(identity,snapshot_phase=True)))
        (phase/'RESULT.json').write_bytes(encoded(result))
        store=root/'shared-training-snapshots'/('b'*64);store.mkdir(parents=True)
        binding=phase/'SNAPSHOT_BINDING.json';binding.write_bytes(encoded(dict(host=socket.gethostname(),snapshot_directory=str(store))))
        paths=['RUN.json','RESULT.json','fit/RUN.json','fit/RESULT.json','fit/SNAPSHOT_BINDING.json']
        receipt=dict(source_host=socket.gethostname(),files={name:dict(sha256=sha(run/name)) for name in paths})
        return study,cid,run,phase,store,binding,receipt

    def test_uncollected_states_are_never_offered_for_retirement(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            study,cid,run,phase,store,binding,receipt=self.fixture(Path(scratch))
            self.assertEqual(watch.collected(study,socket.gethostname()),{})
            (run/'WORKER_COLLECTION.json').write_bytes(encoded(receipt))
            allowed=watch.collected(study,socket.gethostname())
            self.assertEqual(watch.pending(study,allowed)['pending_cells'],[cid])

    def test_worker_drift_is_refused_before_staging(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            study,cid,run,phase,store,binding,receipt=self.fixture(Path(scratch))
            (run/'WORKER_COLLECTION.json').write_bytes(encoded(receipt));allowed=watch.collected(study,socket.gethostname())
            (phase/'RUN.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'differs from collection'):watch.pending(study,allowed)
            self.assertTrue(store.exists())

    def test_missing_state_requires_the_exact_retirement_binding(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            study,cid,run,phase,store,binding,receipt=self.fixture(Path(scratch))
            (run/'WORKER_COLLECTION.json').write_bytes(encoded(receipt));allowed=watch.collected(study,socket.gethostname());store.rmdir()
            with self.assertRaisesRegex(ValueError,'without retirement proof'):watch.pending(study,allowed)
            journal=phase/'SNAPSHOT_RETIREMENT.json';journal.write_bytes(encoded(dict(stage='completed_store_retired',binding_sha256='wrong',snapshot_directory=str(store))))
            with self.assertRaisesRegex(ValueError,'without retirement proof'):watch.pending(study,allowed)
            journal.write_bytes(encoded(dict(stage='completed_store_retired',binding_sha256=sha(binding),snapshot_directory=str(store))))
            self.assertEqual(watch.pending(study,allowed)['pending_cells'],[])

    def test_collected_legacy_store_uses_the_same_retirement_gate(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            study,cid,run,phase,store,binding,receipt=self.fixture(Path(scratch))
            (phase/'RUN.json').write_bytes(encoded(dict(cell=dict(id=cid))))
            root_binding=run/'SNAPSHOT_BINDING.json';root_binding.write_bytes(binding.read_bytes())
            receipt['files']={name:dict(sha256=sha(run/name)) for name in ('RUN.json','RESULT.json','SNAPSHOT_BINDING.json')}
            (run/'WORKER_COLLECTION.json').write_bytes(encoded(receipt))
            allowed=watch.collected(study,socket.gethostname())
            self.assertEqual(allowed[cid]['phases'],[''])
            self.assertEqual(watch.pending(study,allowed)['pending_cells'],[cid])


if __name__=='__main__':unittest.main()
