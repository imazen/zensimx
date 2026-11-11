import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
from shaping_completion import encoded, sha
from shaping_continuation_finalize import finalize


class ContinuationFinalizationTests(unittest.TestCase):
    def fixture(self, base):
        study=base/"study";root=study/"runs/c/attempt-001";phase=root/"fit"
        phase.mkdir(parents=True);(phase/"checkpoints").mkdir()
        native=base/"native";(native/"checkpoints").mkdir(parents=True)
        capture=base/"capture/CAPTURE.json";capture.parent.mkdir()
        cell=dict(id="c",family="relu",partition="outer-2",stage="outer",seed=0)
        initial=dict(cell=cell,epochs=120,build_sha256="build",execution_host="r7900x")
        (root/"RUN.json").write_bytes(encoded(initial))
        command=["trainer","--epochs","120","--out",str(phase/"best.bin"),"--dump-checkpoints-dir",str(phase/"checkpoints")]
        p=dict(cell=cell,build_sha256="build",epochs=120,pairs_per_epoch=50000,
               parent_run_sha256=sha(root/"RUN.json"),command=["run-heavy","--",*command],manifest_sha256={})
        (phase/"RUN.json").write_bytes(encoded(p))
        (phase/"train.log").write_text("".join(f"  epoch {i} | loss=1\n" for i in range(60)))
        (phase/"best.bin").write_bytes(b"old-best");(phase/"best.bin.spec.json").write_text("{}")
        receipts=[]
        for i in range(120):
            model=native/"checkpoints"/f"ckpt_epoch{i:03}.bin";model.write_bytes(f"model-{i}".encode())
            row=dict(epoch=i,path=str(model),bytes=model.stat().st_size,sha256=sha(model))
            receipts.append("  checkpoint dump: "+json.dumps(row))
            if i<60:shutil.copyfile(model,phase/"checkpoints"/model.name)
        snapshot=capture.parent/"identity/source.snap";snapshot.parent.mkdir()
        state=dict(identity="identity",epoch=60,pair=0,checkpoint_receipts=receipts[:60])
        payload=encoded(state);snapshot.write_bytes((60).to_bytes(8,"little")+hashlib.sha256(payload).digest()+payload)
        cap=dict(native=dict(command=command),exact_state_checksum_verified=True,
                 selected=dict(captured=str(snapshot),sha256=sha(snapshot),epoch=60))
        capture.write_bytes(encoded(cap))
        resumed=list(command);resumed[resumed.index("--out")+1]=str(native/"best.bin")
        resumed[resumed.index("--dump-checkpoints-dir")+1]=str(native/"checkpoints")
        done=dict(cell=cell,stage="native_complete_pending_scientific_collection",exit_code=0,
                  completed_epochs_replayed=0,execution_host="i134",source_host="r7900x",
                  source_capture_sha256=sha(capture),snapshot_sha256=sha(snapshot),
                  source_epoch=60,source_pair=0,build_sha256="build",command=resumed)
        (native/"RESULT.json").write_bytes(encoded(done))
        lines="".join(f"  epoch {i} | loss=1\n" for i in range(60,120))
        (native/"train.log").write_text(lines+lines+"\n".join(receipts)+"\n")
        (native/"best.bin").write_bytes(b"new-best");(native/"best.bin.spec.json").write_text("{}")
        shutil.copytree(root,capture.parent/"original-attempt")
        return study,root,capture,native

    def test_exact_state_finalization_preserves_initial_identity_and_raw_logs(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            study,root,cap,native=self.fixture(Path(temp));pin=sha(root/"RUN.json")
            with patch("shaping_continuation_finalize.socket.gethostname",return_value="i134"):
                result=finalize(study,cap,native)
            self.assertEqual(result["completed_epochs_replayed"],0)
            self.assertEqual(sha(root/"RUN.json"),pin)
            self.assertEqual((root/"fit/best.bin.before-continuation").read_bytes(),b"old-best")
            phase=json.loads((root/"fit/RESULT.json").read_text())
            self.assertEqual(phase["bake_sha256"],sha(native/"checkpoints/ckpt_epoch119.bin"))
            self.assertEqual(len(list((root/"fit/checkpoints").glob("*.bin"))),120)
            log=(root/"fit/train.log").read_text()
            self.assertEqual(sum(" | loss=" in line for line in log.splitlines()),120)
            self.assertEqual((root/"continuation-evidence/native-train.log").read_bytes(),(native/"train.log").read_bytes())

    def test_checkpoint_drift_never_publishes_completion(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            study,root,cap,native=self.fixture(Path(temp))
            (native/"checkpoints/ckpt_epoch119.bin").write_bytes(b"changed")
            with patch("shaping_continuation_finalize.socket.gethostname",return_value="i134"):
                with self.assertRaisesRegex(ValueError,"checkpoint drift"):finalize(study,cap,native)
            self.assertFalse((root/"RESULT.json").exists())
            self.assertFalse((root/"CONTINUATION.json").exists())
            self.assertEqual((root/"fit/best.bin").read_bytes(),b"old-best")

    def test_interrupted_result_receipts_are_preserved_on_success(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            study,root,cap,native=self.fixture(Path(temp))
            for directory in (root,root/"fit"):
                initial=json.loads((directory/"RUN.json").read_text())
                (directory/"RESULT.json").write_bytes(encoded(dict(initial,stage="failed",exit_code=143)))
            with patch("shaping_continuation_finalize.socket.gethostname",return_value="i134"):
                finalize(study,cap,native)
                # A crash after publishing only one receipt can reconcile
                # without copying or recomputing any checkpoint.
                result=finalize(study,cap,native)
                self.assertTrue(result["publication_reconciled"])
            for directory in (root,root/"fit"):
                self.assertEqual(json.loads((directory/"RESULT.interrupted.json").read_text())["exit_code"],143)
                self.assertEqual(json.loads((directory/"RESULT.json").read_text())["exit_code"],0)

    def test_one_uncaptured_tail_epoch_requires_exact_raw_checkpoint_equality(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            study,root,cap,native=self.fixture(Path(temp))
            original=cap.parent/"original-attempt/fit"
            shutil.copyfile(native/"checkpoints/ckpt_epoch060.bin",original/"checkpoints/ckpt_epoch060.bin")
            with (original/"train.log").open("a") as f:f.write("  epoch 60 | loss=1\n")
            with patch("shaping_continuation_finalize.socket.gethostname",return_value="i134"):
                result=finalize(study,cap,native)
            self.assertEqual(result["completed_epochs_replayed"],1)
            proof=json.loads((root/"CONTINUATION.json").read_text())
            self.assertEqual(proof["committed_snapshot_epochs_replayed"],0)
            self.assertEqual(proof["recomputed_uncaptured_tail_epochs"],[60])
            self.assertEqual(sum(" | loss=" in line for line in (root/"fit/train.log").read_text().splitlines()),120)
