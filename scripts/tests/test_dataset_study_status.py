import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"byconstruction"))
import dataset_study_status as progress


class StudyProgress(unittest.TestCase):
    def test_queued_launch_failure_is_durable_and_cannot_report_completion(self):
        import dataset_queue as queue
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/"trainer").mkdir()
            binary=p/"trainer/zensim_mlp_train";binary.write_bytes(b"invented binary")
            pin=queue.sha(binary);commit="a"*40
            (p/"trainer/BUILD.json").write_text(json.dumps(dict(binary_sha256=pin,source_commit=commit)))
            calls=[]
            def execute(command,**kwargs):
                calls.append(command)
                if command[0]=="flock":raise subprocess.CalledProcessError(1,command)
            with mock.patch.object(queue.subprocess,"run",side_effect=execute):
                with self.assertRaises(subprocess.CalledProcessError):queue.launch(p,commit,pin)
            self.assertTrue((p/"LAUNCH_FAILED.json").is_file())
            self.assertFalse((p/"QUEUE_EXIT.json").exists())
            d=json.loads((p/"QUEUED.json").read_text())
            self.assertEqual(d["command"],calls[-1])
            self.assertEqual(calls[-1][0:2],["flock",str(Path.home()/"tmp/heavy.lock")])
            self.assertIn("run-heavy",calls[-1][2])
            self.assertEqual(calls[-1][-3:],[str(p),commit,pin])

    def test_active_population_cannot_reuse_completed_control(self):
        p=Path("/invented/study")
        self.assertEqual(progress.population(p,dict(stage="control"),"MlicVote"),
                         (p/"control-cells","MlicControl","kadid","0"))
        state=dict(stage="cells",command=["run","--label","MlicVote","--folds","pipal,pairs","--seeds","1,2"])
        self.assertEqual(progress.population(p,state,"MlicControl"),(p/"cells","MlicVote","pipal,pairs","1,2"))

    def test_dead_queue_never_reports_running_and_launch_failure_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/"QUEUED.json").write_text(json.dumps(dict(queue_process=dict(pid=1,start_ticks=2),
                source_commit="a"*40,binary_sha256="b"*64,next_action="acquire lock")))
            with mock.patch.object(progress,"bound_worker",return_value=None), mock.patch.object(progress,"compute_lock",return_value={}):
                r=progress.status(p)
                self.assertEqual(r["status"],"queue-exited-without-starting")
                self.assertFalse(r["worker_alive"] or r["objective_complete"])
                (p/"QUEUE_CANCELLED.json").write_text(json.dumps(dict(reason="invented admission defect")))
                self.assertEqual(progress.status(p)["status"],"canceled-before-start")
                (p/"LAUNCH_FAILED.json").write_text(json.dumps(dict(error="invented failure")))
                self.assertEqual(progress.status(p)["status"],"failed")


if __name__=="__main__":unittest.main()
