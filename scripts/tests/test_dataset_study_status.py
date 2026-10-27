import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"byconstruction"))
import dataset_study_status as progress


class StudyProgress(unittest.TestCase):
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
                (p/"LAUNCH_FAILED.json").write_text(json.dumps(dict(error="invented failure")))
                self.assertEqual(progress.status(p)["status"],"failed")


if __name__=="__main__":unittest.main()
