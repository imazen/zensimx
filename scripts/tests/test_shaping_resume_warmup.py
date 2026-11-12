import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
import shaping_rescue_dispatch as owner

class ResumeWarmupTests(unittest.TestCase):
    def test_only_the_first_new_resume_checkpoint_releases_warmup(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            base=Path(temp);study=base/"study";study.mkdir()
            point=base/"continuation/checkpoints/ckpt_epoch330.bin";point.parent.mkdir(parents=True)
            (study/"cpu-admission").mkdir()
            (study/"cpu-admission/CLAIMS.json").write_text(json.dumps(dict(old=dict(
                warmup_until_checkpoint=True,warmup_checkpoint=str(point)))))
            def observation(*a,**k):
                return dict(can_start=True,available_memory_bytes=20*1024**3,
                    minimum_available_memory_bytes=6*1024**3,native_threads=0,pending_dispatcher_threads=0)
            with patch.object(owner,"pending_scientific_cells",return_value=set()),patch.object(owner,"worker_slot",side_effect=observation):
                before=owner.reserve_worker(study,"before")
                point.write_bytes(b"completed native checkpoint")
                after=owner.reserve_worker(study,"after")
            self.assertEqual(before["minimum_available_memory_bytes"],8*1024**3)
            self.assertEqual(after["minimum_available_memory_bytes"],6*1024**3)
