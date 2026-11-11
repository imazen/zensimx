import errno,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
import shaping_archive_mount as mount
from shaping_autonomous_finish import job_complete

class AutonomousClosureTests(unittest.TestCase):
    def test_existing_nonterminal_status_never_counts_as_complete(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            p=Path(temp)/"STATUS.json";job=dict(complete_marker=str(p),completion=dict(stage="complete",required=25),completion_lengths=dict(completed=25))
            p.write_text(json.dumps(dict(stage="running",required=25,completed=list(range(25)))))
            self.assertFalse(job_complete(job))
            p.write_text(json.dumps(dict(stage="complete",required=25,completed=list(range(24)))))
            self.assertFalse(job_complete(job))
            p.write_text(json.dumps(dict(stage="complete",required=25,completed=list(range(25)))))
            self.assertTrue(job_complete(job))
    def test_archive_recovery_cannot_replace_an_unowned_mount_or_mask_other_errors(self):
        with patch.object(mount.shutil,"disk_usage",side_effect=OSError(errno.ESTALE,"stale")),patch.object(mount.subprocess,"run") as run:
            with self.assertRaisesRegex(ValueError,"owned study"):mount.ensure(Path("/other"),"export",Path("/unused"),"output/study")
            run.assert_not_called()
        with patch.object(mount.shutil,"disk_usage",side_effect=OSError(errno.ENOSPC,"full")),patch.object(mount.subprocess,"run") as run:
            with self.assertRaises(OSError):mount.ensure(Path("/mnt/tower/output/study"),"export",Path("/unused"),"output/study")
            run.assert_not_called()
