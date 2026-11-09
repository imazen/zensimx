import os
from pathlib import Path
import subprocess
import sys
import unittest


class ArtifactRetentionDependencies(unittest.TestCase):
    def test_snapshot_cleanup_imports_without_site_packages(self):
        source=Path(__file__).resolve().parents[1]/'rev4_featpot'
        program='import sys;sys.path.insert(0,'+repr(str(source))+');import shaping_snapshot_retention,shaping_snapshot_batch,shaping_artifact_retention;assert "numpy" not in sys.modules'
        result=subprocess.run([sys.executable,'-I','-S','-c',program],capture_output=True,text=True,
                              env=dict(os.environ,TMPDIR=str(Path.home()/'tmp')))
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
