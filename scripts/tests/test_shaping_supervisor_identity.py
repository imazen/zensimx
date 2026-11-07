import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_archive_supervision as archive
import shaping_reexport_supervision as reexport
import shaping_selection_watch as selection


class SupervisorIdentityTests(unittest.TestCase):
    def test_status_keeps_loaded_hash_when_source_path_changes(self):
        for module in (archive,reexport,selection):
            with self.subTest(module=module.__name__),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
                root=Path(temporary)
                changed=root/'changed.py';changed.write_text('# replacement source\n')
                expected=module.DRIVER_SHA256
                self.assertNotEqual(module.sha(changed),expected)
                # Simulate an updated source path while the imported code remains live.
                with patch.object(module,'__file__',str(changed)),patch.object(module,'sweep',return_value={} if module is selection else 0):
                    if module is selection:
                        module.watch(SimpleNamespace(study=root,outer=0,watch=False))
                        output=root/'dispatch/selection/outer-0-STATUS.json'
                    else:
                        argv=['supervisor','--study',str(root)]
                        if module is archive:
                            argv+=['--destination',str(root/'archive')]
                            output=root/'ARCHIVE_SUPERVISOR_STATUS.json'
                        else:
                            argv+=['--binary-dir',str(root/'bin'),'--inspector',str(root/'inspector')]
                            output=root/'REEXPORT_EVALUATOR_STATUS.json'
                        with patch.object(sys,'argv',argv):module.main()
                self.assertEqual(json.loads(output.read_text())['driver_sha256'],expected)


if __name__=='__main__':unittest.main()
