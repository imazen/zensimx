import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
from shaping_resume_supervisor import discover


class ResumeSupervisorTests(unittest.TestCase):
    def test_excluded_host_and_ambiguous_control_requests_cannot_enter_collection(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            root=Path(temp);study=root/"study";dispatch=root/"dispatch"
            a=study/"downstream_assignments";a.mkdir(parents=True)
            def write(name,host):
                p=dispatch/name/"jobs/c/REQUEST.json";p.parent.mkdir(parents=True)
                p.write_text(json.dumps(dict(schema="shaping-native-worker-v2",execution_kind="control",
                    cell=dict(id="c",stage="outer"),host=host,source_root=str(root/"frozen"))))
            write("one","i134")
            (a/"outer-c-i134.json").write_text(json.dumps(dict(host="i134",cells=["c"])))
            jobs=discover(study,dispatch);self.assertEqual(set(jobs),{"c"})
            write("two","i134")
            with self.assertRaisesRegex(ValueError,"ambiguous"):discover(study,dispatch)
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            root=Path(temp);study=root/"study";dispatch=root/"dispatch"
            p=dispatch/"one/jobs/c/REQUEST.json";p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(schema="shaping-native-worker-v2",execution_kind="outer",
                cell=dict(id="c",stage="outer"),host="dev",source_root=str(root/"frozen"))))
            with self.assertRaisesRegex(ValueError,"excluded"):discover(study,dispatch)
