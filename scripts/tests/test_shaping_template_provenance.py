import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
from shaping_completion import encoded, sha
from shaping_template_provenance import require_parent


class TemplateProvenanceTests(unittest.TestCase):
    def test_inactive_parent_still_needs_frozen_base_evidence_and_native_run_pin(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            s=Path(temp);p=s/"parent";p.mkdir()
            cell=dict(id="parent",family="separate_joint_identity",stage="inner",partition="inner-1-0")
            (p/"RUN.json").write_bytes(encoded(dict(cell=cell)))
            a=dict(outer=1,decision_sha256={"base":"base-pin"})
            with patch("shaping_execution.load_registration",return_value=({},[cell],{})), \
                 patch("shaping_outer_admission.base_decision",return_value=({"reported_cells":["parent"]},{},{"base":"base-pin"})), \
                 patch("shaping_rescue_selection.verified_candidate",return_value=({},None,{str(p/"RUN.json"):sha(p/"RUN.json")})) as verified:
                require_parent(s,a,p);verified.assert_called_once()
                with self.assertRaisesRegex(ValueError,"base evidence"):
                    require_parent(s,dict(a,decision_sha256={}),p)
                verified.return_value=({},None,{})
                with self.assertRaisesRegex(ValueError,"native run pin"):require_parent(s,a,p)

    def test_direct_admission_pin_drift_is_never_replaced_by_a_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/"tmp") as temp:
            p=Path(temp);(p/"RUN.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"pin drift"):
                require_parent(p,dict(decision_sha256={str(p/"RUN.json"):"old"}),p)
