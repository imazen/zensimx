"""A stale runtime must refuse before any corrected downstream fit is created."""
import json,sys,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_rescue_execution as rescue
import shaping_outer_execution as outer
import shaping_variant_execution as variant


class CorrectedDownstreamTests(unittest.TestCase):
    def test_old_binary_is_rejected_by_all_three_execution_owners(self):
        for owner in (rescue,outer,variant):
            with self.subTest(owner=owner.__name__),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);binaries=root/'bin';binaries.mkdir()
                (binaries/'BUILD.json').write_text(json.dumps(dict(binary_sha256={})))
                protocol=root/'corrected_thresholds';protocol.mkdir()
                for name in ('REGISTRATION.json','ACTIVATION.json'):(protocol/name).write_text('{}')
                family='product_pretrained_joint'
                cell=dict(id='candidate',family=family,partition='inner-0-0' if owner is rescue else 'outer-0',
                          stage='rescue' if owner is rescue else 'outer' if owner is outer else 'human_only')
                if owner is variant:cell['family']='best_shaped'
                args=SimpleNamespace(study=root,binary_dir=binaries,cell=cell['id'],outer=0,attempt=1)
                resolution=dict(cell=cell,resolved_setting=0,resolved_family=family,schedule={})
                with ExitStack() as stack:
                    stack.enter_context(patch.object(owner,'load_registration',return_value=({},[cell],{})))
                    stack.enter_context(patch('shaping_reduced_scope.admit_rescue'))
                    stack.enter_context(patch('shaping_reduced_scope.admit_cell'))
                    process=stack.enter_context(patch.object(owner.subprocess,'Popen'))
                    if owner is outer:
                        stack.enter_context(patch.object(owner,'resolve',return_value=({'additional_cells':[]},root/'admission.json')))
                        stack.enter_context(patch.object(owner,'resolve_cell',return_value={}))
                        stack.enter_context(patch.object(owner,'prepared'))
                    else:stack.enter_context(patch.object(owner,'resolve',return_value=resolution))
                    with self.assertRaisesRegex(ValueError,'registered corrected native build'):
                        (owner.execute if owner is rescue else owner.admit)(args)
                    process.assert_not_called()
                self.assertFalse((root/'runs').exists())


if __name__=='__main__':unittest.main()
