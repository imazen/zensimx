import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_reboot_hold as owner


class RebootHoldTests(unittest.TestCase):
    def run_fixture(self,root):
        from contextlib import ExitStack
        stack=ExitStack();stack.enter_context(patch.object(owner,'CONFIG',root/'hold.conf'))
        stack.enter_context(patch.object(owner,'SCHEDULED',root/'scheduled'))
        stack.enter_context(patch.object(owner.socket,'gethostname',return_value='r3500'))
        stack.enter_context(patch.object(owner.os,'geteuid',return_value=0));return stack

    def test_hold_preserves_updates_and_requires_later_restoration(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            with self.run_fixture(root),patch.object(owner,'settings',side_effect=["ZenResearchAutoReboot='true'\n","ZenResearchAutoReboot='false'\n"]):
                result=owner.hold(root/'journal.json')
            self.assertEqual((root/'hold.conf').read_text(),owner.CONTENT)
            self.assertEqual(result['stage'],'held');self.assertTrue(result['restore_required_at_goal_completion'])
            self.assertFalse(result['updates_disabled']);self.assertFalse(result['training_signalled'])
            self.assertEqual(json.loads((root/'journal.json').read_text()),result)

    def test_existing_policy_and_scheduled_shutdown_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            with self.run_fixture(root),patch.object(owner,'settings',return_value="ZenResearchAutoReboot='true'\n"):
                (root/'hold.conf').write_text('existing policy')
                with self.assertRaisesRegex(ValueError,'preserve existing'):owner.hold(root/'journal.json')
                self.assertEqual((root/'hold.conf').read_text(),'existing policy');(root/'hold.conf').unlink()
                (root/'scheduled').write_text('MODE=reboot\n')
                with self.assertRaisesRegex(ValueError,'separate attribution'):owner.hold(root/'journal.json')
                self.assertFalse((root/'hold.conf').exists());self.assertFalse((root/'journal.json').exists())
                self.assertEqual((root/'scheduled').read_text(),'MODE=reboot\n')

    def test_disabled_policy_is_left_alone(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            with self.run_fixture(root),patch.object(owner,'settings',return_value=''):
                with self.assertRaisesRegex(ValueError,'not explicitly enabled'):owner.hold(root/'journal.json')
            self.assertFalse((root/'hold.conf').exists());self.assertFalse((root/'journal.json').exists())
