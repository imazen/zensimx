import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,NEURAL
from shaping_selection_watch import ready,sweep


class SelectionWatchTests(unittest.TestCase):
    def test_final_report_without_diagnostics_is_not_ready_and_ambiguity_fails(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cell=dict(id='c',family='relu')
            run=root/'reexport_max_pairs/c/attempt-001';(run/'final').mkdir(parents=True)
            (run/'final/COMPLETE.json').write_text('{}')
            self.assertFalse(ready(root,cell,'research-relu-max-pairs-v1'))
            (run/'CONVERGENCE.json').write_text('{}')
            self.assertTrue(ready(root,cell,'research-relu-max-pairs-v1'))
            other=run.parent/'attempt-002/final';other.mkdir(parents=True);(other/'COMPLETE.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'ambiguous'):ready(root,cell,'research-relu-max-pairs-v1')

    def test_no_selection_with_missing_rescue_reports(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);selection=root/'selection/outer-0';selection.mkdir(parents=True)
            (root/'RESCUE_TRIGGER_OUTER_0.json').write_text('{}')
            for family in NEURAL:(selection/(family+'-base.json')).write_text(json.dumps(dict(representation='original')))
            with patch('shaping_selection_watch.load_registration',return_value=({},matrix(),{})),patch('shaping_selection_watch.select_rescue') as select,patch('shaping_selection_watch.register_extension') as extension,patch('shaping_selection_watch.register_outer') as outer:
                result=sweep(root,0)
                self.assertEqual(result['completed_rescue_selections'],[])
                select.assert_not_called();extension.assert_not_called();outer.assert_not_called()

    def test_present_markers_still_require_full_provenance_gate(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);selection=root/'selection/outer-0';selection.mkdir(parents=True)
            (root/'RESCUE_TRIGGER_OUTER_0.json').write_text('{}')
            (selection/'relu-base.json').write_text(json.dumps(dict(representation='research-relu-max-pairs-v1')))
            with patch('shaping_selection_watch.load_registration',return_value=({},matrix(),{})),patch('shaping_selection_watch.ready',return_value=True),patch('shaping_selection_watch.select_rescue',side_effect=ValueError('provenance drift')) as select,patch('shaping_selection_watch.register_outer') as outer:
                with self.assertRaisesRegex(ValueError,'provenance drift'):sweep(root,0)
                select.assert_called_once();outer.assert_not_called()


if __name__=='__main__':unittest.main()
