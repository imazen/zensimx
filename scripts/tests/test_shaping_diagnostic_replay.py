from pathlib import Path
import json,sys,tempfile,unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_diagnostic_replay import canonical_pairs,replay,first_weight_traces,diagnostic_fit_scope

class ReplayTests(unittest.TestCase):
    def test_fresh_control_admission_preserves_scope_and_exposure(self):
        cell=dict(stage='fresh_control_inner',family='relu',id='control')
        original=dict(cell=cell);result=dict(cell=cell,stage='trained')
        diagnostic_fit_scope(original,result,120)
        with self.assertRaisesRegex(ValueError,'120-epoch'):diagnostic_fit_scope(original,result,240)
        for stage in ('outer','full_fit','full_fit_selection'):
            with self.assertRaisesRegex(ValueError,'inner fit'):
                diagnostic_fit_scope(dict(cell=dict(cell,stage=stage)),result,120)
        with self.assertRaisesRegex(ValueError,'result identity'):
            diagnostic_fit_scope(original,dict(result,cell=dict(cell,id='other')),120)
        with self.assertRaisesRegex(ValueError,'120-epoch'):
            diagnostic_fit_scope(dict(cell=dict(cell,family='softplus')),result,120)

    def test_weight_trace_requires_exact_epoch_coverage(self):
        lines=['FIRST_WEIGHT_DIAGNOSTICS '+json.dumps({'epoch':i}) for i in range(3)]
        self.assertEqual(len(first_weight_traces('noise\n'+'\n'.join(lines),3)),3)
        for invalid in (lines[:2],lines+[lines[-1]],list(reversed(lines)),[]):
            with self.assertRaisesRegex(ValueError,'incomplete first-weight'):
                first_weight_traces('\n'.join(invalid),3)

    def test_transient_drift_detected_even_when_final_matches(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);pairs=[]
            for epoch in range(2):
                old=root/f'old{epoch}';new=root/f'new{epoch}'
                old.write_bytes(b'zero' if epoch else b'active')
                new.write_bytes(b'zero')
                pairs.append((epoch,old,new))
            def strip(command,**kwargs):
                source=Path(command[command.index('--in')+1]);dest=Path(command[command.index('--out')+1])
                dest.write_bytes(source.read_bytes())
            with patch('shaping_diagnostic_replay.subprocess.run',side_effect=strip):
                rows=canonical_pairs(root,pairs,root)
            self.assertEqual([r['match'] for r in rows],[False,True])

    def test_replay_requires_storage_reservation_before_launch(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text(json.dumps({'minimum_ssd_free_bytes':30*1024**3}))
            args=SimpleNamespace(study=root,output=root/'replay');held=[]
            @contextmanager
            def reserve(study,dest,minimum):
                self.assertEqual(minimum,30*1024**3);held.append(True)
                try:yield
                finally:held.pop()
            def launch(args):self.assertEqual(held,[True]);return 'launched'
            with patch('shaping_storage.reservation',side_effect=reserve),patch('shaping_diagnostic_replay.admitted_replay',side_effect=launch):
                self.assertEqual(replay(args),'launched')
                args.output.mkdir()
                with self.assertRaisesRegex(ValueError,'preserve existing'):replay(args)
            self.assertFalse(held)
