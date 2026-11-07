from pathlib import Path
import sys,tempfile,json,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_supervision import sweep
from shaping_evaluation import NativeFeatureGateError
from shaping_completion import encoded

class SupervisionTests(unittest.TestCase):
    def test_failed_gate_remains_failed_while_other_cells_finish(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);cells=[dict(id=str(i),stage='inner',family='relu') for i in range(2)]
            for cell in cells:
                run=root/'runs'/cell['id']/'attempt-001';run.mkdir(parents=True)
                (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',epochs=240)))
            seen=[]
            def evaluate(args):
                seen.append((args.run.parent.name,args.checkpoint))
                if args.run.parent.name=='0' and args.checkpoint==119:
                    dest=args.run/'curve_evaluation/epoch-119';dest.mkdir(parents=True)
                    (dest/'FAILED_GATE.json').write_bytes(encoded(dict(damage_increase_violations=1)))
                    raise NativeFeatureGateError('measured failure')
            with patch('shaping_supervision.load_registration',return_value=({},cells,{})),patch('shaping_supervision.evaluate',side_effect=evaluate),patch('shaping_supervision.report') as report:
                self.assertEqual(sweep(SimpleNamespace(study=root,binary_dir=root/'bin')),1)
                report.assert_called_once_with(root/'runs/1/attempt-001')
            self.assertIn(('0',239),seen);self.assertIn(('1',239),seen)
            failure=json.loads((root/'runs/0/attempt-001/EVALUATION_FAILURES.json').read_text())
            self.assertFalse(failure['scientifically_complete']);self.assertTrue(failure['requires_investigation'])
            self.assertEqual(failure['failures'][0]['checkpoint'],119)

if __name__=='__main__':unittest.main()
