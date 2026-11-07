from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix
from shaping_rescue_dispatch import base_ready


class DispatchTests(unittest.TestCase):
    def test_all_paired_reports_and_diagnostics_are_required(self):
        cells=matrix()
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d)
            selected=[c for c in cells if c['stage']=='inner' and c['family']=='relu' and c['partition'] in ('inner-0-0','inner-0-1')]
            for c in selected:
                p=root/'reexport_max_pairs'/c['id']/'attempt-001';(p/'final').mkdir(parents=True)
                (p/'final/COMPLETE.json').write_text('{}');(p/'CONVERGENCE.json').write_text('{}')
            self.assertTrue(base_ready(root,cells,'relu',0))
            (p/'CONVERGENCE.json').unlink();self.assertFalse(base_ready(root,cells,'relu',0))
            (p/'CONVERGENCE.json').write_text('{}')
            (p/'final/COMPLETE.json').unlink();self.assertFalse(base_ready(root,cells,'relu',0))
            # These are presence checks only. Native selection separately hashes
            # and validates reports; empty fixtures never qualify a selection.


if __name__=='__main__':unittest.main()
