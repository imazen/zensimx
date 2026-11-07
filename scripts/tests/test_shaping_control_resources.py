import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_control_resources import batches

class ControlResources(unittest.TestCase):
    def log(self):
        return '\n'.join(['run-heavy: mem-cap=2G :: python3 scripts/rev4_featpot/shaping_controls.py --partition inner-0-0 --tree', '{"cell":"abc","stage":"running","pid":123}', '{"cell":"abc","stage":"complete"}', 'run-heavy: done rc=0 3s | peak-RSS 0.39GiB | min-avail 30000MiB | peak-load 21.84'])
    def test_distinct_batches_preserve_cost_scope(self):
        rows=batches(self.log()+'\n'+self.log().replace('inner-0-0','inner-0-1'))
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['wrapper_elapsed_seconds'],3)
    def test_partial_failed_duplicate_and_overlapping_evidence_refused(self):
        for text in [self.log().replace('rc=0','rc=137'),self.log().rsplit('\n',1)[0],self.log().replace('{"cell":"abc","stage":"complete"}',''),self.log().replace('"stage":"complete"','"stage":"running","pid":123'),self.log()+'\nrun-heavy: done rc=0 3s | peak-RSS 0.39GiB | min-avail 30000MiB | peak-load 21.84']:
            with self.assertRaises(ValueError):batches(text)
if __name__=='__main__':unittest.main()
