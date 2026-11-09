import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_epoch_windows import windows


class EpochWindowTests(unittest.TestCase):
    def test_transition_spanning_intervals_are_excluded(self):
        points=[dict(epoch=i,elapsed=t,mtime=t) for i,t in enumerate([0,10,20,30,40,50,60])]
        result=windows(points,5,25,45)
        self.assertEqual(result['original']['intervals'],[dict(from_epoch=1,to_epoch=2,seconds=10)])
        self.assertEqual(result['priority']['intervals'],[dict(from_epoch=5,to_epoch=6,seconds=10)])
        self.assertEqual(result['original']['median_seconds'],10)

    def test_missing_epoch_is_not_invented_and_empty_window_has_no_estimate(self):
        result=windows([],0,1,2);self.assertIsNone(result['priority']['median_seconds']);self.assertEqual(result['priority']['n'],0)
        with self.assertRaisesRegex(ValueError,'ordered epoch'):
            windows([dict(epoch=0,elapsed=0,mtime=0),dict(epoch=2,elapsed=20,mtime=20)],0,1,2)


if __name__=='__main__':unittest.main()
