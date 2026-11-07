import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_training_resources import parse_log


class ResourceTests(unittest.TestCase):
    def log(self):
        return '  epoch 0 | loss=1.0 | t=0.2s\n  epoch 1 | loss=0.5 | t=1.4s\nrun-heavy: done rc=0 2s | peak-RSS 0.09GiB | min-avail 30000MiB | peak-load 28.12\n'

    def test_reported_precision_is_retained(self):
        r=parse_log(self.log(),2)
        self.assertEqual(r['native_elapsed_seconds_by_epoch'],[.2,1.4])
        self.assertEqual(r['wrapper_elapsed_seconds'],2)
        self.assertEqual(r['wrapper_peak_rss_gib_reported'],.09)

    def test_incomplete_duplicate_reordered_and_failed_logs_refused(self):
        for text in [self.log().replace('epoch 1','epoch 0'),self.log().replace('t=1.4','t=0.1'),self.log().replace('rc=0','rc=137'),self.log()+self.log(),self.log().split('run-heavy:')[0]]:
            with self.assertRaises(ValueError):parse_log(text,2)

    def test_identical_progress_blocks_need_one_resource_receipt(self):
        progress=self.log().split('run-heavy:')[0]
        r=parse_log(progress+self.log(),2)
        self.assertEqual(r['identical_epoch_log_blocks'],2)
        self.assertEqual(r['wrapper_elapsed_seconds'],2)
        with self.assertRaisesRegex(ValueError,'nonidentical'):parse_log(progress.replace('loss=1.0','loss=2.0')+self.log(),2)


if __name__=='__main__':unittest.main()
