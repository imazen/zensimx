from pathlib import Path
import sys, unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_cost_memory import peak_rss


class MemoryTests(unittest.TestCase):
    def test_gnu_time_requires_successful_single_process_observation(self):
        text = '\tMaximum resident set size (kbytes): 8192\n\tExit status: 0\n'
        self.assertEqual(peak_rss(text), 8192)
        for invalid in (text+text, text.replace('8192', '0'), text.replace('status: 0', 'status: 1'),
                        'peak-RSS 0.02GiB', text.replace('kbytes', 'bytes')):
            with self.assertRaises(ValueError): peak_rss(invalid)


if __name__ == '__main__': unittest.main()
