from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import shaping_placement as owner


class PlacementTests(unittest.TestCase):
    def test_storage_and_memory_headroom_exclude_hosts_without_changing_limits(self):
        rows = dict(a=dict(occupied=0, memory_available_bytes=5 * 1024**3, ssd_free_bytes=100 * 1024**3),
                    b=dict(occupied=0, memory_available_bytes=20 * 1024**3, ssd_free_bytes=35 * 1024**3),
                    c=dict(occupied=3, memory_available_bytes=20 * 1024**3, ssd_free_bytes=100 * 1024**3))
        with patch.object(owner, 'observation', side_effect=lambda study, host, capacity: rows[host]):
            loads, measured = owner.loads(Path('study'), dict(a=5, b=14, c=20))
            self.assertEqual(loads['c'], 3)
            self.assertGreater(loads['a'], loads['c'])
            self.assertGreater(loads['b'], loads['c'])
            self.assertEqual(measured, rows)
            with self.assertRaisesRegex(ValueError, 'no worker'):
                owner.loads(Path('study'), dict(a=5, b=14))


if __name__ == '__main__':
    unittest.main()
