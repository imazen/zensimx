"""Family separation and protected aliases must survive split selection."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "byconstruction"))
from dataset_keys import hdr_source_partition


class HdrMembershipTests(unittest.TestCase):
    def test_related_frames_never_cross_roles(self):
        sources = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
        roles = hdr_source_partition(sources, {})
        self.assertEqual(roles["a1"], roles["a2"])
        self.assertEqual(roles["b1"], roles["b2"])
        self.assertNotEqual(roles["a1"], roles["b1"])
        self.assertEqual(roles, hdr_source_partition(dict(reversed(list(sources.items()))), {}))

    def test_reservation_covers_other_frame(self):
        sources = {"a1": "a", "a2": "a", "b": "b", "c": "c"}
        roles = hdr_source_partition(sources, {"a1": "protected elsewhere"})
        self.assertEqual(roles["a2"], "EVAL")
        self.assertEqual(roles["a1"], "EVAL")

    def test_refuse_unsplittable_or_unknown_reservation(self):
        with self.assertRaises(ValueError):
            hdr_source_partition({"a1": "a", "a2": "a"}, {})
        with self.assertRaises(ValueError):
            hdr_source_partition({"a": "a", "b": "b"}, {"missing": "reserved"})

    def test_nearest_ratio_without_family_fragmentation(self):
        sources = {str(i): str(i) for i in range(10)}
        self.assertEqual(list(hdr_source_partition(sources, {}).values()).count("TRAIN"), 7)


if __name__ == "__main__":
    unittest.main()
