from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaped_linear_compare import include_population


class PopulationAdmissionTests(unittest.TestCase):
    def test_missing_population_requires_explicit_declaration(self):
        with self.assertRaises(ValueError):include_population('teacher',0,())
        self.assertFalse(include_population('teacher',0,('teacher',)))
    def test_declared_absence_cannot_drop_present_rows(self):
        with self.assertRaises(ValueError):include_population('human',1,('human',))
        self.assertTrue(include_population('human',1,()))


if __name__=='__main__':unittest.main()
