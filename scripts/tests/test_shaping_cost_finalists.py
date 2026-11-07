from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix
from shaping_cost_finalists import coverage


class FinalistTests(unittest.TestCase):
    def test_full_role_seed_coverage_cannot_be_replaced_by_extra_models(self):
        selected=coverage(matrix());self.assertEqual(len(selected),16)
        with self.assertRaisesRegex(ValueError,'sixteen'):coverage(selected[:-1])
        duplicate=selected[:-1]+[selected[0]]
        with self.assertRaisesRegex(ValueError,'sixteen'):coverage(duplicate)
        changed=[dict(c) for c in selected];changed[0]['seed']=7
        with self.assertRaisesRegex(ValueError,'sixteen'):coverage(changed)


if __name__=='__main__':unittest.main()
