"""Registration must reject leakage and retain paired controls/conditional work."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'rev4_featpot/shaping_completion.py'
spec = importlib.util.spec_from_file_location('shaping_completion', path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class RegistrationTests(unittest.TestCase):
    def test_alias_component_and_development_leak_refuse(self):
        for component, train, test, dev in [
            (['a','a','b'],[0],[1],[2]),
            (['a','b','a'],[0],[1],[2]),
            (['a','b','c'],[0,2],[1],[2]),
            (['a','b','c'],[0],[0,1],[2]),
        ]:
            with self.assertRaises(ValueError):
                study.validate_partition(component,train,test,dev)
        study.validate_partition(['a','b','c'],[0],[1],[2])

    def test_equal_search_and_paired_seeds(self):
        cells=study.matrix()
        for partition in ('inner-0-0','inner-2-1'):
            expected={(setting,seed) for setting in range(4) for seed in range(3)}
            for family in (*study.NEURAL,'extra_trees'):
                got={(c['setting'],c['seed']) for c in cells if c['stage']=='inner'
                     and c['family']==family and c['partition']==partition}
                self.assertEqual(got,expected)
        self.assertTrue(all(c['seed'] is None for c in cells if c['family']=='shaped_additive'))
        self.assertEqual(len({c['id'] for c in cells}),len(cells))

    def test_rescue_is_equal_opportunity_and_not_silently_required(self):
        cells=study.matrix()
        for family in study.NEURAL:
            selected=[c for c in cells if c['family']==family and c['stage']=='rescue']
            self.assertEqual({c['rescue'] for c in selected},set(study.RESCUES))
            self.assertTrue(all('condition' in c for c in selected))
        pilot=[c for c in cells if c['stage']=='resource_pilot']
        self.assertEqual(len(pilot),1)
        self.assertFalse(pilot[0]['scientific_selection'])


if __name__=='__main__':unittest.main()
