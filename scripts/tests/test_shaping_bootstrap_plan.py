import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_bootstrap_plan import population_groups,component_draws,expand


class SourceBootstrapTests(unittest.TestCase):
    def test_entire_sources_and_repeat_multiplicity(self):
        groups=population_groups([0,1,2,3,4],['human']*5,['a','a','b','b','b'])['human']
        self.assertEqual(expand(groups,[1,1]).tolist(),[2,3,4,2,3,4])
        self.assertEqual(expand(groups,[0,1]).tolist(),[0,1,2,3,4])

    def test_cross_population_source_is_refused(self):
        with self.assertRaisesRegex(ValueError,'spans populations'):
            population_groups([0,1],['human','teacher'],['shared','shared'])

    def test_unselected_rows_do_not_enter_draws(self):
        groups=population_groups([1,3],['outside','human','outside','human'],['a','b','a','c'])['human']
        self.assertEqual(expand(groups,[0,1]).tolist(),[1,3])
        with self.assertRaises(ValueError):population_groups([1,1],['human']*2,['a','b'])

    def test_stream_is_reproducible_without_resetting_between_strata(self):
        groups={'a':[0],'b':[1],'c':[2]};rng=np.random.default_rng(20261107)
        first=component_draws(groups,rng,20);second=component_draws(groups,rng,20)
        np.testing.assert_array_equal(first,component_draws(groups,np.random.default_rng(20261107),20))
        self.assertFalse(np.array_equal(first,second))
        self.assertEqual(first.shape,(20,3));self.assertTrue(((first>=0)&(first<3)).all())


if __name__=='__main__':unittest.main()


class MatchedRegistrationTests(unittest.TestCase):
    def test_complete_registration_and_specification_drift_refused(self):
        import json,tempfile
        from unittest.mock import patch
        from shaping_completion import matrix
        from shaping_bootstrap_plan import matched_specs,register_matched,load_comparison
        self.assertEqual(len(matched_specs(matrix())),88)
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study=root/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
            with patch('shaping_bootstrap_plan.load_registration',return_value=({},matrix(),{})):
                record=register_matched(study,root/'comparisons')
                path=root/'comparisons'/record['comparisons'][0]['file']
                spec,pins=load_comparison(study,path)
                self.assertEqual(spec,record['comparisons'][0]['spec']);self.assertIn(str(path),pins)
                other=root/'comparisons'/record['comparisons'][-1]['file'];other.write_text('{}')
                with self.assertRaisesRegex(ValueError,'specification drift'):load_comparison(study,path)
                with self.assertRaises(FileExistsError):register_matched(study,root/'comparisons')
