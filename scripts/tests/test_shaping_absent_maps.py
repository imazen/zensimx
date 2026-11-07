from pathlib import Path
import sys,unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaped_linear_study import mapped_labels,metrics


class AbsentMapTests(unittest.TestCase):
    def test_unmapped_entries_are_explicit_nan_not_uninitialized_memory(self):
        result=mapped_labels(np.array([2.,7.]),np.array(['a','b']),{'a':{'values':[0.,4.]}})
        self.assertEqual(result[0],.5);self.assertTrue(np.isnan(result[1]))
    def test_default_refuses_missing_map_and_optin_uses_rank_only(self):
        y=np.array([1.,2.,3.,4.]);pop=np.array(['a','a','b','b']);ids=np.arange(4)
        d=dict(pop=pop,component=np.array(['s1','s1','s2','s2']),family=np.array(['f']*4))
        maps={'a':{'values':[0.,4.]}}
        with self.assertRaises(ValueError):metrics(y,y,ids,d,maps)
        def native(data,jobs,**kwargs):
            self.assertNotIn('b|mapped',[j[0] for j in jobs]);self.assertNotIn('b|unmapped',[j[0] for j in jobs])
            self.assertIn('b|pooled',[j[0] for j in jobs]);self.assertIn('b|source|s2',[j[0] for j in jobs])
            return [dict(label=j[0],srocc_signed=1.,mae_raw=0.,n_dropped=0) for j in jobs]
        with patch('shaped_linear_study.panel_batch_indexed',native):report=metrics(y,y,ids,d,maps,allow_absent_maps=True)
        self.assertEqual(report['summary']['b']['pooled'],1.)
        self.assertIsNone(report['summary']['b']['mapped_mae']);self.assertIsNone(report['summary']['b']['unmapped_mae'])


if __name__=='__main__':unittest.main()
