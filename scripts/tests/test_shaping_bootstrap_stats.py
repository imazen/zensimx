import os,sys,unittest
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_bootstrap_stats import paired_stratum


class BootstrapStatsTests(unittest.TestCase):
    def fixture(self):
        target=np.array([0,1,1,3,8,5,5,2,9,3,4,7],dtype=float)
        candidate=np.array([0,1,2,1,8,6,4,3,8,3,5,6],dtype=float)
        return np.array([candidate,candidate.copy(),-candidate]),target,[np.arange(4),np.arange(4,8),np.arange(8,12)],np.tile(np.array([[0,1,2],[2,2,0],[1,1,1],[2,0,2]]),(9,1))

    def test_native_matches_independent_tied_rank_oracle_and_pairing(self):
        # Caller supplies the frozen native executable. Missing binaries fail.
        if not Path(os.environ['ZEN_PANEL_BIN']).is_file():raise ValueError('native panel binary missing')
        predictions,target,groups,draws=self.fixture()
        actual=paired_stratum(predictions,target,groups,draws)
        for k,pred in enumerate(predictions):
            for b,draw in enumerate(draws):
                ids=np.concatenate([groups[i] for i in draw])
                pooled=spearmanr(pred[ids],target[ids]).statistic
                within=np.mean([spearmanr(pred[groups[i]],target[groups[i]]).statistic for i in draw])
                self.assertAlmostEqual(actual['pooled'][k,b],pooled,places=9)
                self.assertAlmostEqual(actual['within'][k,b],within,places=9)
        np.testing.assert_array_equal(actual['composite'][0],actual['composite'][1])
        np.testing.assert_allclose(actual['composite'][0],-actual['composite'][2],atol=1e-9)
        self.assertAlmostEqual(actual['point'][0,0],spearmanr(predictions[0],target).statistic,places=9)

    def test_missing_or_repeated_source_rows_are_refused(self):
        pred,y,groups,draws=self.fixture()
        for bad in [groups[:-1],[groups[0],groups[0],groups[2]]]:
            with self.assertRaisesRegex(ValueError,'partition every row'):paired_stratum(pred,y,bad,draws)

    def test_bad_draws_and_nonfinite_scores_are_refused(self):
        pred,y,groups,draws=self.fixture()
        for bad in [draws.astype(float),draws[:,:2],np.full_like(draws,-1),np.full_like(draws,3)]:
            with self.assertRaises(ValueError):paired_stratum(pred,y,groups,bad)
        pred[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):paired_stratum(pred,y,groups,draws)

    def test_native_row_drops_and_identity_drift_are_refused(self):
        pred,y,groups,draws=self.fixture()
        for failure in ('drop','identity','undefined'):
            def corrupt(bases,jobs,stats):
                return [dict(label=('wrong' if failure=='identity' else j[0]),n_dropped=int(failure=='drop'),srocc_signed=float('nan') if failure=='undefined' else 1.) for j in jobs]
            with self.assertRaises(ValueError):paired_stratum(pred,y,groups,draws,panel=corrupt)


if __name__=='__main__':unittest.main()
