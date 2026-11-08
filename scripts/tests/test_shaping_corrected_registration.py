import copy,hashlib,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_corrected_registration as registration
from shaping_completion import matrix,encoded
from shaping_pretraining import matching_parent


class CorrectedRegistrationTests(unittest.TestCase):
    def fixture(self):
        cells=matrix();entries=[]
        for old in cells:
            if old['stage']!='inner' or old['family'] not in registration.FAMILIES:continue
            entry=dict(schema='corrected-shaped-threshold-condition-v1',original_condition=old,
                parent=matching_parent(old,cells),epochs=120,pairs_per_epoch=50000,
                source_build_sha256=registration.BUILD_SHA256,initializer='quantiles-of-effective-shaped-training-input')
            entry['id']=hashlib.sha256(encoded(entry)).hexdigest()[:20];entries.append(entry)
        record=dict(corrected_pretrained_conditions=entries,settings=[0,1,2,3],paired_seeds=[0,1,2],eligible_inner_cells=816,
            reused_original_primary_controls=[c for c in cells if c['stage']=='inner' and c['family'] not in registration.FAMILIES],
            matched_fresh_controls=[c for c in cells if c['stage']=='fresh_control_inner'])
        return cells,record

    def test_all_four_settings_and_paired_seeds_preserve_exact_parent_controls(self):
        cells,record=self.fixture();result=registration.validate_matrix(record,cells)
        self.assertEqual(len(result),288);self.assertEqual(len(record['reused_original_primary_controls']),528)
        self.assertEqual(len(record['matched_fresh_controls']),72)

    def test_missing_pair_wrong_parent_reused_id_or_changed_budget_refuses(self):
        for change in ('missing','parent','id','budget','control','settings'):
            with self.subTest(change=change):
                cells,record=self.fixture();record=copy.deepcopy(record);entries=record['corrected_pretrained_conditions']
                if change=='missing':entries.pop()
                if change=='parent':entries[0]['parent']=entries[1]['parent']
                if change=='id':entries[1]['id']=entries[0]['id']
                if change=='budget':entries[0]['epochs']=240
                if change=='control':record['matched_fresh_controls'].pop()
                if change=='settings':record['settings']=[0,3]
                with self.assertRaises(ValueError):registration.validate_matrix(record,cells)


if __name__=='__main__':unittest.main()
