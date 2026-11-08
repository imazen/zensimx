import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from scripts.tests.test_shaping_corrected_registration import CorrectedRegistrationTests
import shaping_corrected_view as view
from shaping_selection import aggregate


class CorrectedViewTests(unittest.TestCase):
    def test_only_pretrained_ids_change_and_old_reports_cannot_select(self):
        cells,matrix=CorrectedRegistrationTests().fixture();effective=view.corrected_cells(cells,matrix)
        self.assertEqual(len(cells),len(effective))
        changed=[(old,new) for old,new in zip(cells,effective) if old!=new]
        self.assertEqual(len(changed),288)
        for old,new in changed:
            self.assertEqual({k:v for k,v in old.items() if k!='id'},{k:v for k,v in new.items() if k!='id'})
        family='product_pretrained_frozen'
        reports={}
        for cell in cells:
            if cell['stage']!='inner' or cell['family']!=family or cell['partition'] not in ('inner-0-0','inner-0-1'):continue
            reports[cell['id']]=dict(cell=cell,outer_results_read=False,inner_validation=dict(
                selection_score=.5,summary={p:dict(pooled=.5,within=.5) for p in ('kadid','tid2013','konfig','cid22_a25')}))
        with self.assertRaisesRegex(ValueError,'missing or extra paired reports'):aggregate(effective,reports,family,0)
        mapping={old['id']:new for old,new in changed}
        reports={mapping[cid]['id']:dict(row,cell=mapping[cid]) for cid,row in reports.items()}
        decision=aggregate(effective,reports,family,0)
        self.assertEqual(decision['required_reports'],24);self.assertEqual(decision['selected_setting'],0)

    def proof(self,family='relu'):
        case=dict(cell=dict(id='original',family=family),original_final_sha256='a'*64)
        rows={f'ckpt_epoch{epoch:03}.bin':dict(equal=True,input_sha256=dict(original='a'*64,neutral='b'*64),
                    canonical_sha256=dict(original='c'*64,neutral='c'*64)) for epoch in range(240)}
        comparison=dict(required=240,all_equal=True,stripped_metadata_keys=['zentrain.repro'],compared=rows)
        proof=dict(cell=case['cell'],outer_results_read=False,build_sha256=view.BUILD_SHA256,
            stage='qualified',scientific_selection=False,epochs=240,pairs_per_epoch=50000,
            all_240_checkpoints_equal=True,trajectory=comparison)
        return case,proof

    def test_terminal_all_epoch_identity_required(self):
        case,proof=self.proof();view.trajectory(proof,'relu',case)
        for change in ('missing','mismatch','scope','original','running','exposed','build'):
            with self.subTest(change=change):
                broken=copy.deepcopy(proof);comparison=broken['trajectory']
                if change=='missing':del comparison['compared']['ckpt_epoch001.bin']
                if change=='mismatch':comparison['compared']['ckpt_epoch001.bin']['canonical_sha256']['neutral']='d'*64
                if change=='scope':comparison['stripped_metadata_keys'].append('zentrain.input_shaping')
                if change=='original':comparison['compared']['ckpt_epoch239.bin']['input_sha256']['original']='d'*64
                if change=='running':broken['stage']='running'
                if change=='exposed':broken['outer_results_read']=True
                if change=='build':broken['build_sha256']='d'*64
                with self.assertRaises(ValueError):view.trajectory(broken,'relu',case)

    def test_analytic_requires_research_metadata_scope(self):
        case,proof=self.proof('softplus')
        with self.assertRaises(ValueError):view.trajectory(proof,'softplus',case)
        proof['trajectory']['stripped_metadata_keys'].append('zentrain.monotone_research')
        view.trajectory(proof,'softplus',case)


if __name__=='__main__':unittest.main()
