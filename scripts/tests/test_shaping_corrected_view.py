import copy,sys,unittest,json,tempfile
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from scripts.tests.test_shaping_corrected_registration import CorrectedRegistrationTests
import shaping_corrected_view as view
from shaping_selection import aggregate


class CorrectedViewTests(unittest.TestCase):
    def test_downstream_cannot_restore_raw_coordinate_threshold_runtime(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            root=Path(temporary)
            # Historical and unchanged control families retain existing gates.
            view.require_downstream_build(root,'product_pretrained_joint','historical')
            directory=root/'corrected_thresholds';directory.mkdir()
            (directory/'REGISTRATION.json').write_text('{}')
            for family in view.FAMILIES:
                with self.assertRaisesRegex(ValueError,'awaits native compatibility'):
                    view.require_downstream_build(root,family,view.BUILD_SHA256)
            (directory/'ACTIVATION.json').write_text('{}')
            for family in view.FAMILIES:
                with self.assertRaisesRegex(ValueError,'registered corrected native build'):
                    view.require_downstream_build(root,family,'historical')
                view.require_downstream_build(root,family,view.BUILD_SHA256)
            for family in ('relu','softplus','silu_monotone','silu_diagnostic',
                           'product_joint_identity','separate_joint_identity'):
                view.require_downstream_build(root,family,'independently-qualified')

    def test_pending_correction_cannot_freeze_historical_shortlists_or_outer_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            root=Path(temporary);(root/'corrected_thresholds').mkdir();(root/'corrected_thresholds/REGISTRATION.json').write_text('{}')
            from shaping_outer_admission import build
            from shaping_reduced_scope import shortlist
            from shaping_selection import select
            from types import SimpleNamespace
            for operation in (lambda:build(root,0),lambda:shortlist(root,0,register=True),
                              lambda:select(SimpleNamespace(study=root,family='product_pretrained_joint'))):
                with self.assertRaisesRegex(ValueError,'awaits native compatibility'):operation()
            self.assertFalse((root/'selection').exists());self.assertFalse((root/'reduced_scope').exists())

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

    def test_activation_requires_six_proofs_and_keeps_historical_selections(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)/'study';(root/'corrected_thresholds').mkdir(parents=True)
            (root/'PLAN.json').write_text('{}');(root/'corrected_thresholds/REGISTRATION.json').write_text('{}')
            compatibility=root.parent/'corrected-control-qualification-20261108/REGISTRATION.json'
            compatibility.parent.mkdir();cases={};proofs={}
            for family in view.CONTROL_CLASSES:
                case,proof=self.proof(family);run=root/'runs'/family;run.mkdir(parents=True)
                (run/'RESULT.json').write_text('{}');case.update(run=str(run),original_build_sha256='old-build')
                if family in ('softplus','silu_monotone','silu_diagnostic'):
                    proof['trajectory']['stripped_metadata_keys'].append('zentrain.monotone_research')
                if family=='silu_monotone':
                    comparison=proof.pop('trajectory');proof.update(comparison,schema='native-control-trajectory-parity-v1',selection_candidate=False)
                    proof['original_result_sha256']=view.sha(run/'RESULT.json')
                else:proof['source_result_sha256']=view.sha(run/'RESULT.json')
                cases[family]=case;proofs[family]=root.parent/(family+'.json')
                proofs[family].write_text(json.dumps(proof))
            compatibility.write_text(json.dumps(dict(build_sha256=view.BUILD_SHA256,plan_sha256=view.sha(root/'PLAN.json'),cases=cases)))
            for family,path in proofs.items():
                if family=='silu_monotone':continue
                proof=json.loads(path.read_text());proof['registration_sha256']=view.sha(compatibility);path.write_text(json.dumps(proof))
            original,matrix=CorrectedRegistrationTests().fixture();effective=view.corrected_cells(original,matrix)
            registered=dict(plan_sha256=view.sha(root/'PLAN.json'))
            with patch('shaping_execution.load_registration',return_value=({},original,{})),patch.object(view,'registration',return_value=(registered,matrix,effective)):
                missing=dict(proofs);missing.pop('relu')
                with self.assertRaisesRegex(ValueError,'all six'):view.activate(root,missing)
                self.assertFalse((root/'corrected_thresholds/ACTIVATION.json').exists())
                view.activate(root,proofs)
                self.assertEqual(view.activated_view(root,original),effective)
                self.assertEqual(view.base_path(root,'relu',0).name,'relu-base.json')
                self.assertEqual(view.base_path(root,'product_pretrained_joint',0).name,'product_pretrained_joint-corrected-threshold-base.json')
                with self.assertRaisesRegex(ValueError,'preserve prior'):view.activate(root,proofs)
                view.admit_parent(root,dict(next(c for c in effective if c['id']==matrix['corrected_pretrained_conditions'][0]['id'])),
                    matrix['corrected_pretrained_conditions'][0]['parent'],'old-build',view.BUILD_SHA256)
                proofs['relu'].write_text('{}')
                with self.assertRaisesRegex(ValueError,'evidence drift'):view.activated_view(root,original)


if __name__=='__main__':unittest.main()
