import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_rescue_wave_prepare import protocol_snapshot,selected_conditions,prepare_selected


class RescueProtocolTests(unittest.TestCase):
    def fixture(self,root):
        from shaping_completion import matrix
        study=root/'study';study.mkdir();scope_path=study/'SHORTLIST.json';scope_path.write_text('verified fixture shortlist')
        family='product_pretrained_frozen'
        cells=[c for c in matrix() if c['stage']=='rescue' and c['family']==family and c['partition'] in ('inner-0-0','inner-0-1') and c['rescue'] in ('init_half','constant_lr')]
        scope=dict(roles=dict(relu='relu',best_analytic='softplus',best_product=family,best_separate='separate_joint_identity'),cells=cells)
        return study,family,cells,scope,scope_path

    def test_shortlisted_plan_preserves_active_failed_and_assigned_conditions(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,family,cells,scope,path=self.fixture(root)
            active=study/'runs'/cells[0]['id']/'attempt-001';active.mkdir(parents=True);(active/'RUN.json').write_text('active witness')
            failed=study/'runs'/cells[1]['id']/'attempt-001';failed.mkdir(parents=True);(failed/'RESULT.json').write_text('failed witness')
            assigned=study/'downstream_assignments/already.json';assigned.parent.mkdir();assigned.write_bytes(encoded(dict(cells=[cells[2]['id']])))
            before={p:p.read_bytes() for p in study.rglob('*') if p.is_file()};lookup={c['id']:c for c in cells}
            with patch('shaping_reduced_scope.shortlist',return_value=(scope,path)),patch('shaping_rescue.resolve',side_effect=lambda root,cid:dict(cell=lookup[cid])):
                pending,evidence=selected_conditions(study,0,family)
            self.assertEqual(len(pending),9);self.assertEqual({c['id'] for c in pending},{c['id'] for c in cells[3:]})
            self.assertEqual(len(evidence['preserved_conditions']),3)
            self.assertTrue(all(p['completion_claim'] is False for p in evidence['preserved_conditions']))
            self.assertEqual(before,{p:p.read_bytes() for p in before});self.assertEqual(evidence['shortlist_sha256'],sha(path))

    def test_no_existing_condition_is_replayed_and_no_false_completion(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,family,cells,scope,path=self.fixture(root);lookup={c['id']:c for c in cells}
            for cell in cells:(study/'runs'/cell['id']).mkdir(parents=True)
            output=root/'output'
            with patch('shaping_reduced_scope.shortlist',return_value=(scope,path)),patch('shaping_rescue.resolve',side_effect=lambda root,cid:dict(cell=lookup[cid])):
                result=prepare_selected(study,root/'source',root/'bin',root/'qualification',0,family,{},output)
            self.assertEqual(result['stage'],'no_unassigned_conditions');self.assertFalse(result['completion_claim'])
            self.assertEqual(result['jobs'],[]);self.assertEqual(len(result['preserved_conditions']),12);self.assertFalse(output.exists())

    def test_unready_shortlist_wrong_family_and_changed_coverage_refuse_before_writes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study,family,cells,scope,path=self.fixture(root);output=root/'output'
            with patch('shaping_reduced_scope.shortlist',side_effect=ValueError('paired evidence incomplete')):
                with self.assertRaisesRegex(ValueError,'paired evidence'):prepare_selected(study,root/'source',root/'bin',root/'qualification',0,family,{},output)
            with patch('shaping_reduced_scope.shortlist',return_value=(scope,path)):
                with self.assertRaisesRegex(ValueError,'outside'):selected_conditions(study,0,'silu_diagnostic')
                scope['cells']=cells[:-1]
                with self.assertRaisesRegex(ValueError,'complete paired'):selected_conditions(study,0,family)
            self.assertFalse(output.exists());self.assertFalse((study/'runs').exists())

    def test_exported_registration_uses_only_identical_frozen_public_policy(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';study.mkdir();source=root/'export';(source/'docs').mkdir(parents=True)
            policy=source/'docs/SHAPING_REDUCED_SCOPE.md';policy.write_text('frozen public study policy')
            (study/'REDUCED_SCOPE.json').write_bytes(encoded(dict(evidence_sha256={'/original/checkout/docs/'+policy.name:sha(policy)})))
            target=protocol_snapshot(study,source);self.assertEqual(target.read_bytes(),policy.read_bytes())
            before=target.stat().st_mtime_ns;self.assertEqual(protocol_snapshot(study,source),target)
            self.assertEqual(target.stat().st_mtime_ns,before)
            target.write_text('changed')
            with self.assertRaisesRegex(ValueError,'preserve changed'):protocol_snapshot(study,source)
            self.assertEqual(target.read_text(),'changed')


if __name__=='__main__':unittest.main()
