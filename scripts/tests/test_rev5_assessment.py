#!/usr/bin/env python3
"""Features-only admission boundaries through the bank and table owners."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rev4_featpot'))
import rev5_bank as bank
import v2c_wide as wide
sys.path.insert(0, str(REPO / 'scripts/lib'))
import assessment_identity as identity


class Assessment(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='assessment-', dir=os.environ.get('TMPDIR'))
        self.work = Path(self.temp.name)
        self.root = self.work / "inputs"
        self.source = self.root / 'source'
        self.base = self.source / 'fixture'
        self.base.mkdir(parents=True)
        self.ids = json.loads((REPO/'benchmarks/costset2_2026-10-03.candidate_ids.json').read_text())['candidates']['by_v2fy']
        self.old_profile = wide.PROFILE
        wide.PROFILE = wide.rev5_profile('rev5_localwin', 'basic+peaks+v2@w1825/rev5_localwin#36c3f3af', 'fixture-binary', 'fixture-build', None)
        self.keys = pa.table({'pair_key':['a','b'], 'row_id':[0,1], 'ref_group':['same','other'], 'ref_path':['reference','reference'], 'dist_path':['reference','distorted'], 'pixels_identical':[True,False]})
        pq.write_table(self.keys, self.base/'keys.parquet')
        self.features = pa.table({'pair_key':['a','b'], 'row_id':[0,1], **{f'f{i}':pa.array([i/7, i/11], type=pa.float64()) for i in range(1825)}})
        pq.write_table(self.features, self.base/'features.parquet')
        self.manifest = {'set':'fixture', 'schema':bank.SCHEMA, 'rows':2, 'feature_width':1825, 'formula_revision':'Rev5', 'era_label':wide.PROFILE.era, 'feature_set_id':wide.PROFILE.feature_set_id, 'binary_sha256':wide.PROFILE.binary, 'build_commit':wide.PROFILE.build, 'dtype':'float64', 'requested_slot_ranges':[list(v) for v in bank.SLOT_RANGES]}
        self.pin()

    def tearDown(self):
        wide.PROFILE = self.old_profile
        self.temp.cleanup()

    def pin(self):
        self.manifest.update(keys_sha256=identity.sha(self.base/'keys.parquet'), features_parquet_sha256=identity.sha(self.base/'features.parquet'))
        (self.base/'_MANIFEST.json').write_text(json.dumps(self.manifest))

    def build(self, out=None, ids=None):
        return wide.build_assessment(self.source, out or self.root/'view', ['fixture'], ids or self.ids)

    def test_projection_preserves_identity_rows_order_cast_and_absence(self):
        record = self.build()
        result = pq.read_table(record['tables'][0]['path'])
        self.assertEqual(result['pair_key'].to_pylist(), ['a','b'])
        self.assertEqual(result['pixels_identical'].to_pylist(), [True,False])
        self.assertNotIn('target', result.column_names)
        self.assertNotIn('human_score', result.column_names)
        for i in range(720):
            v = result[f'f{i}'].to_numpy()
            if i in self.ids:
                self.assertTrue(np.array_equal(v.view('uint32'), self.features[f'f{i}'].to_numpy().astype('float32').view('uint32')))
            else:
                self.assertTrue(np.isnan(v).all())
        self.assertEqual(record['tables'][0]['sha256'], identity.sha(record['tables'][0]['path']))

    def test_fx2_requires_explicit_revision7_and_preserves_new_measured_slots(self):
        from v2_common import derived_declaration
        _, pin, declaration = derived_declaration('fx2')
        ids = sorted(set(declaration['direct']) | {i for p in declaration['products'] for i in p})
        with self.assertRaises(ValueError):
            wide.build_assessment(self.source, self.root/'old-profile', ['fixture'], ids, 'fx2')
        wide.PROFILE = wide.rev5_profile('rev7_rowlanes', 'basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af',
                                       'fixture-binary', 'fixture-build', None, 7)
        self.manifest.update(formula_revision='Rev7', era_label=wide.PROFILE.era,
                             feature_set_id=wide.PROFILE.feature_set_id)
        self.pin()
        with self.assertRaises(ValueError):
            wide.build_assessment(self.source, self.root/'implicit', ['fixture'], ids)
        with self.assertRaises(ValueError):
            wide.build_assessment(self.source, self.root/'incomplete', ['fixture'], ids[:-1], 'fx2')
        with self.assertRaises(ValueError):
            wide.build_assessment(self.source, self.root/'no-owner', ['fixture'], ids, 'fx2')
        tool = self.root/'byc_fsid'; tool.write_bytes(b'canonical-owner-fixture')
        response = 'basic+v2@w720/rev7_rowlanes#ee342d8c slots_hash8=ee342d8c n_slots=504'
        with patch.object(wide.subprocess, 'check_output', return_value=response) as compute:
            record = wide.build_assessment(self.source, self.root/'fx2', ['fixture'], ids, 'fx2', tool)
        compute.assert_called_once_with([str(tool), 'basic+v2', '720', wide.PROFILE.era, ','.join(map(str, ids))], text=True)
        self.assertEqual(record['feature_set_id'], response.split()[0])
        empty = Path(record['empty_corpus']['path'])
        self.assertEqual([p.name for p in empty.iterdir()], ['_MANIFEST.json'])
        self.assertEqual(json.loads((empty/'_MANIFEST.json').read_text())['feature_set_id'], record['feature_set_id'])
        with patch.object(wide.subprocess, 'check_output', return_value=response.replace('n_slots=504', 'n_slots=503')):
            with self.assertRaises(ValueError):
                wide.build_assessment(self.source, self.root/'wrong-owner-response', ['fixture'], ids, 'fx2', tool)
        self.assertFalse((self.root/'wrong-owner-response').exists())
        self.assertEqual(record['derived_inputs']['sha256'], pin)
        table = pq.read_table(record['tables'][0]['path'])
        for i in ids:
            np.testing.assert_array_equal(table[f'f{i}'].to_numpy().view('uint32'),
                                          self.features[f'f{i}'].to_numpy().astype('float32').view('uint32'))
        self.assertEqual(len(ids), 504)
        self.assertEqual(table['pair_key'].to_pylist(), ['a', 'b'])
        manifest = json.loads(Path(record['tables'][0]['declaration']['path']).read_text())
        self.assertEqual(manifest['feature_ids'], ids)
        self.assertEqual(manifest['derived_inputs'], record['derived_inputs'])

    def test_refuses_wrong_ids_hash_order_nan_and_label_schema_before_output(self):
        for case in ['ids', 'hash', 'order', 'nan', 'labels']:
            with self.subTest(case=case):
                out = self.root/('refused-'+case)
                pq.write_table(self.keys, self.base/'keys.parquet')
                pq.write_table(self.features, self.base/'features.parquet')
                self.pin()
                ids = self.ids.copy()
                if case == 'ids': ids[0] = 12
                if case == 'hash': (self.base/'features.parquet').write_bytes(b'changed')
                if case == 'order':
                    pq.write_table(self.keys.take(pa.array([1,0])), self.base/'keys.parquet'); self.pin()
                if case == 'nan':
                    pq.write_table(self.features.set_column(self.features.schema.get_field_index('f13'),'f13',pa.array([np.nan,0.])), self.base/'features.parquet'); self.pin()
                if case == 'labels':
                    pq.write_table(self.keys.append_column('target',pa.array([1.,2.])), self.base/'keys.parquet'); self.pin()
                with self.assertRaises(Exception): self.build(out, ids)
                self.assertFalse(out.exists())

    def test_original_bank_and_bound_source_ancestry_refuse_before_payload(self):
        original = self.root/'original'; original.mkdir()
        self.manifest['assessment'] = {'immutable_roots':[str(original)]}; self.pin()
        link = self.root/'source-link'; link.symlink_to(original, target_is_directory=True)
        for out in [self.source/'view', original/'view', link/'view']:
            with patch.object(wide, 'read_matrix', side_effect=AssertionError('payload read')):
                with self.assertRaises((ValueError,PermissionError)): self.build(out)
            self.assertFalse(out.exists())

    def test_bank_preflight_refuses_stale_keys_labels_protected_and_source_outputs(self):
        spec = {'schema':'rev5-assessment-keys-v1','labels_read':False,'rows':2,'keys':{'path':str(self.base/'keys.parquet'),'sha256':identity.sha(self.base/'keys.parquet')},'extractor':{'sha256':identity.sha(__file__)},'immutable_roots':[str(self.source)]}
        manifest = self.root/'instrument.json'; manifest.write_text(json.dumps(spec))
        args = argparse.Namespace(set='fixture',instrument_manifest=manifest,out=self.source,limit=0,revision=5,bin=__file__)
        with patch.object(bank.pq,'read_table',side_effect=AssertionError('payload read')):
            with self.assertRaises((ValueError,PermissionError)): bank.cmd_extract(args)
        for case in ['stale', 'labels', 'protected']:
            args.out = self.root/('bank-'+case)
            spec['keys']['sha256'] = '0'*64
            if case == 'labels':
                pq.write_table(self.keys.append_column('target',pa.array([1.,2.])), self.base/'keys.parquet')
            if case == 'protected': spec['keys']['path'] = str(self.root/'_sealed'/'never-opened')
            manifest.write_text(json.dumps(spec))
            with patch.object(bank.pq,'read_table',side_effect=AssertionError('payload read')):
                with self.assertRaises((ValueError,PermissionError)): bank.cmd_extract(args)
            self.assertFalse(args.out.exists())

    def test_extractor_row_reordering_is_joined_by_ordinal_and_audited(self):
        extractor = self.root/'reordered-extractor'
        extractor.write_text("""#!/usr/bin/env python3
import csv,json,pathlib,sys
args=sys.argv[1:]
def arg(k): return args[args.index(k)+1]
rows=list(csv.DictReader(open(arg('--path')),delimiter='\t'))
out=pathlib.Path(arg('--out'))
with out.open('w') as f:
 writer=csv.writer(f);writer.writerow(['row_id']+['f'+str(i) for i in range(1825)])
 for r in reversed(rows):writer.writerow([r['row_id']]+[int(r['row_id'])+i/7 for i in range(1825)])
pathlib.Path(str(out)+'.manifest.json').write_text(json.dumps({'feature_set_id':'basic+peaks+v2@w1825/rev5_localwin#36c3f3af','era_label':'rev5_localwin','formula_revision':5}))
pathlib.Path(str(out)+'.research_manifest.json').write_text(json.dumps({'formula_revision_eras':['rev5']}))
with open(arg('--audit-jsonl'),'w') as f:
 for r in reversed(rows):
  f.write(json.dumps({'human_score':int(r['row_id']),'reference':r['ref_path'],'distorted':r['dist_path'],'model_inputs':[],'pixels_identical':r['ref_path']==r['dist_path'],'width':1,'height':1,'reference_file_sha256':'synthetic','distorted_file_sha256':'synthetic','reference_pixels_sha256':'synthetic','distorted_pixels_sha256':'synthetic'})+'\\n')
""")
        extractor.chmod(0o755)
        spec = {'schema':'rev5-assessment-keys-v1','labels_read':False,'rows':2,'keys':{'path':str(self.base/'keys.parquet'),'sha256':identity.sha(self.base/'keys.parquet')},'extractor':{'sha256':identity.sha(extractor)},'immutable_roots':[str(self.source)],'instrument':{'name':'software row-join fixture'},'exposure_freeze':{'state':'synthetic software only'}}
        manifest=self.root/'instrument.json';manifest.write_text(json.dumps(spec))
        args=argparse.Namespace(set='fixture',instrument_manifest=manifest,out=self.work/'joined-bank',limit=0,revision=5,bin=str(extractor),era='rev5_localwin',tier='native',threads=1,chunk=2,build_commit='synthetic')
        self.assertEqual(bank.cmd_extract(args),0)
        result=pq.read_table(args.out/'fixture/features.parquet')
        self.assertEqual(result['pair_key'].to_pylist(),['a','b'])
        self.assertEqual(result['row_id'].to_pylist(),[0,1])
        self.assertEqual(result['f13'].to_pylist(),[13/7,1+13/7])
        manifest_out=json.loads((args.out/'fixture/_MANIFEST.json').read_text())
        self.assertEqual(manifest_out['pairs_origin'],str(self.base/'keys.parquet')+' (ref_path,dist_path in row order)')
        self.assertEqual(pq.read_table(args.out/'fixture/keys.parquet')['pixels_identical'].to_pylist(),[True,False])

    def metadata_fixture(self):
        record=self.build()
        assessment=self.root/'view/ASSESSMENT.json'
        proof=self.root/'proof.json';proof.write_text(json.dumps({'qualified_provenance':True,'tables':record['tables']}))
        corpus=self.root/'corpus';corpus.mkdir()
        table={'path':record['tables'][0]['path'],'sha256':record['tables'][0]['sha256']}
        (corpus/'ext_cid22val.parquet').symlink_to(table['path'])
        capsule={'schema':'rev5-assessment-eval-identity-v1','features_only':True,'labels_read':False,'features_root':str(corpus),'assessments':[{'path':str(assessment),'sha256':identity.sha(assessment)}],'admission':{'path':str(proof),'sha256':identity.sha(proof)},'inputs':{k:table for k in ['ext_cid22val.parquet','dial-grid','identity-probe','negtail-probe']}}
        cp=self.root/'capsule.json';cp.write_text(json.dumps(capsule))
        return cp,Path(table['path']),corpus

    def protected_metadata_case(self, location):
        import builtins
        import io
        cp,table,corpus=self.metadata_fixture()
        # Synthetic sentinels only; never inspect a scientific protected tree.
        protected=self.work/'_sealed';protected.mkdir()
        sentinel=protected/'sentinel.json';sentinel.write_text('{"sampling":{"human_score":[987654321]}}')
        alias=corpus/'_MANIFEST.json' if location=='root' else Path(str(table)+'._MANIFEST.json')
        alias.symlink_to(sentinel)
        opened=[]
        def tripwire(original):
            def guarded(path,*args,**kwargs):
                if not isinstance(path,int) and Path(path).resolve()==sentinel:
                    opened.append(str(path))
                    raise AssertionError('protected sentinel opened')
                return original(path,*args,**kwargs)
            return guarded
        with patch.object(builtins,'open',tripwire(builtins.open)),patch.object(io,'open',tripwire(io.open)):
            with self.assertRaises(PermissionError): identity.validate(cp,self.work/'refused')
        self.assertEqual(opened,[])
        self.assertFalse((self.work/'refused').exists())

    def test_reviewer_root_manifest_symlink_refuses_before_open(self):
        self.protected_metadata_case('root')

    def test_reviewer_alternate_table_sidecar_symlink_refuses_before_open(self):
        self.protected_metadata_case('alternate')

    def holdout_companion_case(self, kind, symlinked):
        import builtins
        import io
        cp, _, _ = self.metadata_fixture()
        protected = self.work/'private-HOLDOUT-population'
        protected.mkdir()
        sentinel = protected/(kind+'.json')
        sentinel.write_text('{"synthetic_human_score":[987654321]}')
        candidate = sentinel
        if symlinked:
            candidate = self.root/(kind+'-alias.json')
            candidate.symlink_to(sentinel)
        opened = []
        def tripwire(original):
            def guarded(path, *args, **kwargs):
                if not isinstance(path, int) and Path(path).resolve() == sentinel:
                    opened.append(str(path))
                    raise AssertionError('holdout sentinel opened')
                return original(path, *args, **kwargs)
            return guarded
        discovery = {'schema':'bake-verdict-input-paths-v1', 'complete':True,
                     'metadata_read':False, 'files':[str(candidate)]}
        with patch.object(builtins, 'open', tripwire(builtins.open)), patch.object(io, 'open', tripwire(io.open)):
            with self.assertRaises(PermissionError):
                identity.validate(cp, self.work/'refused', owner_inputs=discovery)
            # Inventory must preflight the whole set before even hashing an
            # allowed prefix; no caller can bypass discovery's refusal.
            with patch.object(identity, 'sha') as hashing:
                with self.assertRaises(PermissionError):
                    identity.checked_inventory([cp, candidate])
                hashing.assert_not_called()
        self.assertEqual(opened, [])
        self.assertFalse((self.work/'refused').exists())

    def test_direct_holdout_companion_source_refuses_before_open(self):
        self.holdout_companion_case('source', False)

    def test_symlinked_holdout_companion_source_refuses_before_open(self):
        self.holdout_companion_case('source', True)

    def test_direct_holdout_companion_binding_refuses_before_open(self):
        self.holdout_companion_case('binding', False)

    def test_symlinked_holdout_companion_binding_refuses_before_open(self):
        self.holdout_companion_case('binding', True)

    def test_all_declaration_locations_and_model_sidecars_are_checked(self):
        cp,table,corpus=self.metadata_fixture()
        bake=self.root/'model';bake.write_text('synthetic')
        sealed=self.work/'_sealed';sealed.mkdir();sentinel=sealed/'sentinel';sentinel.write_text('synthetic')
        aliases=[table.parent/'_MANIFEST.json',Path(str(corpus/'ext_cid22val.parquet')+'.manifest.json'),Path(str(bake)+'.spec.json')]
        for alias in aliases:
            with self.subTest(alias=alias):
                alias.symlink_to(sentinel)
                with self.assertRaises(PermissionError):identity.validate(cp,self.work/'out',[bake])
                alias.unlink()
        discovery={'schema':'bake-verdict-input-paths-v1','complete':True,'metadata_read':False,'files':[str(sentinel)]}
        with self.assertRaises(PermissionError):identity.validate(cp,self.work/'out',[bake],discovery)

    def test_metadata_presence_and_bytes_are_bound_and_incomplete_claim_is_unknown(self):
        cp,table,corpus=self.metadata_fixture()
        before=identity.validate(cp,self.work/'out')
        self.assertIsNone(before['labels_read'])
        self.assertFalse(before['label_boundary']['complete_discovery_checked'])
        root=corpus/'_MANIFEST.json'
        discovery={'schema':'bake-verdict-input-paths-v1','complete':True,'metadata_read':False,'files':[str(root)]}
        root.write_text('{"formula_revision":5}')
        first=identity.validate(cp,self.work/'out',owner_inputs=discovery)
        self.assertFalse(first['labels_read'])
        self.assertTrue(first['label_boundary']['complete_discovery_checked'])
        root.write_text('{"formula_revision":4}')
        second=identity.validate(cp,self.work/'out',owner_inputs=discovery)
        self.assertNotEqual(first['checked_inputs'],second['checked_inputs'])
        stale={'files':[{'path':str(root),'sha256':next(v['sha256'] for v in first['checked_inputs'] if v['path']==str(root))}]}
        with self.assertRaises(ValueError):identity.validate(cp,self.work/'out',owner_inputs=discovery,owner_identity=stale)

    def test_identity_capsule_binds_all_tables_and_never_scores(self):
        record = self.build()
        assessment = self.root/'view'/'ASSESSMENT.json'
        proof = self.root/'proof.json'; proof.write_text(json.dumps({'qualified_provenance':True,'tables':record['tables']}))
        corpus = self.root/'corpus'; corpus.mkdir()
        slot = corpus/'ext_cid22val.parquet'; slot.symlink_to(record['tables'][0]['path'])
        table = {'path':record['tables'][0]['path'],'sha256':record['tables'][0]['sha256']}
        capsule = {'schema':'rev5-assessment-eval-identity-v1','features_only':True,'labels_read':False,'features_root':str(corpus), 'assessments':[{'path':str(assessment),'sha256':identity.sha(assessment)}], 'admission':{'path':str(proof),'sha256':identity.sha(proof)}, 'inputs':{k:table for k in ['ext_cid22val.parquet','dial-grid','identity-probe','negtail-probe']}}
        cp = self.root/'capsule.json'; cp.write_text(json.dumps(capsule))
        heavy = self.root/'heavy'; heavy.write_text('#!/bin/bash\nshift 4\nexec "$@"\n'); heavy.chmod(0o755)
        bv = self.root/'verdict'; bv.write_text('''#!/usr/bin/env python3
import json,sys
if '--print-input-paths' in sys.argv:
 print(json.dumps({'schema':'bake-verdict-input-paths-v1','complete':True,'metadata_read':False,'files':[]}));sys.exit(0)
assert '--print-inputs' in sys.argv
assert '--fulleval' not in sys.argv
assert sys.argv[sys.argv.index('--corpora')+1]=='cid22'
assert '--corruption-grid' in sys.argv and '--perpair-metrics' in sys.argv
print(json.dumps({'argv':sys.argv[1:],'files':[]}))
'''); bv.chmod(0o755)
        bake = self.root/'model'; bake.write_text('synthetic model; fake evaluator')
        out = self.work/'transport'
        env = dict(os.environ,ZENSIM_EVAL_INSTRUMENTS=str(cp), ZENSIM_BAKE_VERDICT=str(bv),ZENSIM_RUN_HEAVY=str(heavy), ZENSIM_FULLEVAL_OUT=str(out),ZENSIM_EVAL_CORRUPTION_HEAD=str(bake))
        cmd = [str(REPO/'scripts/run_full_eval.sh'),'--stage','identities',str(bake),'fixture','720',str(corpus)]
        run = subprocess.run(cmd,env=env,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        transported=json.loads((out/'fixture.identities.json').read_text())
        self.assertEqual(transported['scoring'],'not_requested')
        self.assertIn('--corruption-head',transported['input_identity']['argv'])
        self.assertFalse((out/'fixture.fulleval.json').exists())
        with self.assertRaises((ValueError,PermissionError)): identity.validate(cp, self.source/'output')
        Path(table['path']).write_bytes(b'changed')
        refused=self.work/'refused-transport';env['ZENSIM_FULLEVAL_OUT']=str(refused)
        self.assertNotEqual(subprocess.run(cmd,env=env,capture_output=True).returncode,0)
        self.assertFalse(refused.exists())


if __name__ == '__main__': unittest.main()
