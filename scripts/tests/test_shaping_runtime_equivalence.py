"""Certificate metadata fixtures; native trajectory evidence is archived separately."""
import copy,hashlib,json,shutil,sys,tempfile,unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_runtime_equivalence as owner
from shaping_completion import encoded,sha


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
        root=Path(directory);study=root/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
        baseline=root/'base';candidate=root/'candidate';baseline.mkdir();candidate.mkdir()
        for target in (baseline,candidate):
            for name in ('zensim_mlp_train','bake_dial_refit','panel','corrhead'):
                (target/name).write_bytes((target.name if name=='zensim_mlp_train' else name).encode())
            (target/'BUILD.json').write_bytes(encoded(dict(binary_sha256={p.name:sha(p) for p in target.iterdir()})))
        base=sha(baseline/'BUILD.json');build=sha(candidate/'BUILD.json');(study/'corrected_thresholds').mkdir()
        (study/'corrected_thresholds/COMPATIBILITY.json').write_bytes(encoded(dict(build_sha256=base)))
        for which,epochs in (('identity',240),('pretrained',120)):
            folder=root/which;folder.mkdir();cases={}
            for family,(key,count) in owner.CLASSES.items():
                if count!=epochs:continue
                cell=dict(id=key,family=family,stage='inner',partition='inner-0-0',setting=0,seed=0)
                refs={f'ckpt_epoch{e:03}.bin':hashlib.sha256(f'{family}:{e}'.encode()).hexdigest() for e in range(epochs)}
                compared={name:dict(equal=True,canonical_sha256={'original':pin,'neutral':pin},input_sha256={'original':pin,'neutral':pin}) for name,pin in refs.items()}
                proof=dict(stage='qualified',build_sha256=build,**{f'all_{epochs}_checkpoints_equal':True},
                    trajectory=dict(required=epochs,all_equal=True,stripped_metadata_keys=['zentrain.repro'],compared=compared))
                (folder/family).mkdir();run=dict(cell=cell,epochs=epochs,pairs_per_epoch=50000,build_sha256=build)
                (folder/family/'RUN.json').write_bytes(encoded(run))
                (folder/family/'RESULT.json').write_bytes(encoded(dict(run,**proof,exit_code=0) if epochs==240 else dict(run,stage='trained',exit_code=0)))
                if epochs==120:(folder/family/'QUALIFICATION.json').write_bytes(encoded(proof))
                cases[family]=dict(cell=cell,reference_sha256=refs)
            schema='paired-gradient-native-trajectory-v1' if epochs==240 else 'paired-gradient-corrected-pretrained-v1'
            (folder/'REGISTRATION.json').write_bytes(encoded(dict(schema=schema,cases=cases,epochs=epochs,pairs_per_epoch=50000,
                plan_sha256=sha(study/'PLAN.json'),build_sha256=build,outer_results_read=False,imported_original_shaping_banks=True,pretraining_replayed=False)))
            lines=''.join(sha(p)+'  '+p.relative_to(folder).as_posix()+'\n' for p in sorted(folder.rglob('*')) if p.is_file())
            (folder/'SHA256SUMS').write_text(lines);durable=root/(which+'-durable');shutil.copytree(folder,durable)
            (folder/'MIRROR_VERIFIED.json').write_bytes(encoded(dict(destination=str(durable),all_artifact_sha256_verified=True)))
        with patch.object(owner,'BASE_BUILD',base):yield root,study,baseline,candidate


class RuntimeEquivalenceTests(unittest.TestCase):
    def prepare(self,root,study,baseline,candidate):
        return owner.prepare(study,baseline,candidate,root/'identity',root/'pretrained')

    def test_activation_changes_only_supported_future_runtime_selection(self):
        with fixture() as (root,study,base,candidate):
            path=self.prepare(root,study,base,candidate)
            self.assertEqual(owner.validate(study,path)['checkpoint_comparisons'],720)
            with patch.object(owner,'runtime_family',return_value='product_pretrained_joint'):
                self.assertEqual(owner.select(study,'new','outer',base),(base,None))
                owner.activate(study,path);chosen,evidence=owner.select(study,'new','outer',base)
                self.assertEqual(chosen,candidate);self.assertEqual(owner.canonical_build(study,'product_pretrained_joint',sha(candidate/'BUILD.json')),sha(base/'BUILD.json'))
                record=dict(runtime_compatibility=evidence,build_sha256=sha(candidate/'BUILD.json'),binary_dir=str(candidate),cell={'id':'new'},execution_kind='outer')
                owner.verify_binding(study,record);record['build_sha256']='unqualified'
                with self.assertRaisesRegex(ValueError,'binding differs'):owner.verify_binding(study,record)
            with patch.object(owner,'runtime_family',return_value='softplus'):self.assertEqual(owner.select(study,'new','variant',base),(base,None))
            self.assertEqual(owner.select(study,'existing','corrected_inner',base),(base,None))
            with self.assertRaises(ValueError):owner.canonical_build(study,'product_pretrained_frozen',sha(candidate/'BUILD.json'))
            with self.assertRaisesRegex(ValueError,'preserve existing'):owner.activate(study,path)

    def test_partial_comparisons_and_numeric_drift_refuse(self):
        with fixture() as (root,study,base,candidate):
            path=self.prepare(root,study,base,candidate);value=json.loads(path.read_text());item=value['cases']['product_joint_identity'];proof=json.loads(Path(item['proof']).read_text())
            for change in ('missing','different','extra_strip'):
                bad=copy.deepcopy(proof)
                if change=='missing':bad['trajectory']['compared'].pop('ckpt_epoch000.bin')
                elif change=='different':bad['trajectory']['compared']['ckpt_epoch000.bin']['canonical_sha256']['neutral']='f'*64
                else:bad['trajectory']['stripped_metadata_keys'].append('weights')
                refs=json.loads(Path(item['registration']).read_text())['cases']['product_joint_identity']['reference_sha256']
                with self.subTest(change=change),self.assertRaises(ValueError):owner.check_trajectory(bad,240,refs)

    def test_changed_evidence_or_inference_executable_refuses(self):
        with fixture() as (root,study,base,candidate):
            path=self.prepare(root,study,base,candidate);(candidate/'panel').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'component drift'):owner.validate(study,path)
        with fixture() as (root,study,base,candidate):
            path=self.prepare(root,study,base,candidate);p=Path(json.loads(path.read_text())['cases']['separate_pretrained_joint']['proof']);p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'evidence drift'):owner.validate(study,path)

    def test_unchanged_baseline_and_unknown_runtime_preserve_original_gate(self):
        with fixture() as (_,study,base,_):
            self.assertEqual(owner.canonical_build(study,'product_pretrained_joint',sha(base/'BUILD.json')),sha(base/'BUILD.json'))
            with self.assertRaises(ValueError):owner.canonical_build(study,'product_pretrained_joint','unknown')


if __name__=='__main__':unittest.main()
