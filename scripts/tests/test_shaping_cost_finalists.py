from pathlib import Path
import json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,ROLES,encoded,sha
from shaping_cost_finalists import coverage,prepare_outer


class FinalistTests(unittest.TestCase):
    def test_outer_workloads_use_each_folds_training_rows_and_preserve_all_models(self):
        from shaping_cost_fixture import float_bits
        from tree_study import RAW_IDS
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';study.mkdir();old=root/'old';(old/'data').mkdir(parents=True)
            populations=np.array(['cid22_a25','cid22_ssim2','kadid','konfig','tid2013']*24)
            raw=np.zeros((len(populations),len(RAW_IDS)),dtype=np.float32)
            raw[:,0]=np.arange(len(raw));np.save(old/'data/pop.npy',populations);np.save(old/'data/raw.npy',raw)
            plan=dict(input_pins={str(p):sha(p) for p in (old/'data').glob('*.npy')})
            (study/'REDUCED_SCOPE.json').write_text('approved');scope=dict(registration_sha256=sha(study/'REDUCED_SCOPE.json'))
            binaries=root/'bin';binaries.mkdir();converter=binaries/'bake_dial_refit';converter.write_text('converter')
            (binaries/'BUILD.json').write_bytes(encoded(dict(binary_sha256={'bake_dial_refit':sha(converter)})))
            families=dict(zip(ROLES,('relu','extra_trees','shaped_additive','softplus','product_joint_identity','separate_joint_identity')))
            rows=[];partitions={};cells=matrix()
            for outer in range(3):
                test=list(range(outer*40,(outer+1)*40));train=sorted(set(range(120))-set(test))
                partitions[f'outer-{outer}']=dict(train=train,test=test)
                for role,family in families.items():
                    for seed in ([None] if role=='shaped_additive' else range(3)):
                        cell=next(c for c in cells if c['stage']=='outer' and c['partition']==f'outer-{outer}' and c['family']==family and c['seed']==seed)
                        run=study/'runs'/cell['id']/'attempt-001';panel=run/'evaluation';panel.mkdir(parents=True)
                        (run/'RUN.json').write_bytes(encoded(dict(admission_sha256=f'admission-{outer}',epochs=1,schedule=dict(phase_epochs=[1]))))
                        model=run/('model.opaque-vector.zcth' if family=='extra_trees' else 'native/model.bin' if family=='shaped_additive' else 'fit/checkpoints/ckpt_epoch000.bin')
                        model.parent.mkdir(parents=True,exist_ok=True);model.write_text('model '+cell['id']);pin=sha(model)
                        if family in ('extra_trees','shaped_additive'):
                            qualification={'native_tree_gate':{'model_sha256':pin}} if family=='extra_trees' else {'model_sha256':pin}
                            report=dict(qualification=qualification)
                        else:
                            dense=panel/'dense'/(pin[:16]+'.bin');dense.parent.mkdir(exist_ok=True);dense.write_bytes(model.read_bytes())
                            report=dict(checkpoint_sha256=pin)
                        rows.append(dict(cell=cell,role=role,report=report,indices=list(range(120)),predictions=np.arange(120,dtype=float),run_directory=str(run),evaluation_directory=str(panel)))
            def convert(command,**kwargs):
                Path(command[-1]).write_bytes(Path(command[-3]).read_bytes())
                return SimpleNamespace(returncode=0,stdout='BIT-IDENTICAL',stderr='')
            def admitted(*args):
                # All row identities are chosen before the first model table is read.
                self.assertEqual(choose.call_count,3)
                return plan,partitions,rows,{},dict.fromkeys(('0','1','2'),{})
            from shaping_cost_fixture import choose_rows
            with patch('shaping_reduced_scope.coverage',return_value=scope),patch('shaping_cost_finalists.load_registration',return_value=(plan,cells,partitions)),patch('shaped_linear_study.OLD',old),patch('shaping_cost_finalists.choose_rows',wraps=choose_rows) as choose,patch('shaping_outer_evidence.collect',side_effect=admitted),patch('shaping_cost_finalists.subprocess.run',side_effect=convert),patch('shaping_cost_complexity.inputs',return_value=({},root/'inventory',root/'inspector')),patch('shaping_cost_complexity.write_fold') as complexity:
                result=prepare_outer(study,binaries,root/'out');self.assertEqual(result['models'],48)
                self.assertEqual(complexity.call_count,3)
                for outer in range(3):
                    workload=json.loads((root/f'out/outer-{outer}/WORKLOAD.json').read_text())
                    provenance=json.loads((root/f'out/outer-{outer}/PROVENANCE.json').read_text())
                    ids=provenance['original_training_indices'];self.assertEqual(len(ids),40)
                    self.assertFalse(set(ids)&set(partitions[f'outer-{outer}']['test']))
                    self.assertEqual(len(workload['arms']),16)
                    for arm in workload['arms']:self.assertEqual(arm['expected_score_bits'],[float_bits(float(i)) for i in ids])
                # A missing trained seed prevents cost preparation before output creation.
                rows.pop()
                choose.reset_mock()
                with self.assertRaisesRegex(ValueError,'48 reduced'):prepare_outer(study,binaries,root/'missing')
                self.assertFalse((root/'missing').exists())

    def test_full_role_seed_coverage_cannot_be_replaced_by_extra_models(self):
        selected=coverage(matrix());self.assertEqual(len(selected),16)
        with self.assertRaisesRegex(ValueError,'sixteen'):coverage(selected[:-1])
        duplicate=selected[:-1]+[selected[0]]
        with self.assertRaisesRegex(ValueError,'sixteen'):coverage(duplicate)
        changed=[dict(c) for c in selected];changed[0]['seed']=7
        with self.assertRaisesRegex(ValueError,'sixteen'):coverage(changed)


if __name__=='__main__':unittest.main()
