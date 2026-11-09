"""Admission/transport tests; fixture bytes are not scoring qualification."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import ROLES,encoded,sha
import shaping_cost_worker as owner
import shaping_vector_bench as bench
import shaping_rescue_worker as worker


def fixture(root):
    from scripts.tests.test_shaping_cost_complexity import fixture as complexity_fixture
    study=root/'study';study.mkdir();(study/'PLAN.json').write_text('{}')
    prepared=root/'prepared';prepared.mkdir();marker=root/'evidence.json';marker.write_text('{}');pins={str(marker):sha(marker)}
    resolution=dict(input_sha256=pins,cells=[],chosen_rows={str(o):list(range(o*40,(o+1)*40)) for o in range(3)},qualifications={str(o):{} for o in range(3)},raw_ids=[0])
    record=dict(execution_kind='cost_prepare',study=str(study),result_output=str(prepared),source_root=str(root/'source'),host='i270',cell=dict(id='cost-condition'),resolution=resolution)
    paths=[]
    families=dict(zip(ROLES,('relu','extra_trees','shaped_additive','softplus','product_joint_identity','separate_joint_identity')))
    for outer in range(3):
        directory=prepared/f'outer-{outer}';(directory/'models').mkdir(parents=True);cells=[];arms=[]
        for role in ROLES:
            for seed in ((None,) if role=='shaped_additive' else range(3)):
                cell=dict(id=f'{outer}-{role}-{seed}',family=families[role],partition=f'outer-{outer}',seed=seed)
                item=dict(cell=cell,role=role,schedule={'phase_epochs':[240]});resolution['cells'].append(item);cells.append(dict(item,resolved_family=families[role]))
                name=role+'-'+('deterministic' if seed is None else str(seed));model=directory/'models'/(name+'.bin');model.write_bytes(name.encode())
                arms.append(dict(id=name,kind='bake',model='models/'+model.name,sha256=sha(model),expected_score_bits=[0]*40))
        path=directory/'WORKLOAD.json';path.write_bytes(encoded(dict(schema='shaping-cost-v1',admission_sha256='admitted',raw_ids=[0],feature_bits=[[0]]*40,arms=arms)));paths.append(str(path))
        complexity_fixture(directory/'complexity',path,cells,pins)
        (directory/'PROVENANCE.json').write_bytes(encoded(dict(outer=outer,workload_sha256=sha(path),original_training_indices=resolution['chosen_rows'][str(outer)],cells=cells,source_pins=pins,timing_measured=False,convergence_qualifications={})))
    (prepared/'WORKLOADS.json').write_bytes(encoded(dict(workloads=paths,models=48,timing_measured=False)))
    request=root/'REQUEST.json';request.write_bytes(encoded(record));(prepared/'WORKER_COLLECTION.json').write_bytes(encoded(dict(request_sha256=sha(request),source_host='i270')))
    return study,prepared,request,record


class CostWorkerTests(unittest.TestCase):
    def test_capsule_reproduces_cost_gate_and_keeps_existing_training_caps(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study,_,_,record=fixture(root);binaries=root/'bin';binaries.mkdir();(binaries/'BUILD.json').write_text('{}')
            def context(*args):
                (root/'evidence.json').read_bytes();return record['resolution']
            with patch.object(owner,'context',side_effect=context),patch.object(worker,'DATA_ROOTS',(root,)):
                q=owner.register(study,binaries,root/'new-prepared','i270',root/'registered')
                self.assertTrue(q['outer_results_read']);self.assertFalse(q['final_holdout_read'])
                self.assertEqual(q['execution_kind'],'cost_prepare');self.assertIn(str(root/'evidence.json'),q['files'])
                for module in ('shaping_parameter_count.py','shaping_training_resources.py'):
                    self.assertTrue(any(Path(p).name==module for p in q['files']))
                with patch.object(worker.socket,'gethostname',return_value='i270'):
                    self.assertEqual(worker.verify(q),[]);resolution,command=owner.reproduce(q)
                    self.assertEqual(resolution,q['resolution']);self.assertEqual(Path(command[2]).name,'shaping_cost_finalists.py')
                    (root/'evidence.json').write_text('drift')
                    with self.assertRaisesRegex(ValueError,'dependency drift'):worker.verify(q)
            self.assertEqual(worker.limits('cost_prepare'),(2*1024**3,16000))
            self.assertEqual(worker.limits('corrected_inner'),(384*1024**2,4000))

    def test_complete_transport_preserves_every_model(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);_,prepared,request,record=fixture(root);owner.validate_result(record,prepared);bundle=root/'result.tar'
            with patch.object(worker,'verify'):receipt=owner.pack(request,bundle)
            saved=root/'worker-preserved';prepared.rename(saved)
            with patch.object(owner,'reproduce',return_value=(record['resolution'],[])):owner.collect(request,bundle,receipt['bundle_sha256'])
            for source in saved.glob('outer-*/models/*.bin'):self.assertEqual(sha(source),sha(prepared/source.relative_to(saved)))

    def test_changed_train_rows_or_model_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            _,prepared,_,record=fixture(Path(d));path=prepared/'outer-1/PROVENANCE.json';original=path.read_bytes();p=json.loads(original);p['original_training_indices'][0]=999;path.write_bytes(encoded(p))
            with self.assertRaisesRegex(ValueError,'TRAIN rows'):owner.validate_result(record,prepared)
            path.write_bytes(original);next(prepared.glob('outer-2/models/*.bin')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'model hash'):owner.validate_result(record,prepared)

    def test_missing_fold_and_changed_source_pin_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            _,prepared,_,record=fixture(Path(d));p=prepared/'WORKLOADS.json';original=p.read_bytes();v=json.loads(original);v['workloads'].pop();p.write_bytes(encoded(v))
            with self.assertRaisesRegex(ValueError,'complete untimed'):owner.validate_result(record,prepared)
            p.write_bytes(original);record['resolution']['input_sha256'][str(Path(d)/'evidence.json')]='changed'
            with self.assertRaisesRegex(ValueError,'input pins'):owner.validate_result(record,prepared)

    def test_changed_reproduction_never_claims_cpu(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            _,_,request,_=fixture(Path(d))
            with patch.object(worker,'verify'),patch.object(owner,'reproduce',return_value=({'changed':True},['command'])),patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'reproduce original evidence'):worker.work(request)
                claim.assert_not_called()

    def test_refused_admission_creates_no_output(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study=root/'study';study.mkdir()
            with patch.object(owner,'context',side_effect=ValueError('outer evidence incomplete')):
                with self.assertRaisesRegex(ValueError,'outer evidence incomplete'):owner.register(study,root/'bin',root/'results/prepared','i270',root/'request')
            self.assertFalse((root/'request').exists());self.assertFalse((root/'results/prepared').exists())


class VectorBenchTests(unittest.TestCase):
    def registered(self,root):
        study,prepared,request,_=fixture(root);binary=root/'shaping_cost';binary.write_bytes(b'fixture binary');(root/'BUILD.json').write_bytes(encoded(dict(binary_sha256=sha(binary))))
        q=root/'QUALIFICATION.json';q.write_bytes(encoded(dict(native_instrument_qualified=True,models=8,rows=40,binary_sha256=sha(binary))))
        launcher=root/'profile_test_cpuset.sh';launcher.write_text('fixture launcher');tool=root/'zenbench';tool.write_text('fixture CPU tool');out=root/'benchmark'
        bench.register(study,prepared,request,binary,q,launcher,tool,out);return study,prepared,out

    def test_all_models_bound_and_changed_model_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,prepared,out=self.registered(Path(d));self.assertEqual(bench.verify(out,study)['models'],48);next(prepared.glob('outer-0/models/*.bin')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'input drift'):bench.verify(out,study)

    def test_existing_owner_never_replays(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,_,out=self.registered(Path(d));(out/'STATUS.json').write_text('{}')
            with patch('shaping_reduced_scope.admit_compute'),patch.object(bench.platform,'node',return_value='i270'),patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'no implicit replay'):bench.run(out,study,'0-7')
                claim.assert_not_called()

    def test_failed_timing_preserves_evidence_and_releases_cpu(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study,_,out=self.registered(Path(d))
            from contextlib import nullcontext
            from types import SimpleNamespace
            child=SimpleNamespace(pid=4321,wait=lambda:1)
            with patch('shaping_reduced_scope.admit_compute'),patch.object(bench.platform,'node',return_value='i270'),patch('shaping_rescue_dispatch.reserve_worker',return_value={'can_start':True}) as reserve,patch('shaping_rescue_dispatch.bind_worker'),patch('shaping_rescue_dispatch.release_worker') as release,patch('shaping_storage.reservation',return_value=nullcontext()),patch.object(bench.subprocess,'Popen',return_value=child):
                with self.assertRaisesRegex(RuntimeError,'preserve original attempt'):bench.run(out,study,'0-7')
                self.assertTrue(reserve.call_args.kwargs['require_idle'])
                release.assert_called_once();self.assertTrue((out/'outer-0.log').exists());self.assertEqual(json.loads((out/'STATUS.json').read_text())['stage'],'failed_with_preserved_evidence')


if __name__=='__main__':unittest.main()
