import json,sys,tempfile,unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_bootstrap_worker as bootstrap
import shaping_rescue_worker as worker
from shaping_completion import encoded,sha


class BootstrapWorkerTests(unittest.TestCase):
    def test_large_bootstrap_input_keeps_the_existing_one_gib_per_reservation_limit(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);request=root/'REQUEST.json';request.write_bytes(encoded(dict(study=str(root),execution_kind='bootstrap')))
            with patch.object(worker,'verify'),patch('shaping_storage.reservation',return_value=nullcontext()) as reserve,patch.object(worker.subprocess,'run'):
                worker.receive(request,['--server'])
            self.assertEqual(reserve.call_count,2)
            self.assertEqual({call.kwargs['bytes_required'] for call in reserve.call_args_list},{1024**3})
            self.assertEqual(len({call.args[1] for call in reserve.call_args_list}),2)

    def test_terminal_pack_collect_preserves_source_and_checks_exact_admitted_models(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';study.mkdir();source=root/'results/primary';source.mkdir(parents=True)
            (source/'samples.npz').write_bytes(b'fixture transport payload')
            pins={'source/scripts/owner.py':'source-hash','/data/response.tsv':'response-hash'}
            resolution=dict(input_sha256=pins,cells=[dict(id='outer-model')],comparison=None,qualifications={})
            record=dict(study=str(study),result_output=str(source),host='r3500',cell=dict(id='bootstrap-condition'),resolution=resolution)
            summary=dict(input_sha256={'/worker/export/scripts/owner.py':'source-hash','/data/response.tsv':'response-hash'},
                source_root='/worker/export',samples_sha256=sha(source/'samples.npz'),convergence_qualifications={},
                outer_results_read=True,final_holdout_read=False,outer_population_reports={'outer-model':{}})
            def save(value):
                (source/'SUMMARY.json').write_bytes(encoded(value));(source/'COMPLETE.json').write_bytes(encoded(dict(summary_sha256=sha(source/'SUMMARY.json'))))
            save(dict(summary,outer_population_reports={'wrong-model':{}}))
            with self.assertRaisesRegex(ValueError,'paired model coverage'):bootstrap.validate_result(record,source)
            save(summary);request=root/'REQUEST.json';request.write_bytes(encoded(record));bundle=root/'bundles/result.tar'
            with patch.object(worker,'verify'):receipt=bootstrap.pack(request,bundle)
            saved=root/'worker-preserved';source.rename(saved)
            with patch.object(bootstrap,'reproduce',return_value=(resolution,[])):
                result=bootstrap.collect(request,bundle,receipt['bundle_sha256'])
            self.assertEqual(result['scientific_fit_count'],0);self.assertTrue((source/'WORKER_COLLECTION.json').exists())
            self.assertEqual((source/'samples.npz').read_bytes(),(saved/'samples.npz').read_bytes())

    def test_source_export_relocation_preserves_scope_but_not_changed_hashes(self):
        coordinator=Path('/coordinator/source');remote=Path('/worker/export')
        def context(source,pin):return dict(pins={str(source/'scripts/owner.py'):pin,'/data/frozen.tsv':'data-hash'},spec=None,outputs=[],qualifications={})
        with patch.object(worker,'REPO',coordinator):expected=bootstrap.scope(context(coordinator,'source-hash'))
        with patch.object(worker,'REPO',remote):
            self.assertEqual(bootstrap.scope(context(remote,'source-hash')),expected)
            self.assertNotEqual(bootstrap.scope(context(remote,'changed')),expected)
        self.assertEqual(worker.source_identity('/data/frozen.tsv',coordinator),'/data/frozen.tsv')

    def test_capsule_uses_original_gate_and_binds_outer_inputs_before_cpu_claim(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';study.mkdir();draws=root/'draws';draws.mkdir();binaries=root/'binaries';binaries.mkdir()
            for path in (study/'PLAN.json',draws/'REGISTRATION.json',binaries/'BUILD.json'):path.write_text('{}')
            marker=study/'prediction-evidence.json';marker.write_text('{}')
            context=dict(pins={str(marker):sha(marker)},spec=None,outputs=[dict(cell=dict(id='outer-cell'))],qualifications={})
            def admit(args,host=None):
                self.assertEqual(host,'r3500');marker.read_bytes();return context
            with patch('shaping_bootstrap.admit',side_effect=admit) as gate,patch.object(worker,'DATA_ROOTS',(root,)):
                record=bootstrap.register(study,draws,binaries,root/'results/primary','r3500',root/'request')
                gate.assert_called_once();self.assertTrue(record['outer_results_read']);self.assertFalse(record['final_holdout_read'])
                self.assertEqual(record['execution_kind'],'bootstrap');self.assertIn(str(marker),record['files'])
                with patch.object(worker.socket,'gethostname',return_value='r3500'):
                    self.assertEqual(worker.verify(record),[])
                    resolution,command=bootstrap.reproduce(record)
                    self.assertEqual(resolution,record['resolution']);self.assertEqual(Path(command[2]).name,'shaping_bootstrap.py')
                    marker.write_text('drift')
                    with self.assertRaisesRegex(ValueError,'drift'):worker.verify(record)

    def test_refused_outer_gate_writes_no_request_or_result(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';study.mkdir()
            with patch('shaping_bootstrap.admit',side_effect=ValueError('outer matrix incomplete')):
                with self.assertRaisesRegex(ValueError,'outer matrix incomplete'):
                    bootstrap.register(study,root/'draws',root/'bin',root/'results/primary','r3500',root/'request')
            self.assertFalse((root/'request').exists());self.assertFalse((root/'results/primary').exists())

    def test_worker_cannot_bypass_changed_original_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);request=root/'REQUEST.json'
            request.write_bytes(encoded(dict(study=str(root),cell=dict(id='boot'),resolution={'fixed':'original'},execution_kind='bootstrap')))
            with patch.object(worker,'verify'),patch.object(bootstrap,'reproduce',return_value=({'fixed':'changed'},['owner'])),patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'reproduce original evidence'):worker.work(request)
                claim.assert_not_called()

    def test_only_analysis_kind_gets_larger_closure_and_outer_response_admission(self):
        self.assertEqual(worker.limits('bootstrap'),(2*1024**3,16000))
        self.assertEqual(worker.limits('corrected_inner'),(384*1024**2,4000))
        record=dict(schema='shaping-native-worker-v2',host='r3500',driver_sha256=sha(worker.__file__),
            execution_kind='bootstrap',cell=dict(stage='bootstrap'),outer_results_read=False)
        with patch.object(worker.socket,'gethostname',return_value='r3500'):
            with self.assertRaisesRegex(ValueError,'identity differs'):worker.verify(record)


if __name__=='__main__':unittest.main()
