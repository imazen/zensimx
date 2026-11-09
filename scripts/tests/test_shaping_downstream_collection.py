import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import encoded,sha
from shaping_worker_collect import validate,pack_reserved,unpack,collect


class DownstreamCollectionTests(unittest.TestCase):
    def fixture(self,study,kind,family,epochs):
        root=study/'runs/cell/attempt-001';root.mkdir(parents=True)
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value));return path
        write(study/'PLAN.json',{})
        declaration=write(study/'declaration.json',{})
        cell=dict(id='cell',stage={'outer':'fresh_control_outer_matched','variant':'human_only','extension':'exposure_extension'}[kind],
                  family=family if kind in ('outer','extension') else 'best_product',partition='outer-0')
        schedule=dict(phase_epochs=epochs,total_pair_draws=sum(epochs)*50000,setting=0,optimizer_choice='base')
        resolution=dict(cell=cell,schedule=schedule,admission_sha256='admission',
                        resolution=dict(cell=cell,resolved_family=family),outer_results_read=False)
        request=write(study/'worker-request.json',dict(schema='shaping-native-worker-v2',execution_kind=kind,
            host='r3500',study=str(study),cell=cell,plan_sha256=sha(study/'PLAN.json'),build_sha256='build',resolution=resolution))
        assignment=dict(host='r3500',cells=['cell'],plan_sha256=sha(study/'PLAN.json'),build_sha256='build',
                        worker_request_path=str(request),worker_request_sha256=sha(request))
        assignment_path=write(study/'assignment.json',assignment)
        run=dict(cell=cell,execution_host='r3500',plan_sha256=assignment['plan_sha256'],build_sha256='build',
                 epochs=epochs[-1],pairs_per_epoch=50000,process_role=kind+'_controller',fit_directory='fit',
                 training_exposure_pair_draws=schedule['total_pair_draws'],outer_results_read=False)
        if kind=='outer':run.update(schedule=schedule,admission_sha256='admission')
        else:run.update(resolution=resolution['resolution'])
        phases=[];names=['pretraining','fit'] if '_pretrained_' in family else ['fit']
        for name,count in zip(names,epochs):
            directory=root/name;models=directory/'checkpoints';models.mkdir(parents=True)
            for epoch in range(count):(models/f'ckpt_epoch{epoch:03}.bin').write_bytes(epoch.to_bytes(2,'little'))
            phase=dict(command=['trainer','--derived-inputs',str(declaration)],epochs=count,pairs_per_epoch=50000,
                       stage='running',build_sha256='build')
            write(directory/'RUN.json',phase)
            phase.update(stage='trained',exit_code=0,bake_sha256=sha(models/f'ckpt_epoch{count-1:03}.bin'))
            write(directory/'RESULT.json',phase);phases.append(phase)
        write(root/'RUN.json',run);result=dict(run,phases=phases,stage='trained',exit_code=0,bake_sha256=phases[-1]['bake_sha256'])
        write(root/'RESULT.json',result)
        return root,run,result,assignment,assignment_path,resolution,declaration

    def test_pack_unpack_preserves_all_native_phases_and_resolves_conditional_cells(self):
        cases=[('outer','relu',[120]),('outer','product_pretrained_joint',[240,240]),
               ('variant','product_pretrained_frozen',[480,480]),('variant','softplus',[240]),
               ('extension','relu',[480]),('extension','separate_pretrained_frozen',[240,240])]
        for kind,family,epochs in cases:
            with self.subTest(kind=kind,family=family),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                study=Path(directory);root,run,result,assignment,path,resolution,declaration=self.fixture(study,kind,family,epochs)
                with patch('shaping_downstream_worker.reproduce',return_value=(resolution,[])) as gate, \
                     patch('shaping_execution.load_registration',return_value=({},[],{})), \
                     patch('shaping_worker_collect.socket.gethostname',return_value='r3500'), \
                     patch('shaping_worker_collect.DECLARATION_SHA256',sha(declaration)):
                    bundle=study/'result.tar';receipt=pack_reserved(study,path,'cell','attempt-001',bundle)
                    imported=study/'unpacked';unpack(bundle,imported,receipt['bundle_sha256'])
                    self.assertEqual(validate(imported,assignment,[],study),run)
                    self.assertEqual((imported/'RESULT.json').read_bytes(),(root/'RESULT.json').read_bytes())
                    self.assertTrue(gate.called)
                    self.assertEqual(len(list(imported.rglob('ckpt_epoch*.bin'))),sum(epochs))

    def test_missing_checkpoint_and_changed_gate_or_request_are_rejected(self):
        for change in ('checkpoint','gate','request','exposure','phase'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                study=Path(directory);root,run,result,assignment,path,resolution,declaration=self.fixture(study,'variant','product_pretrained_frozen',[120,120])
                returned=json.loads(json.dumps(resolution))
                if change=='checkpoint':(root/'pretraining/checkpoints/ckpt_epoch000.bin').unlink()
                if change=='gate':returned['schedule']['setting']=3
                if change=='request':Path(assignment['worker_request_path']).write_text('{}')
                if change=='exposure':
                    run['training_exposure_pair_draws']+=50000;result['training_exposure_pair_draws']+=50000
                    (root/'RUN.json').write_bytes(encoded(run));(root/'RESULT.json').write_bytes(encoded(result))
                if change=='phase':
                    result['phases']=result['phases'][1:];(root/'RESULT.json').write_bytes(encoded(result))
                with patch('shaping_downstream_worker.reproduce',return_value=(returned,[])):
                    with self.assertRaises(ValueError):validate(root,assignment,[],study)

    def test_complete_transport_import_keeps_request_and_phase_identities(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study=Path(directory);root,run,result,assignment,path,resolution,declaration=self.fixture(study,'variant','product_pretrained_joint',[120,120])
            expected=(root/'RESULT.json').read_bytes()
            with patch('shaping_downstream_worker.reproduce',return_value=(resolution,[])), \
                 patch('shaping_execution.load_registration',return_value=(dict(minimum_ssd_free_bytes=0),[],{})), \
                 patch('shaping_worker_collect.socket.gethostname',return_value='r3500'), \
                 patch('shaping_worker_collect.DECLARATION_SHA256',sha(declaration)):
                bundle=study/'result.tar';receipt=pack_reserved(study,path,'cell','attempt-001',bundle)
                root.parent.rename(study/'preserved-worker-cell')
                imported=Path(collect(study,path,bundle,receipt['bundle_sha256']))
                self.assertEqual((imported/'RESULT.json').read_bytes(),expected)
                self.assertEqual(validate(imported,assignment,[],study),run)
                imported_receipt=json.loads((imported/'WORKER_COLLECTION.json').read_text())
                self.assertEqual(imported_receipt['assignment_sha256'],sha(path))
                self.assertEqual(imported_receipt['bundle_sha256'],receipt['bundle_sha256'])
                self.assertEqual(len(list(imported.rglob('ckpt_epoch*.bin'))),240)

    def test_downstream_cannot_use_legacy_collection_without_original_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study=Path(directory);root,run,result,assignment,path,resolution,declaration=self.fixture(study,'outer','relu',[240])
            with self.assertRaisesRegex(ValueError,'original study gate'):validate(root,assignment,[run['cell']])


if __name__=='__main__':unittest.main()
