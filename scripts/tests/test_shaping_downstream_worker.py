import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_downstream_worker as downstream
import shaping_rescue_worker as worker
from shaping_completion import encoded,sha


class DownstreamWorkerTests(unittest.TestCase):
    def test_outer_identity_must_resolve_uniquely_and_variants_need_human_qualification(self):
        cell=dict(id='c',partition='outer-2')
        with patch('shaping_execution.load_registration',return_value=({},[cell],{})):
            args=downstream.arguments(Path('/study'),'c',Path('/bin'),Path('/q'),'outer')
            self.assertEqual(args.outer,2)
        with patch('shaping_execution.load_registration',return_value=({},[cell,cell],{})):
            with self.assertRaisesRegex(ValueError,'one registered outer'):downstream.arguments(Path('/study'),'c',Path('/bin'),Path('/q'),'outer')
        with self.assertRaisesRegex(ValueError,'human qualification'):downstream.arguments(Path('/study'),'c',Path('/bin'),Path('/q'),'variant')
        with self.assertRaisesRegex(ValueError,'execution kind'):downstream.arguments(Path('/study'),'c',Path('/bin'),Path('/q'),'rescue')

    def fixture(self,root,kind):
        study=root/'study';binaries=root/'bin';source=study/'runs/template/attempt-001'
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value));return path
        write(study/'PLAN.json',{});write(binaries/'BUILD.json',{})
        write(study/'policies/POLICIES.json',{'test':'snapshot present'})
        write(source/'RUN.json',{});write(source/'RESULT.json',{})
        write(source/'COLLECTION_DERIVED_INPUTS.json',{});write(source/'WORKER_COLLECTION.json',{})
        marker=write(study/'gate-input.json',{'gate':'audited'})
        admission=write(study/'selection/outer-0/OUTER_ADMISSION.json',{})
        write(study/'outer_inputs/outer-0/PREPARED.json',{})
        cell=dict(id='new',partition='outer-0',stage={'outer':'outer','variant':'human_only','extension':'exposure_extension'}[kind],family='relu')
        context=dict(cell=cell,schedule=dict(setting=1,phase_epochs=[240]),
                     templates=[(source,dict(command=['trainer','--derived-inputs','/old/declaration.json']))],
                     admission_path=admission,resolution=dict(file_sha256={str(marker):sha(marker)}))
        return study,binaries,marker,context

    def test_capsule_captures_gate_reads_and_refuses_drift_before_worker_execution(self):
        for kind in ('outer','variant','extension'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);study,binaries,marker,context=self.fixture(root,kind)
                def gate(args,execution_kind,host=None):
                    self.assertEqual(execution_kind,kind);self.assertEqual(host,'r3500')
                    marker.read_bytes();return context
                with patch.object(downstream,'arguments',return_value=SimpleNamespace()), \
                     patch.object(downstream,'admitted',side_effect=gate),patch.object(worker,'DATA_ROOTS',(root,)):
                    record=downstream.resolve_inputs(study,'new','r3500',binaries,root/'qualification',kind,root/'human-q')
                    self.assertIn(str(marker),record['files'])
                    self.assertEqual(record['files'][str(marker)]['sha256'],sha(marker))
                    self.assertEqual(record['schema'],'shaping-native-worker-v2')
                    self.assertEqual(record['resolution'],downstream.scope(context,kind))
                    with patch.object(worker.socket,'gethostname',return_value='r3500'):
                        self.assertEqual(worker.verify(record),[])
                        marker.write_text('changed gate evidence')
                        with self.assertRaisesRegex(ValueError,'drift'):worker.verify(record)

    def test_reproduction_reuses_original_owner_and_builds_its_original_cli(self):
        for kind in ('outer','variant','extension'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);study,binaries,marker,context=self.fixture(root,kind)
                record=dict(study=str(study),cell=context['cell'],binary_dir=str(binaries),
                            neutral_qualification=str(root/'q'),human_qualification=str(root/'hq'),execution_kind=kind)
                with patch.object(downstream,'arguments',return_value=SimpleNamespace(outer=0)), \
                     patch.object(downstream,'admitted',return_value=context) as gate:
                    resolution,command=downstream.reproduce(record)
                gate.assert_called_once();self.assertEqual(resolution,downstream.scope(context,kind))
                self.assertEqual(Path(command[2]).name,'shaping_'+kind+'_execution.py')
                self.assertEqual(command[command.index('--cell')+1],'new')
                if kind=='variant':self.assertEqual(command[command.index('--human-qualification')+1],record['human_qualification'])

    def test_worker_gate_mismatch_cannot_claim_cpu_or_create_an_attempt(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);request=root/'REQUEST.json'
            request.write_bytes(encoded(dict(study=str(root),cell=dict(id='new'),resolution={'setting':0},execution_kind='outer')))
            with patch.object(worker,'verify'),patch.object(downstream,'reproduce',return_value=({'setting':1},['owner'])), \
                 patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'reproduce original evidence'):worker.work(request)
                claim.assert_not_called();self.assertFalse((root/'runs/new').exists())

    def test_wrong_schema_stage_combination_is_rejected_before_dependencies(self):
        record=dict(schema='shaping-native-worker-v2',host='r3500',driver_sha256=sha(worker.__file__),
                    outer_results_read=False,execution_kind='outer',cell=dict(stage='human_only'))
        with patch.object(worker.socket,'gethostname',return_value='r3500'):
            with self.assertRaisesRegex(ValueError,'stage differs'):worker.verify(record)
            record.update(schema='shaping-rescue-worker-v1')
            with self.assertRaisesRegex(ValueError,'legacy rescue'):worker.verify(record)


if __name__=='__main__':unittest.main()
