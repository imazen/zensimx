import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_corrected_worker as corrected
import shaping_rescue_worker as worker
from shaping_completion import encoded,sha


class CorrectedWorkerTests(unittest.TestCase):
    def test_unactivated_or_historical_cell_cannot_reach_native_owner(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'BUILD.json').write_text('{}')
            with patch('shaping_execution.load_registration',return_value=({},[],{})),patch('shaping_execution.run_native') as native:
                with self.assertRaisesRegex(ValueError,'effective corrected'):corrected.fit(root,'new',root)
                native.assert_not_called()

    def test_transport_rechecks_corrected_gate_and_captures_parent_bank(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';study.mkdir();source=study/'runs/parent/attempt-001';source.mkdir(parents=True)
            (study/'PLAN.json').write_text('{}');(source/'SHAPING.json').write_text('{}')
            marker=study/'corrected-evidence.json';marker.write_text('{}');binaries=root/'binaries';binaries.mkdir();(binaries/'BUILD.json').write_text('{}')
            cell=dict(id='corrected',family='product_pretrained_joint',partition='inner-0-0',stage='inner')
            context=dict(cell=cell,parent=source,scope=dict(cell=cell,outer_results_read=False,prefix_sha256='prefix'))
            def gate(*args,**kwargs):marker.read_bytes();return context
            qualification=root/'corrected-control-qualification-20261108'
            with patch.object(corrected,'context',side_effect=gate),patch.object(worker,'DATA_ROOTS',(root,)):
                record=corrected.resolve_inputs(study,'corrected','r3500',binaries,qualification)
                self.assertEqual(record['execution_kind'],'corrected_inner')
                self.assertIn(str(marker),record['files']);self.assertIn(str(source/'SHAPING.json'),record['files'])
                with patch.object(worker.socket,'gethostname',return_value='r3500'):
                    self.assertEqual(worker.verify(record),[])
                    marker.write_text('drift')
                    with self.assertRaisesRegex(ValueError,'drift'):worker.verify(record)
            with patch.object(corrected,'context',return_value=context) as gate:
                scope,command=corrected.reproduce(record,host='r3500')
                self.assertEqual(scope,context['scope']);self.assertEqual(Path(command[2]).name,'shaping_corrected_worker.py')
                self.assertEqual(command[command.index('--cell')+1],'corrected')

    def test_changed_gate_refuses_before_cpu_claim(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);request=root/'REQUEST.json'
            request.write_bytes(encoded(dict(study=str(root),cell=dict(id='new'),resolution={'prefix':'original'},execution_kind='corrected_inner')))
            with patch.object(worker,'verify'),patch.object(corrected,'reproduce',return_value=({'prefix':'changed'},['owner'])),patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'reproduce original evidence'):worker.work(request)
                claim.assert_not_called();self.assertFalse((root/'runs/new').exists())


if __name__=='__main__':unittest.main()
