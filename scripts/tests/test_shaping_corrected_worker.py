import json,sys,tempfile,unittest,os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_corrected_worker as corrected
import shaping_rescue_worker as worker
from shaping_completion import encoded,sha


class CorrectedWorkerTests(unittest.TestCase):
    def test_strict_native_command_uses_the_admitted_binary_override(self):
        import v2_lodo_mlp as owner
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);binaries=root/'admitted'
            def command():return owner.train_command([],1101,101,1853,root/'keep.txt','N',root/'best.bin',{},strict_admission=True)
            with patch.dict(os.environ,{'REV4_V2_BIN_DIR':str(root/'wrong')}),patch.object(owner,'strict_training_groups'),patch.object(owner,'strict_group_input_roots',return_value=()):
                # Merely setting TRAINER does not control the strict builder.
                with patch.object(owner,'TRAINER',binaries/'zensim_mlp_train'):
                    self.assertNotEqual(Path(command()[0]),binaries/'zensim_mlp_train')
                def native(*args):
                    cmd=command();self.assertEqual(cmd[cmd.index('--epochs')+1],'120');return cmd
                with patch.object(corrected,'context',return_value=dict(cell={},groups={})),patch('shaping_execution.run_native',side_effect=native):
                    self.assertEqual(Path(corrected.fit(root,'new',binaries)[0]),binaries/'zensim_mlp_train')

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
            guard=study/'worker_assignments/selected.json';guard.parent.mkdir();guard.write_text('{}')
            other=guard.with_name('historical.json');other.write_text('{}')
            context=dict(cell=cell,parent=source,scope=dict(cell=cell,outer_results_read=False,prefix_sha256='prefix',placement=dict(assignment=str(guard))))
            def gate(*args,**kwargs):marker.read_bytes();guard.read_bytes();other.read_bytes();return context
            qualification=root/'corrected-control-qualification-20261108'
            with patch.object(corrected,'context',side_effect=gate),patch.object(worker,'DATA_ROOTS',(root,)):
                record=corrected.resolve_inputs(study,'corrected','r3500',binaries,qualification)
                self.assertEqual(record['execution_kind'],'corrected_inner')
                self.assertIn(str(marker),record['files']);self.assertIn(str(source/'SHAPING.json'),record['files'])
                self.assertIn(str(guard),record['files']);self.assertNotIn(str(other),record['files'])
                with patch.object(worker.socket,'gethostname',return_value='r3500'):
                    self.assertEqual(worker.verify(record),[])
                    marker.write_text('drift')
                    with self.assertRaisesRegex(ValueError,'drift'):worker.verify(record)
            with patch.object(corrected,'context',return_value=context) as gate:
                scope,command=corrected.reproduce(record,host='r3500')
                self.assertEqual(scope,context['scope']);self.assertTrue(any(Path(arg).name=='shaping_corrected_worker.py' for arg in command))
                self.assertEqual(command[command.index('--mem')+1],'512M')
                self.assertEqual(command[command.index('--cell')+1],'corrected')

    def test_changed_gate_refuses_before_cpu_claim(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);request=root/'REQUEST.json'
            request.write_bytes(encoded(dict(study=str(root),cell=dict(id='new'),resolution={'prefix':'original'},execution_kind='corrected_inner')))
            with patch.object(worker,'verify'),patch.object(corrected,'reproduce',return_value=({'prefix':'changed'},['owner'])),patch('shaping_rescue_dispatch.reserve_worker') as claim:
                with self.assertRaisesRegex(ValueError,'reproduce original evidence'):worker.work(request)
                claim.assert_not_called();self.assertFalse((root/'runs/new').exists())


if __name__=='__main__':unittest.main()
