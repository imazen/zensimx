import sys
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_evaluation import admitted_epoch, evaluation_epoch, reexport_bake, prepare_validation, outer_context, training_targets
from shaping_completion import sha


class EvaluationAdmissionTests(unittest.TestCase):
    def test_shuffled_training_targets_leave_actual_outer_targets_intact(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path=Path(directory)/'train.parquet'
            pq.write_table(pa.table({'human_score':[30.,10.,20.]}),path)
            labels=np.array([10.,91.,20.,92.,30.]);positions={100:0,101:1,102:2,103:3,104:4}
            training_targets(labels,positions,[100,102,104],path)
            np.testing.assert_array_equal(labels,[30.,91.,10.,92.,20.])
            with self.assertRaisesRegex(ValueError,'coverage'):training_targets(labels,positions,[100],path)

    def test_variant_outer_context_revalidates_resolution_and_qualification(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);gate=root/'admission';gate.write_text('{}')
            qualification=root/'qualification';qualification.write_text('{}')
            cell=dict(id='v',stage='shuffle',partition='outer-1',family='best_analytic')
            resolution=dict(schedule={'phase_epochs':[480]},resolved_family='silu_diagnostic')
            run=dict(cell=cell,resolution=resolution,qualification_sha256={str(qualification):sha(qualification)},epochs=480)
            membership=dict(train=[1],test=[2])
            with patch('shaping_outer_admission.resolve',return_value=({'additional_cells':[]},gate)),\
                 patch('shaping_variant_execution.resolve',return_value=resolution),\
                 patch('shaping_variant_inputs.load_variant',return_value=({},cell,'id',membership)):
                context=outer_context(root,run,[cell]);self.assertEqual(context['membership'],membership)
                self.assertEqual(context['variant']['resolved_family'],'silu_diagnostic')
                self.assertEqual(admitted_epoch(run,dict(stage='trained',exit_code=0),[cell],context),480)
                with self.assertRaisesRegex(ValueError,'resolution drift'):outer_context(root,dict(run,resolution={}),[cell])
                with self.assertRaisesRegex(ValueError,'cannot diagnose'):outer_context(root,run,[cell],479)
                qualification.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'qualification drift'):outer_context(root,run,[cell])

    def test_cached_outer_view_cannot_bypass_live_admission(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);dest=root/'evaluation_views/outer-0';dest.mkdir(parents=True)
            receipt=dest/'RECEIPT.json'
            receipt.write_text(json.dumps(dict(indices=[1],groups=[],admission_sha256='stale')))
            with patch('shaping_outer_admission.resolve',side_effect=ValueError('incomplete inner comparisons')):
                with self.assertRaisesRegex(ValueError,'incomplete inner'):prepare_validation(root,'outer-0',{'test':[1]})
            gate=root/'OUTER_ADMISSION.json';gate.write_text('{}')
            with patch('shaping_outer_admission.resolve',return_value=({},gate)):
                with self.assertRaisesRegex(ValueError,'admission drift'):prepare_validation(root,'outer-0',{'test':[1]})
                receipt.write_text(json.dumps(dict(indices=[1],groups=[],admission_sha256=sha(gate))))
                self.assertEqual(prepare_validation(root,'outer-0',{'test':[1]})['indices'],[1])

    def test_outer_context_requires_exact_schedule_cell_and_prepared_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);gate=root/'OUTER_ADMISSION.json';gate.write_text('{}')
            cell=dict(id='outer',stage='outer',partition='outer-0',family='relu')
            schedule=dict(phase_epochs=[480],total_pair_draws=24000000,setting=2,optimizer_choice='base')
            admission=dict(outer=0,outer_evaluation_admitted=True,additional_cells=[],schedules={'relu':schedule})
            run=dict(cell=cell,schedule=dict(schedule,fresh=False),admission_sha256=sha(gate),epochs=480)
            with patch('shaping_outer_admission.resolve',return_value=(admission,gate)),patch('shaping_outer_execution.prepared') as prepared:
                context=outer_context(root,run,[cell]);prepared.assert_called_once_with(root,0,gate)
                self.assertEqual(admitted_epoch(run,dict(stage='trained',exit_code=0),[cell],context),480)
                for bad in (dict(run,admission_sha256='wrong'),dict(run,schedule=dict(schedule,fresh=True))):
                    with self.assertRaises(ValueError):outer_context(root,bad,[cell])
                with self.assertRaises(ValueError):outer_context(root,run,[])
                with self.assertRaisesRegex(ValueError,'cannot diagnose'):outer_context(root,run,[cell],479)
                with self.assertRaises(ValueError):admitted_epoch(dict(run,epochs=240),dict(stage='trained',exit_code=0),[cell],context)
                prepared.side_effect=ValueError('outer TRAIN input drift')
                with self.assertRaisesRegex(ValueError,'TRAIN input drift'):outer_context(root,run,[cell])

    def test_rescue_fresh_phase_retains_its_registered_budget(self):
        result={'stage':'trained','exit_code':0}
        for family,epochs in [('relu',240),('product_pretrained_frozen',120),('separate_pretrained_joint',120)]:
            cell=dict(id=family,stage='rescue',family=family)
            self.assertEqual(admitted_epoch(dict(cell=cell,epochs=epochs),result,[cell]),epochs)
            with self.assertRaises(ValueError):admitted_epoch(dict(cell=cell,epochs=epochs+1),result,[cell])

    def test_reexport_is_bound_to_original_runtime_and_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            original=root/'original';original.write_bytes(b'original')
            bake=root/'repaired';bake.write_bytes(b'repaired')
            build=root/'BUILD.json';build.write_text('{}')
            receipt=root/'RECEIPT.json'
            record=dict(representation='research-relu-max-pairs-v1',original_sha256=sha(original),
                        training_build_sha256='training',evaluation_build_sha256=sha(build),
                        bake=str(bake),file_sha256={str(bake):sha(bake)})
            receipt.write_text(json.dumps(record))
            run=dict(build_sha256='training')
            self.assertEqual(reexport_bake(receipt,original,run,build),bake)
            for key in ('representation','original_sha256','training_build_sha256','evaluation_build_sha256'):
                receipt.write_text(json.dumps({**record,key:'wrong'}))
                with self.assertRaises(ValueError):reexport_bake(receipt,original,run,build)
            receipt.write_text(json.dumps(record));bake.write_bytes(b'changed')
            with self.assertRaises(ValueError):reexport_bake(receipt,original,run,build)

    def test_diagnostics_do_not_change_final_selection(self):
        self.assertEqual(evaluation_epoch(240,None),239)
        self.assertEqual(evaluation_epoch(120,None),119)
        self.assertEqual(evaluation_epoch(240,49),49)
        for budget,epoch in [(120,149),(240,0),(240,238),(240,240)]:
            with self.assertRaises(ValueError):evaluation_epoch(budget,epoch)

    def test_fresh_control_keeps_its_separate_half_exposure(self):
        cell={'id':'fresh','stage':'fresh_control_inner','family':'relu'}
        result={'stage':'trained','exit_code':0}
        self.assertEqual(admitted_epoch({'cell':cell,'epochs':120},result,[cell]),120)
        with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':240},result,[cell])

    def test_final_scheduled_epoch_is_required(self):
        cell={'id':'x','stage':'inner','family':'relu'}
        result={'stage':'trained','exit_code':0}
        self.assertEqual(admitted_epoch({'cell':cell,'epochs':240},result,[cell]),240)
        for epochs in (1,119,120,239):
            with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':epochs},result,[cell])

    def test_outer_pilot_unregistered_and_failed_runs_are_refused(self):
        for stage in ('outer','resource_pilot','full_fit'):
            cell={'id':'x','stage':stage,'family':'relu'}
            with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':240},{'stage':'trained','exit_code':0},[cell])
        cell={'id':'x','stage':'inner','family':'relu'}
        with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':240},{'stage':'trained','exit_code':0},[])
        for stage,code in [('running',0),('failed',1),('trained',1)]:
            with self.assertRaises(ValueError):admitted_epoch({'cell':cell,'epochs':240},{'stage':stage,'exit_code':code},[cell])


if __name__=='__main__':unittest.main()
