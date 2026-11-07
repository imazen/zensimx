import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_convergence import diagnostics,UndefinedInnerComposite


class ConvergenceTests(unittest.TestCase):
    def test_same_cycle_pair_and_loss_trigger(self):
        scores={49:.8,99:.8,119:.8,149:.8,199:.8,219:.803,239:.9}
        result=diagnostics([1.]*20+[.98]*20,scores,{k:4 for k in scores})
        self.assertEqual(result['same_cycle_epochs'],[119,219])
        self.assertEqual(len(result['reasons']),2)

    def test_one_coefficient_and_threshold_uncertainty_are_not_convergence(self):
        scores={49:.8,99:.8,119:.8}
        result=diagnostics([1.]*40,scores,{49:1,99:3,119:3})
        self.assertEqual(result['reasons'],['one_output_coefficient'])
        self.assertTrue(diagnostics([1.]*20+[.99]*20,scores,{k:3 for k in scores})['investigation_required'])
        self.assertFalse(diagnostics([1.]*40,scores,{k:3 for k in scores})['investigation_required'])

    def test_undefined_same_cycle_scores_remain_refused(self):
        for missing in (None,float('nan'),float('inf')):
            for scores in ({49:missing,99:.8},{49:.8,99:missing}):
                with self.assertRaises(UndefinedInnerComposite):
                    diagnostics([1.]*40,scores,{49:0,99:0})

    def test_missing_and_nonfinite_losses_refused(self):
        for losses in ([1.]*39,[float('nan')]*40):
            with self.assertRaises(ValueError):diagnostics(losses,{49:.8,99:.8},{49:3,99:3})


if __name__=='__main__':unittest.main()

class TrainingOnlyTests(unittest.TestCase):
    def test_complete_phase_logs_preserve_unknown_human_trend(self):
        import tempfile,json
        from shaping_completion import encoded,sha
        from shaping_convergence import training_report
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            run=Path(directory)
            def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(encoded(d))
            cell=dict(id='full-fit-search',stage='full_fit_selection')
            identity=dict(cell=cell,selection_scope='full_fit_only',development_results_read=False,
                          epochs=120,plan_sha256='plan',schedule={'phase_epochs':[120,120]})
            phases=[]
            for name in ['pretraining','fit']:
                p=run/name/'checkpoints/ckpt_epoch119.bin';p.parent.mkdir(parents=True);p.write_bytes(name.encode())
                log=run/name/'train.log';log.write_text(''.join(f'epoch {i} | loss={1. if i<100 else .98}\n' for i in range(120)))
                phase=dict(epochs=120,stage='trained',exit_code=0,bake_sha256=sha(p));write(run/name/'RESULT.json',phase);phases.append(phase)
            write(run/'RUN.json',identity);write(run/'RESULT.json',dict(cell=cell,stage='trained',exit_code=0,phases=phases))
            report=run/'evaluation/REPORT.json'
            row=dict(cell=cell,selection_scope='full_fit_only',development_results_read=False,checkpoint_epoch=119,
                     plan_sha256='plan',active_output_coefficients=1,full_fit_validation={'selection_score':-999})
            write(report,row);write(report.parent/'COMPLETE.json',{'report_sha256':sha(report)})
            result=training_report(run,report)
            self.assertEqual(result['conclusion'],'inconclusive_training_only');self.assertFalse(result['validation_scores_used'])
            self.assertEqual(len(result['observed_training_triggers']),3)
            row['full_fit_validation']['selection_score']=999;write(report,row);write(report.parent/'COMPLETE.json',{'report_sha256':sha(report)})
            changed=training_report(run,report)
            self.assertEqual(changed['observed_training_triggers'],result['observed_training_triggers'])
            self.assertEqual(changed['phase_loss_windows'],result['phase_loss_windows'])
            cell['stage']='full_fit';identity['selection_scope']='full_fit_refit'
            write(run/'RUN.json',identity);write(run/'RESULT.json',dict(cell=cell,stage='trained',exit_code=0,phases=phases))
            row.update(selection_scope='descriptive_only',development_results_read=True)
            row['exposed_development']=row.pop('full_fit_validation')
            write(report,row);write(report.parent/'COMPLETE.json',{'report_sha256':sha(report)})
            final=training_report(run,report)
            self.assertEqual(final['selection_scope'],'full_fit_refit');self.assertFalse(final['development_results_used'])
            self.assertEqual(final['phase_loss_windows'],result['phase_loss_windows'])
            with (run/'fit/train.log').open('a') as stream:stream.write('epoch 119 | loss=.98\n')
            with self.assertRaisesRegex(ValueError,'coverage'):training_report(run,report)


class StructureTraceTests(unittest.TestCase):
    def test_all_epochs_required_and_native_reader_pinned(self):
        import json,tempfile
        from unittest.mock import patch
        from shaping_convergence import structure_trace
        from shaping_completion import encoded,sha
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);run=root/'run';binary=root/'bin';run.mkdir();binary.mkdir()
            reader=binary/'bake_dial_refit';reader.write_text('fixture native reader')
            build=binary/'BUILD.json';build.write_bytes(encoded({'binary_sha256':{'bake_dial_refit':sha(reader)}}))
            cell={'id':'fixture','stage':'inner'}
            (run/'RUN.json').write_bytes(encoded(dict(cell=cell,epochs=2,build_sha256=sha(build))))
            (run/'RESULT.json').write_bytes(encoded(dict(cell=cell,stage='trained',exit_code=0)))
            checkpoints=run/'checkpoints';checkpoints.mkdir()
            (checkpoints/'ckpt_epoch000.bin').write_bytes(b'first')
            output=root/'trace.json'
            with patch('subprocess.check_output') as native:
                with self.assertRaisesRegex(ValueError,'complete scheduled'):structure_trace(run,binary,output)
                native.assert_not_called()
            (checkpoints/'ckpt_epoch001.bin').write_bytes(b'second')
            layers=[{'stored_weights':64,'zero_weights':60}]
            with patch('subprocess.check_output',return_value=json.dumps({'layers':layers})) as native:
                record=structure_trace(run,binary,output)
                self.assertEqual(native.call_count,2)
                self.assertEqual([r['epoch'] for r in record['checkpoints']],[0,1])
                self.assertFalse(record['scores_computed']);self.assertFalse(record['checkpoint_selection'])
                with self.assertRaisesRegex(ValueError,'preserve previous'):structure_trace(run,binary,output)
            reader.write_text('changed')
            with self.assertRaisesRegex(ValueError,'reader identity'):structure_trace(run,binary,root/'other.json')
