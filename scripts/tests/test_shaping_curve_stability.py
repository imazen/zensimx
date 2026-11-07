import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_curve_stability import compare,parameter_activity,checkpoint_config
from shaping_completion import encoded,sha


class CurveStabilityTests(unittest.TestCase):
    def config(self,separate=False):
        return dict(schema='zensim-input-shaping-v1',terms=[dict(column=1,damage=1,reference=2 if separate else None,separate=separate,knots=[1.,2.],slopes=[1.,1.,1.],gate=[0.,.5,1.])])
    def test_linear_shape_is_knot_and_strength_invariant(self):
        a=self.config();b=copy.deepcopy(a);b['terms'][0].update(knots=[2.,3.],slopes=[2.,2.,2.])
        r=compare([a,b])[0];self.assertEqual(r['normalized_response_max_spread'],0.);self.assertEqual(r['strength_ratio'],2.)
    def test_equivalent_f32_spellings_have_identical_curves(self):
        a=self.config();b=copy.deepcopy(a);b['terms'][0]['slopes'][0]=1.00000001
        original=copy.deepcopy(b)
        self.assertEqual(compare([a,b])[0]['normalized_response_max_spread'],0.)
        self.assertEqual(b,original)
    def test_separate_gate_difference_is_not_hidden_by_damage_normalization(self):
        a=self.config(True);b=copy.deepcopy(a);b['terms'][0]['gate']=[1.,1.,0.]
        r=compare([a,b])[0];self.assertEqual(r['normalized_response_max_spread'],1.)
        b['terms'][0]['gate']=[0.,0.,0.];r=compare([a,b])[0];self.assertEqual(r['active_fits'],1);self.assertIsNone(r['normalized_response_max_spread'])
    def test_zero_terms_and_invalid_terms_are_not_stable_useful_candidates(self):
        a=self.config();a['terms'][0]['slopes']=[0.,0.,0.]
        r=compare([a,a])[0];self.assertEqual(r['active_fits'],0);self.assertIsNone(r['strength_ratio'])
        a['terms'][0]['slopes'][0]=-1.
        with self.assertRaises(ValueError):compare([a])

    def test_zero_paths_use_wire_precision_and_ignore_unused_product_gate(self):
        product=self.config();product['terms'][0]['gate']=[0.,0.,0.]
        self.assertEqual(parameter_activity(product)['nonzero_parameter_paths'],1)
        separate=self.config(True);separate['terms'][0]['gate']=[0.,0.,0.]
        separate['terms'][0]['slopes']=[1e-50,0.,0.]
        row=parameter_activity(separate)
        self.assertEqual(row['zero_damage_terms'],1);self.assertEqual(row['zero_reference_gates'],1)
        self.assertEqual(row['zero_parameter_paths'],1);self.assertEqual(row['nonzero_parameter_paths'],0)

    def test_diagnostic_checkpoint_is_bound_to_its_report_and_model(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);run=root/'run';(run/'checkpoints').mkdir(parents=True)
            prefix=run/'checkpoints/ckpt_epoch119.bin';prefix.write_bytes(b'prefix')
            final=run/'checkpoints/ckpt_epoch239.bin';final.write_bytes(b'final')
            cell={'id':'cell'};final_report=dict(cell=cell,plan_sha256='plan',checkpoint_epoch=239,
                checkpoint_sha256=sha(final),outer_results_read=False,active_output_coefficients=1)
            row=dict(final_report,checkpoint_epoch=119,checkpoint_sha256=sha(prefix),active_output_coefficients=3)
            folder=run/'curve_evaluation/epoch-119';folder.mkdir(parents=True)
            report=folder/'REPORT.json';report.write_bytes(encoded(row))
            marker=folder/'DIAGNOSTIC_COMPLETE.json';marker.write_bytes(encoded(dict(report_sha256=sha(report))))
            pins={}
            with patch('shaping_curve_stability.subprocess.check_output',return_value=json.dumps(self.config())) as read:
                config,epoch,active=checkpoint_config(root,run,cell,240,119,final_report,pins)
            self.assertEqual((epoch,active),(119,3));self.assertIn(str(prefix),pins);self.assertIn(str(report),pins)
            self.assertNotIn(str(final),pins);self.assertEqual(read.call_args.args[0][3],str(prefix))
            self.assertEqual(config,self.config())
            report.write_text('{}')
            with patch('shaping_curve_stability.subprocess.check_output') as read:
                with self.assertRaisesRegex(ValueError,'diagnostic report drift'):
                    checkpoint_config(root,run,cell,240,119,final_report,{})
                read.assert_not_called()

    def test_unregistered_checkpoint_is_refused_before_metadata_read(self):
        with patch('shaping_curve_stability.subprocess.check_output') as read:
            with self.assertRaisesRegex(ValueError,'unregistered diagnostic checkpoint'):
                checkpoint_config(Path('/unused'),Path('/unused'),{},240,1,{}, {})
            read.assert_not_called()


if __name__=='__main__':unittest.main()

class PairedSettingTests(unittest.TestCase):
    def test_explicit_partitions_preserve_each_seed_architecture_and_setting(self):
        from shaping_completion import matrix
        from shaping_curve_stability import selected_cells
        cells=matrix();partitions=['inner-0-0','inner-0-1','inner-1-1','inner-2-0']
        chosen=selected_cells(cells,'direct_joint_both',None,[0],partitions=partitions)
        self.assertEqual(len(chosen),24)
        self.assertEqual({c['partition'] for c in chosen},set(partitions))
        for partition in partitions:
            self.assertEqual({(c['family'],c['seed']) for c in chosen if c['partition']==partition},
                             {(f,s) for f in ['product_joint_identity','separate_joint_identity'] for s in range(3)})
        with self.assertRaisesRegex(ValueError,'complete paired'):
            selected_cells([c for c in cells if c!=chosen[0]],'direct_joint_both',None,[0],partitions=partitions)
        for outer,fold,parts in [(0,None,partitions),(None,0,partitions),(None,None,partitions+partitions[:1]),(None,None,['outer-0'])]:
            with self.assertRaisesRegex(ValueError,'distinct registered partitions'):
                selected_cells(cells,'direct_joint_both',outer,[0],fold,parts)

    def test_two_settings_three_seeds_per_architecture_without_other_fold(self):
        from shaping_completion import matrix
        from shaping_curve_stability import selected_cells
        cells=matrix();chosen=selected_cells(cells,'direct_joint_both',0,[0,1],1)
        self.assertEqual(len(chosen),12)
        self.assertEqual({c['partition'] for c in chosen},{'inner-0-1'})
        self.assertEqual({c['setting'] for c in chosen},{0,1})
        with self.assertRaisesRegex(ValueError,'complete paired'):
            selected_cells([c for c in cells if c!=chosen[0]],'direct_joint_both',0,[0,1],1)
        with self.assertRaisesRegex(ValueError,'distinct settings'):
            selected_cells(cells,'direct_joint_both',0,[0,0],1)
        old=selected_cells(cells,'product_joint_identity',0,[0])
        self.assertEqual(len(old),6);self.assertEqual({c['partition'] for c in old},{'inner-0-0','inner-0-1'})
