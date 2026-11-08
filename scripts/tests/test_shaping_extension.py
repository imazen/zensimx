import copy
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import NEURAL
from shaping_extension import matrix


def selections():
    return {family:dict(family=family,outer=0,outer_results_read=False,base_setting=1,
                       selected_choice='constant_lr',selected_convergence={
                           f'{fold}-{seed}':dict(cell=dict(partition=f'inner-0-{fold}',seed=seed),reasons=[])
                           for fold in range(2) for seed in range(3)}) for family in NEURAL}


class ExtensionTests(unittest.TestCase):
    def test_reduced_scope_preserves_paired_extension_requirements(self):
        required=('relu','softplus','product_joint_identity','separate_joint_identity')
        decisions={k:v for k,v in selections().items() if k in required}
        decisions['softplus']['selected_convergence']['0-0']['reasons']=[
            'loss_improvement_over_one_percent_or_rounding_boundary']
        result=matrix(decisions,0,candidate_families=required)
        self.assertEqual(set(result['extended_families']),{'relu','softplus'})
        self.assertEqual(len(result['cells']),12)
        self.assertEqual({(c['family'],c['partition'],c['seed']) for c in result['cells']},
                         {(f,f'inner-0-{fold}',seed) for f in ('relu','softplus')
                          for fold in range(2) for seed in range(3)})
        del decisions['relu']
        with self.assertRaisesRegex(ValueError,'all neural'):
            matrix(decisions,0,candidate_families=required)
        with self.assertRaisesRegex(ValueError,'four-role'):
            matrix(decisions,0,candidate_families=('softplus','product_joint_identity','separate_joint_identity'))

    def test_extended_diagnostics_compare_the_final_cycle_without_changing_old_schedules(self):
        from shaping_evaluation import diagnostic_epochs,evaluation_epoch,admitted_epoch
        from shaping_convergence import diagnostics
        self.assertEqual(diagnostic_epochs(120),(49,99,119))
        self.assertEqual(diagnostic_epochs(240),(49,99,119,149,199,219,239))
        scores={e:.8 for e in diagnostic_epochs(480)};scores[479]=.804
        d=diagnostics([1.]*480,scores,{e:64 for e in scores})
        self.assertEqual(d['same_cycle_epochs'],[429,479]);self.assertTrue(d['investigation_required'])
        with self.assertRaises(ValueError):evaluation_epoch(240,429)
        cell=dict(stage='exposure_extension',family='relu',phase_epochs=[480])
        run=dict(cell=cell,epochs=480);result=dict(stage='trained',exit_code=0)
        self.assertEqual(admitted_epoch(run,result,[cell]),480)
        with self.assertRaises(ValueError):admitted_epoch(dict(run,epochs=240),result,[cell])

    def test_native_command_changes_only_registered_exposure_and_fresh_seeds(self):
        from shaping_extension_execution import command,fresh_command,optimizer
        original=dict(command=['run-heavy','--mem','1G','--jobs','1','--','/old/zensim_mlp_train',
                               '--epochs','240','--out','old.bin','--dump-checkpoints-dir','old-checkpoints',
                               '--init-seed','1107','--sample-seed','200000101','--lr','0.001','--group','unchanged'])
        frozen=copy.deepcopy(original);fresh=fresh_command(original,1)
        self.assertEqual(original,frozen)
        self.assertEqual(fresh['command'][fresh['command'].index('--init-seed')+1],'1001107')
        self.assertEqual(fresh['command'][fresh['command'].index('--sample-seed')+1],'201000101')
        result=command(original['command'],Path('/binaries'),Path('/output'),480)
        self.assertEqual(result[result.index('--epochs')+1],'480');self.assertEqual(result[result.index('--mem')+1],'4G')
        for flag in ('--jobs','--init-seed','--sample-seed','--lr','--group'):
            self.assertEqual(result[result.index(flag)+1],original['command'][original['command'].index(flag)+1])
        with self.assertRaises(ValueError):command(original['command'],Path('/bin'),Path('/out'),481)
        with self.assertRaises(ValueError):fresh_command(original,0)
        control=optimizer('base',True,[dict(name='cid22',families=[0])])
        self.assertEqual((control['init_scale'],control['bias_scale'],control['schedule']),(1.,1.,'cosine'))
        self.assertEqual((control['teacher_mass'],control['dropout'],control['balance']),(1.,0.,'none'))

    def test_plateau_collapse_is_not_mislabeled_as_continuing_improvement(self):
        d=selections();d['softplus']['selected_convergence']['0-0']['reasons']=['one_output_coefficient']
        self.assertEqual(matrix(d,0)['cells'],[])
        del d['relu']
        with self.assertRaisesRegex(ValueError,'all neural'):matrix(d,0)

    def test_improvement_extends_every_pair_and_relu_control(self):
        d=selections();d['softplus']['selected_convergence']['0-0']['reasons']=['loss_improvement_over_one_percent_or_rounding_boundary']
        r=matrix(d,0)
        self.assertEqual(r['extended_families'],['relu','softplus']);self.assertEqual(len(r['cells']),12)
        self.assertTrue(all(c['total_pair_draws']==24000000 and c['phase_epochs']==[480] for c in r['cells']))
        self.assertTrue(all(c['optimizer_choice']=='constant_lr' and c['setting']==1 for c in r['cells']))
        d['softplus']['selected_convergence'].pop('1-2')
        with self.assertRaisesRegex(ValueError,'pairing'):matrix(d,0)

    def test_pretrained_exposure_and_fresh_controls_are_paired_and_deduplicated(self):
        d=selections()
        for family in ('product_pretrained_frozen','product_pretrained_joint'):
            d[family]['selected_convergence']['1-2']['reasons']=['same_cycle_inner_composite_improvement_over_0.002']
        original=copy.deepcopy(d);r=matrix(d,0)
        self.assertEqual(d,original);self.assertEqual(len(r['cells']),24)
        fresh=[c for c in r['cells'] if c['role']=='fresh_control']
        self.assertEqual(len(fresh),6)
        self.assertTrue(all(c['phase_epochs']==[240] and c['total_pair_draws']==12000000 for c in fresh))
        self.assertTrue(all(len(c['paired_families'])==2 for c in fresh))
        self.assertTrue(all(c['phase_epochs']==[240,240] for c in r['cells'] if '_pretrained_' in c['family']))
        d['product_pretrained_joint']['selected_choice']='init_double'
        self.assertEqual(len(matrix(d,0)['cells']),30)


if __name__=='__main__':unittest.main()
