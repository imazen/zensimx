import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_outer_execution import command,resolve_cell
from shaping_outer_plan import plan
from shaping_completion import matrix,FAMILIES

class OuterExecutionTests(unittest.TestCase):
    def test_refit_changes_training_rows_and_budget_not_selected_optimizer_or_seed(self):
        original=['run-heavy','--mem','1G','--jobs','1','--','/old/zensim_mlp_train',
                  '--group','old:inner-only','--epochs','240','--out','old.bin','--dump-checkpoints-dir','old',
                  '--lr','0.0003','--l2','0.00001','--init-seed','1107','--sample-seed','200000101',
                  '--keep-features','frozen.txt','--mse-weight','1']
        groups=[(n,'/outer/train/'+n,1.,0.,'withinref,both' if n=='cid22' else 'withinref,rank') for n in
                ('human_kadid','human_tid2013','human_konfig','human_cid22_a25','cid22')]
        frozen=copy.deepcopy(original)
        for epochs in (120,240,480):
            result=command(original,Path('/bin'),Path('/out'),epochs,groups)
            self.assertNotIn('old:inner-only',result);self.assertEqual(result.count('--group'),5)
            self.assertEqual(result[result.index('--epochs')+1],str(epochs))
            for flag in ('--jobs','--lr','--l2','--init-seed','--sample-seed','--keep-features','--mse-weight'):
                self.assertEqual(result[result.index(flag)+1],original[original.index(flag)+1])
        self.assertEqual(original,frozen)
        with self.assertRaises(ValueError):command(original,Path('/bin'),Path('/out'),119,groups)
        with self.assertRaises(ValueError):command(original,Path('/bin'),Path('/out'),120,groups[:-1])
        groups[0]=(*groups[0][:3],1.,groups[0][4])
        with self.assertRaisesRegex(ValueError,'TRAIN only'):command(original,Path('/bin'),Path('/out'),120,groups)

    def test_only_admitted_outer_cells_and_exact_fresh_tuple_resolve(self):
        cells=matrix();decisions={f:dict(family=f,outer=0,outer_results_read=False,setting=0,optimizer_choice='base',inner_score=.8) for f in FAMILIES}
        admission=plan(decisions,['relu','product_pretrained_joint'],0,cells)
        primary=next(c for c in cells if c['stage']=='outer' and c['family']=='product_pretrained_joint' and c['partition']=='outer-0')
        with self.assertRaisesRegex(ValueError,'not admitted'):resolve_cell(primary,admission)
        admission['outer_evaluation_admitted']=True
        self.assertEqual(resolve_cell(primary,admission)['phase_epochs'],[240,240])
        fresh=admission['additional_cells'][0];result=resolve_cell(fresh,admission)
        self.assertTrue(result['fresh']);self.assertEqual(result['phase_epochs'],[240])
        with self.assertRaisesRegex(ValueError,'exactly admitted'):resolve_cell(dict(fresh,epochs=120),admission)
        with self.assertRaisesRegex(ValueError,'not admitted'):resolve_cell(dict(primary,partition='outer-1'),admission)

if __name__=='__main__':unittest.main()
