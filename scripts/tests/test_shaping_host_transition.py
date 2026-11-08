import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_host_assignment import transition_records

class TransitionTests(unittest.TestCase):
    def cells(self):
        return [dict(id=str(i),stage='inner' if i<5 else 'rescue',family='product_pretrained_joint',
                     partition='inner-0-0',setting=0,seed=i%3) for i in range(6)]
    def record(self,host,ids):
        return dict(host=host,cells=ids,partition='inner-0-0',plan_sha256='p',build_sha256='b')
    def test_started_failed_reserved_and_allowed_worker_cells_stay(self):
        old={'a.json':self.record('dev',['0','1','2']),'b.json':self.record('r3800x',['3'])}
        original=repr(old)
        q=transition_records(old,self.cells(),{'0','1'},['dev'],['i134','r5600g'],'p','b')
        self.assertEqual(q['moved'],['2','4']);self.assertEqual(q['replaced']['a.json']['cells'],['0','1'])
        self.assertNotIn('b.json',q['replaced']);self.assertEqual(repr(old),original)
        self.assertEqual({r['host'] for r in q['created'].values()},{'i134','r5600g'})
        self.assertTrue(all(r['parent_capsule_required'] for r in q['created'].values()))
        self.assertNotIn('5',q['moved']);self.assertFalse(q['training_replayed'])
    def test_all_started_means_no_change(self):
        q=transition_records({},self.cells(),set('01234'),['dev'],['i134'],'p','b')
        self.assertEqual(q['created'],{});self.assertEqual(q['replaced'],{})
    def test_overlap_unknown_build_and_retired_target_refuse(self):
        cases=[({'a':self.record('dev',['0']),'b':self.record('dev',['0'])},['i134']),
               ({'a':self.record('dev',['99'])},['i134']),
               ({'a':dict(self.record('dev',['0']),build_sha256='other')},['i134']),
               ({},['dev'])]
        for records,targets in cases:
            with self.subTest(records=records,targets=targets),self.assertRaises(ValueError):
                transition_records(records,self.cells(),set(),['dev'],targets,'p','b')

if __name__=='__main__':unittest.main()
