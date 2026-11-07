import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_pretrained_dispatch as module
from shaping_completion import sha


class PretrainedDispatchTests(unittest.TestCase):
    def test_cli_ownership_does_not_precreate_initial_audit_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d)
            def dispatch(args):
                self.assertFalse((root/'dispatch/pretrained/inner-0-0').exists())
                self.assertTrue((root/'dispatch/pretrained/locks/inner-0-0.lock').exists())
            argv=['dispatch','--study',str(root),'--binary-dir',str(root/'bin'),'--partition','inner-0-0']
            with patch.object(sys,'argv',argv),patch.object(module,'dispatch',side_effect=dispatch) as owner:
                module.main()
            owner.assert_called_once()

    def fixture(self,root):
        (root/'PLAN.json').write_text('{}')
        parent=dict(id='parent',stage='inner',family='product_joint_identity',partition='inner-0-0',setting=0,seed=0)
        child={**parent,'id':'child','family':'product_pretrained_frozen'}
        run=root/'runs/parent/attempt-001';run.mkdir(parents=True)
        record=dict(cell=parent,plan_sha256=sha(root/'PLAN.json'),build_sha256='build',epochs=240,pairs_per_epoch=50000)
        (run/'RUN.json').write_text(json.dumps(record))
        result={**record,'stage':'interrupted','exit_code':None,'restart_approval_pending':True,'preserved_archive_receipt_sha256':'a'*64}
        (run/'RESULT.json').write_text(json.dumps(result))
        return parent,child,run,result

    def test_preserved_interruption_waits_without_using_partial_prefix(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);parent,child,run,result=self.fixture(root)
            with patch.object(module,'completed_parent') as prefix:
                ready,dependency=module.parent_state(root,child,[parent,child],'build')
            self.assertFalse(ready);prefix.assert_not_called()
            self.assertEqual(dependency['parent_cell'],'parent');self.assertEqual(len(dependency['evidence_sha256']),2)

    def test_failed_and_drifted_parents_remain_refusals(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);parent,child,run,result=self.fixture(root)
            for change in [dict(stage='failed',exit_code=1),dict(restart_approval_pending=False),dict(preserved_archive_receipt_sha256=''),dict(epochs=120),dict(build_sha256='other')]:
                (run/'RESULT.json').write_text(json.dumps({**result,**change}))
                with self.assertRaises(ValueError):module.parent_state(root,child,[parent,child],'build')

    def test_completed_parent_still_uses_existing_owner_gate(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);parent,child,run,result=self.fixture(root)
            (run/'RESULT.json').write_text(json.dumps({**result,'stage':'trained','exit_code':0}))
            with patch.object(module,'completed_parent',side_effect=ValueError('ambiguous')) as gate:
                with self.assertRaisesRegex(ValueError,'ambiguous'):module.parent_state(root,child,[parent,child],'build')
            gate.assert_called_once()

    def test_waiting_parent_does_not_block_independent_registered_child(self):
        class Observed(Exception):pass
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);(root/'PLAN.json').write_text('{}');binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}')
            cells=[dict(id=str(i),stage='inner',family='product_pretrained_frozen',partition='inner-0-0',setting=0,seed=i) for i in range(2)]
            launched=[];released=[];ticks=[0]
            class Process:
                pid=123;returncode=0
                def __init__(self,command,**kw):launched.append(command)
                def poll(self):return 0
            def sleep(_):
                ticks[0]+=1
                if ticks[0]==2:raise Observed()
            def readiness(_,cell,*rest):return (False,{'parent_cell':'blocked'}) if cell['id']=='0' else (True,None)
            args=SimpleNamespace(study=root,binary_dir=binary,partition='inner-0-0',workers=2,resume_name='test')
            with patch.object(module,'load_registration',return_value=({},cells,{})),patch.object(module,'existing_trial',return_value=None),patch.object(module,'parent_state',side_effect=readiness),patch.object(module,'reserve_worker',return_value={'can_start':True}),patch.object(module,'bind_worker'),patch.object(module,'release_worker',side_effect=lambda _,c:released.append(c)),patch.object(module,'verify_inner_fit'),patch.object(module.subprocess,'Popen',Process),patch.object(module.time,'sleep',side_effect=sleep):
                with self.assertRaises(Observed):module.dispatch(args)
            status=json.loads((root/'dispatch/pretrained/inner-0-0/test/STATUS.json').read_text())
            self.assertEqual(len(launched),1);self.assertEqual(launched[0][-3:],['1','--workers','1'])
            self.assertEqual(status['completed'],['1']);self.assertEqual(released,['1']);self.assertFalse(status['active'])
            self.assertEqual(set(status['interrupted_parent_dependencies']),{'0'})
            self.assertEqual(status['stage'],'supervising');self.assertIsNone(status['failure'])


if __name__=='__main__':unittest.main()
