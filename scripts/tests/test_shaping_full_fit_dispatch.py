import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,NEURAL,encoded,sha
from shaping_full_fit_execution import search_jobs,watch_search

class FullFitDispatchTests(unittest.TestCase):
    def fixture(self,root):
        cells=matrix();admission=dict(selection_cells=[c for c in cells if c['stage']=='full_fit_selection'],
            finalist_roles=dict(relu='relu',extra_trees='extra_trees',shaped_additive='shaped_additive',
                best_analytic='softplus',best_product='product_joint_identity',best_separate='separate_joint_identity'))
        binary=root/'bin';binary.mkdir();(binary/'BUILD.json').write_text('{}');(root/'PLAN.json').write_text('{}')
        p=root/'full_fit/PLAN.json';p.parent.mkdir();p.write_bytes(encoded(admission))
        args=SimpleNamespace(study=root,binary_dir=binary,neutral_qualification=root/'neutral')
        return cells,admission,p,args

    def test_all_search_cells_group_controls_without_repeated_linear_cache(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            cells,a,_,_=self.fixture(Path(directory));jobs=search_jobs(cells,a)
            self.assertEqual(sum(map(len,jobs)),192);self.assertEqual(len(jobs),150)
            self.assertEqual(sorted(len(j) for j in jobs if len(j)>1),[4,4,4,12,12,12])
            a['selection_cells']=a['selection_cells'][:-1]
            with self.assertRaisesRegex(ValueError,'complete registered'):search_jobs(cells,a)

    def test_missing_outer_admission_opens_no_fit_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells,a,p,args=self.fixture(root)
            with patch('shaping_full_fit_execution.time.sleep',side_effect=RuntimeError('observation ended')),\
                 patch('shaping_full_fit_execution.resolve_admission') as resolve,\
                 patch('shaping_full_fit_execution.prepare') as prepare,\
                 patch('shaping_full_fit_execution.subprocess.Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'observation ended'):watch_search(args)
                resolve.assert_not_called();prepare.assert_not_called();launch.assert_not_called()

    def test_complete_queue_has_all192_cells_and_two_slots(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells,a,path,args=self.fixture(root)
            for outer in range(3):
                p=root/'selection'/f'outer-{outer}'/'OUTER_ADMISSION.json';p.parent.mkdir(parents=True);p.write_text('{}')
                p=root/'outer_inputs'/f'outer-{outer}'/'PREPARED.json';p.parent.mkdir(parents=True);p.write_text('{}')
            jobs=search_jobs(cells,a);groups={j[0]['id']:j for j in jobs};started=[];live=0;peak=0
            class Process:
                def __init__(child,cmd,**kwargs):
                    nonlocal live,peak
                    if '--cell' in cmd:group=groups[cmd[cmd.index('--cell')+1]]
                    else:
                        family='extra_trees' if '--tree' in cmd else 'shaped_additive';partition=cmd[cmd.index('--partition')+1]
                        group=next(j for j in jobs if j[0]['family']==family and j[0]['partition']==partition)
                        self.assertIn('--full-fit-selection',cmd)
                    child.group=group;started.extend(c['id'] for c in group);child.pid=1000+len(started);child.returncode=None
                    live+=1;peak=max(peak,live)
                def poll(child):
                    nonlocal live
                    if child.returncode is not None:return child.returncode
                    for c in child.group:
                        r=root/'runs'/c['id']/'attempt-001';r.mkdir(parents=True)
                        identity=dict(cell=c,admission_sha256=sha(path),plan_sha256=sha(root/'PLAN.json'),
                            build_sha256=sha(args.binary_dir/'BUILD.json'),selection_scope='full_fit_only',development_results_read=False)
                        (r/'RUN.json').write_bytes(encoded(identity));(r/'RESULT.json').write_bytes(encoded(dict(identity,
                            stage='trained' if a['finalist_roles'][c['family']] in NEURAL else 'complete',exit_code=0)))
                    live-=1;child.returncode=0;return 0
            with patch('shaping_full_fit_execution.resolve_admission',return_value=(a,path)),\
                 patch('shaping_full_fit_execution.load_registration',return_value=({},cells,{})),\
                 patch('shaping_full_fit_execution.prepare') as prepare,\
                 patch('shaping_full_fit_execution.subprocess.Popen',side_effect=Process),patch('shaping_full_fit_execution.time.sleep'):
                watch_search(args);self.assertEqual(prepare.call_count,3)
            self.assertEqual(peak,2);self.assertEqual(live,0);self.assertEqual(len(started),192);self.assertEqual(len(set(started)),192)
            status=json.loads((root/'dispatch/full-fit-search/STATUS.json').read_text())
            self.assertEqual(status['stage'],'search_fits_finished_reports_required')

    def test_failed_child_stops_new_launches_and_drains_admitted_children(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells,a,path,args=self.fixture(root)
            for outer in range(3):
                p=root/'selection'/f'outer-{outer}'/'OUTER_ADMISSION.json';p.parent.mkdir(parents=True);p.write_text('{}')
                p=root/'outer_inputs'/f'outer-{outer}'/'PREPARED.json';p.parent.mkdir(parents=True);p.write_text('{}')
            children=[]
            class Process:
                def __init__(child,*_args,**_kwargs):
                    child.pid=1000+len(children);child.returncode=None;child.polls=0;children.append(child)
                def poll(child):
                    child.polls+=1
                    if child is children[0] or child.polls>=2:child.returncode=1
                    return child.returncode
            with patch('shaping_full_fit_execution.resolve_admission',return_value=(a,path)),\
                 patch('shaping_full_fit_execution.load_registration',return_value=({},cells,{})),\
                 patch('shaping_full_fit_execution.prepare'),\
                 patch('shaping_full_fit_execution.subprocess.Popen',side_effect=Process),patch('shaping_full_fit_execution.time.sleep'):
                with self.assertRaisesRegex(RuntimeError,'child failed'):watch_search(args)
            self.assertEqual(len(children),2);self.assertTrue(all(c.returncode==1 for c in children))
            status=json.loads((root/'dispatch/full-fit-search/STATUS.json').read_text())
            self.assertEqual(status['stage'],'failed');self.assertEqual(status['active'],{})

    def test_shared_outer_preparation_must_finish_before_search_preparation(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);cells,a,path,args=self.fixture(root)
            for outer in range(3):
                p=root/'selection'/f'outer-{outer}'/'OUTER_ADMISSION.json';p.parent.mkdir(parents=True);p.write_text('{}')
            with patch('shaping_full_fit_execution.resolve_admission',return_value=(a,path)),\
                 patch('shaping_full_fit_execution.load_registration',return_value=({},cells,{})),\
                 patch('shaping_full_fit_execution.prepare') as prepare,\
                 patch('shaping_full_fit_execution.subprocess.Popen') as launch,\
                 patch('shaping_full_fit_execution.time.sleep',side_effect=RuntimeError('waiting for shared inputs')):
                with self.assertRaisesRegex(RuntimeError,'shared inputs'):watch_search(args)
                prepare.assert_not_called();launch.assert_not_called()
