"""Incremental dispatch must preserve cached work and refuse mismatched identities."""
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_scatter_prepare import missing_native_jobs


class NativeReuseTests(unittest.TestCase):
    def fixture(self,root):
        job=dict(cell=dict(id='cell',stage='inner'),report_sha256='a'*64,sha256='points',path='points.gz')
        manifest=dict(schema='zensim-native-dashboard-jobs-v1',outer_results_read=False,panel_sha256='runtime',jobs=[job])
        panel=dict(schema='zensim-native-detailed-panel-v1',cell=job['cell'],report_sha256='a'*64,
                   input_sha256='points',panel_sha256='runtime',outer_results_read=False,statistics={})
        path=root/('a'*64+'.json');path.write_text(json.dumps(panel));return manifest,panel,path

    def test_only_missing_jobs_are_dispatched_without_changing_original_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);manifest,panel,path=self.fixture(root);before=path.read_bytes()
            manifest['jobs'].append(dict(manifest['jobs'][0],report_sha256='b'*64))
            delta,reuse=missing_native_jobs(manifest,root)
            self.assertEqual(delta['jobs'],manifest['jobs'][1:]);self.assertEqual(len(manifest['jobs']),2)
            self.assertEqual(reuse['required_panels'],2);self.assertEqual(len(reuse['reused']),1)
            self.assertEqual(path.read_bytes(),before)

    def test_every_identity_field_must_match_before_reuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);manifest,panel,path=self.fixture(root)
            for key in ('schema','cell','report_sha256','input_sha256','panel_sha256','outer_results_read'):
                with self.subTest(key=key):
                    changed=dict(panel,**{key:'different'});path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError,'changed cached'):missing_native_jobs(manifest,root)
                    self.assertEqual(json.loads(path.read_text()),changed)

    def test_duplicate_and_pathlike_report_names_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);manifest,_,_=self.fixture(root)
            manifest['jobs']*=2
            with self.assertRaisesRegex(ValueError,'duplicate'):missing_native_jobs(manifest,root)
            manifest['jobs']=[dict(manifest['jobs'][0],report_sha256='../other')]
            with self.assertRaisesRegex(ValueError,'invalid'):missing_native_jobs(manifest,root)

    def test_symlink_cache_and_outer_scope_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);manifest,_,path=self.fixture(root);other=root/'other';path.rename(other);path.symlink_to(other)
            with self.assertRaisesRegex(ValueError,'regular cached'):missing_native_jobs(manifest,root)
            manifest['outer_results_read']=True
            with self.assertRaisesRegex(ValueError,'inner native'):missing_native_jobs(manifest,root)
            manifest['outer_results_read']=False;manifest['jobs'][0]['cell']['stage']='outer'
            with self.assertRaisesRegex(ValueError,'inner native cell'):missing_native_jobs(manifest,root)


if __name__=='__main__':unittest.main()
