import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_stable_export_prepare import registered_extensions


class ExtensionRegistrationTests(unittest.TestCase):
    def fixture(self,root):
        values={}
        for outer in range(3):
            p=root/'extensions'/f'outer-{outer}'/'REGISTRATION.json';p.parent.mkdir(parents=True)
            cells=[dict(id=f'{outer}-{i}',stage='exposure_extension') for i in range(20)]
            p.write_text(json.dumps(dict(cells=cells)));values[p]=dict(cells=cells)
        return values

    def test_all_sixty_cells_require_three_complete_fold_validations(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);values=self.fixture(root);primary=(dict(id='primary'),)
            with patch('shaping_extension.registered_fold',side_effect=lambda study,path,**kw:values[path]) as full:
                cells=registered_extensions(root,primary)
            self.assertEqual(len(cells),60);self.assertEqual(full.call_count,3)
            self.assertEqual({c['id'] for c in cells},{c['id'] for row in values.values() for c in row['cells']})
            self.assertTrue(all(call.kwargs['cells'] is primary for call in full.call_args_list))

    def test_registration_failure_and_duplicate_ids_still_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);values=self.fixture(root)
            with patch('shaping_extension.registered_fold',side_effect=ValueError('extension selection drift')):
                with self.assertRaisesRegex(ValueError,'selection drift'):registered_extensions(root,())
            first=next(iter(values.values()))['cells'][0]
            with patch('shaping_extension.registered_fold',side_effect=lambda study,path,**kw:values[path]):
                with self.assertRaisesRegex(ValueError,'one registered'):registered_extensions(root,(first,))


if __name__=='__main__':unittest.main()
