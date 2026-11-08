import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_evaluation_inputs import preflight,verify,validate,PARTS
from shaping_completion import sha,encoded


class EvaluationInputTests(unittest.TestCase):
    def fixture(self,root):
        study=root/'study';study.mkdir();views=[];files={}
        for name in sorted(PARTS):
            view=study/'evaluation_views'/name;view.mkdir(parents=True);p=view/'RECEIPT.json';p.write_text('{}')
            files[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size);views.append(str(view))
        array=root/'y.npy';array.write_bytes(b'frozen-label-array')
        (study/'PLAN.json').write_bytes(encoded(dict(input_pins={str(array):sha(array)})))
        files[str(array)]=dict(sha256=sha(array),bytes=array.stat().st_size)
        return study,dict(schema='shaping-inner-evaluation-input-closure-v1',plan_sha256=sha(study/'PLAN.json'),files=files,views=views,outer_or_final_views_included=False)

    def test_incomplete_views_are_preserved_before_missing_inputs_are_copied(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study,record=self.fixture(root);view=Path(record['views'][0]);(view/'RECEIPT.json').unlink();(view/'partial.parquet').write_bytes(b'preserve partial')
            result=preflight(study,record,root/'preflight.json');backup=Path(result['incomplete_views_preserved'][0]['preserved'])
            self.assertEqual((backup/'partial.parquet').read_bytes(),b'preserve partial');self.assertFalse(view.exists())
            view.mkdir();(view/'RECEIPT.json').write_text('{}');self.assertEqual(verify(study,record)['files'],7)

    def test_complete_different_data_is_refused_without_overwriting(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study,record=self.fixture(root);p=Path(record['views'][0])/'RECEIPT.json';p.write_text('different preserved input')
            with self.assertRaisesRegex(ValueError,'preserve existing different'):preflight(study,record,root/'preflight.json')
            self.assertEqual(p.read_text(),'different preserved input')
            record['outer_or_final_views_included']=True
            with self.assertRaisesRegex(ValueError,'registration differs'):validate(study,record)

    def test_active_evaluation_prevents_cache_relocation(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);study,record=self.fixture(root);cmd=root/'cmdline';cmd.write_bytes(('python3\0shaping_evaluation_worker.py\0--work\0--study\0'+str(study)+'\0').encode())
            with patch('shaping_evaluation_inputs.Path.glob',return_value=[cmd]):
                with self.assertRaisesRegex(ValueError,'still active'):preflight(study,record,root/'preflight.json')


if __name__=='__main__':unittest.main()
