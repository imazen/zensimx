import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_external_publish import publish,sha


class ExternalPublicationTests(unittest.TestCase):
    def fixture(self,root,cid='fixed'):
        cell=dict(id=cid,family='relu',partition='inner-0-0',setting=0,seed=0)
        registry=dict(cells=[cell],datasets=dict(csiq=dict(rows=3)))
        (root/'REGISTRATION.json').write_text(json.dumps(registry))
        model=dict(cell=cell,bake_sha256='bake',report_sha256='report')
        (root/'BINDINGS-initial.json').write_text(json.dumps(dict(registration_sha256=sha(root/'REGISTRATION.json'),models=[model])))
        (root/'EXECUTION_SOURCE.json').write_text(json.dumps(dict(driver_sha256='executed')))
        result=root/'results'/cid/'csiq';result.mkdir(parents=True)
        pairs=result/'all-scatter.json';pairs.write_text(json.dumps(dict(indices=[0,1,2],predicted=[1.,2.,3.],target=[2.,1.,3.],source=['a','a','b'],distortion=['codec']*3)))
        (result/'native.tsv').write_text('preserved native predictions')
        value=dict(schema='shaping-fixed-external-result-v1',cell=cell,dataset='csiq',
            registration_sha256=sha(root/'REGISTRATION.json'),bake_sha256='bake',report_sha256='report',
            prediction_sha256=sha(result/'native.tsv'),driver_sha256='executed',selection_permitted=False,
            training_cells_added=0,populations=dict(all=dict(rows=3,sources=2,scatter=str(pairs),statistics=dict(n=3,n_dropped=0))))
        path=result/'RESULT.json';path.write_text(json.dumps(value));return path,value,pairs

    def test_publishes_scalar_links_and_pins_without_modifying_results(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);path,_,_=self.fixture(root);before=path.read_bytes()
            self.assertEqual(publish(root,[root]),dict(completed_models=1,dataset_reports=1,pending=0))
            value=json.loads((root/'INDEX.json').read_text());self.assertEqual(path.read_bytes(),before)
            self.assertEqual(value['results'][0]['source_result_sha256'],sha(path))
            self.assertEqual(value['results'][0]['populations']['all']['scatter_url'],'results/fixed/csiq/all-scatter.json')

    def test_duplicate_batch_and_registration_drift_cannot_replace_page(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);self.fixture(root);publish(root,[root]);before=(root/'INDEX.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'duplicate external dataset'):publish(root,[root,root])
            self.assertEqual((root/'INDEX.json').read_bytes(),before)

    def test_disjoint_batches_keep_the_same_registry_and_relative_scalar_links(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            base=Path(directory);first=base/'first';second=base/'second';first.mkdir();second.mkdir()
            entries=[(first,*self.fixture(first)),(second,*self.fixture(second,'second'))]
            registry=dict(cells=[value['cell'] for _,_,value,_ in entries],datasets=dict(csiq=dict(rows=3)))
            for root,path,value,_ in entries:
                (root/'REGISTRATION.json').write_text(json.dumps(registry));pin=sha(root/'REGISTRATION.json')
                bound=json.loads((root/'BINDINGS-initial.json').read_text());bound['registration_sha256']=pin
                (root/'BINDINGS-initial.json').write_text(json.dumps(bound));value['registration_sha256']=pin
                path.write_text(json.dumps(value))
            self.assertEqual(publish(first,[first,second])['completed_models'],2)
            index=json.loads((first/'INDEX.json').read_text())
            self.assertTrue(index['results'][1]['populations']['all']['scatter_url'].startswith('../second/'))
            before=(first/'INDEX.json').read_bytes();(second/'REGISTRATION.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'batch registration differs'):publish(first,[first,second])
            self.assertEqual((first/'INDEX.json').read_bytes(),before)

    def test_drifted_predictions_unbound_model_and_nonfinite_scatter_refuse(self):
        for change in ('predictions','model','scatter','driver','selection','legacy'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root=Path(directory);path,value,pairs=self.fixture(root)
                if change=='predictions':(path.parent/'native.tsv').write_text('changed')
                if change=='model':value['bake_sha256']='unbound'
                if change=='scatter':
                    pair=json.loads(pairs.read_text());pair['predicted'][0]=float('nan');pairs.write_text(json.dumps(pair))
                if change=='driver':value['driver_sha256']='unknown'
                if change=='selection':value['selection_permitted']=True
                if change=='legacy':del value['driver_sha256']
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):publish(root,[root])
                self.assertFalse((root/'INDEX.json').exists())


if __name__=='__main__':unittest.main()
