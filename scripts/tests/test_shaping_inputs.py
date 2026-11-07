import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile,json
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_inputs as shaping


class ShapeInitializationTests(unittest.TestCase):
    def test_cached_shapes_bind_raw_inputs_and_declaration(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);raw=root/'data/raw.npy';raw.parent.mkdir()
            np.save(raw,np.array([[.2,.8],[.8,.2]],np.float32))
            declaration=root/'scripts/rev4_featpot/color_noise_fx2_declaration.json'
            declaration.parent.mkdir(parents=True)
            declaration.write_text(json.dumps(dict(direct=[13],products=[[13,422]])))
            (root/'PLAN.json').write_bytes(b'{}')
            plan=dict(input_pins={str(raw):shaping.sha(raw)})
            with patch('shaped_linear_study.OLD',root),patch('tree_study.RAW_IDS',[13,422]), \
                 patch('shaping_completion.REPO',root),patch('shaping_completion.DECLARATION_SHA256',shaping.sha(declaration)):
                first=shaping.prepare(root,'variant-test',dict(train=[0,1]),plan,False)
                self.assertEqual(first,shaping.prepare(root,'variant-test',dict(train=[0,1]),plan,False))
                with self.assertRaisesRegex(ValueError,'scientific identity'):
                    shaping.prepare(root,'variant-test',dict(train=[0,1]),dict(input_pins={str(raw):'changed'}),False)
                declaration.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'declaration drift'):
                    shaping.prepare(root,'variant-test',dict(train=[0,1]),plan,False)

    def test_held_out_values_cannot_move_training_knots(self):
        raw=np.array([[.2,.8],[.8,.2],[100.,.9]],np.float32)
        declaration={'direct':[13],'products':[[13,422]]}
        before=shaping.initial_config(raw,[13,422],[0,1],declaration,720,False)
        raw[2]=[1e10,.001]
        self.assertEqual(before,shaping.initial_config(raw,[13,422],[0,1],declaration,720,False))
        self.assertNotEqual(before,shaping.initial_config(raw,[13,422],[0,1,2],declaration,720,False))

    def test_separate_and_product_have_distinct_knot_domains(self):
        raw=np.array([[.2,.8],[.8,.2]],np.float32)
        declaration={'direct':[13],'products':[[13,422]]}
        product=shaping.initial_config(raw,[13,422],[0,1],declaration,720,False)
        separate=shaping.initial_config(raw,[13,422],[0,1],declaration,720,True)
        self.assertEqual(separate['terms'][0]['knots'],separate['terms'][1]['knots'])
        self.assertNotEqual(product['terms'][1]['knots'],separate['terms'][1]['knots'])
        self.assertEqual(separate['terms'][1]['gate'],[0.,.5,1.])
        self.assertEqual(separate['terms'][1]['column'],720)

    def test_domains_and_degenerate_knots(self):
        self.assertEqual(shaping.knots([0,0]),[1.,2.])
        self.assertEqual(shaping.knots([1,1]),[1.,2.])
        for values in ([-.1,1],[float('nan'),1],[float('inf'),1]):
            with self.assertRaises(ValueError):shaping.knots(values)
        with self.assertRaises(ValueError):
            shaping.initial_config(np.array([[1,2]],np.float32),[13,422],[0],{'direct':[13],'products':[[13,422]]},720,True)


if __name__=='__main__':unittest.main()
