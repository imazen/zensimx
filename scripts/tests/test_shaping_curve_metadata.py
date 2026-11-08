import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_curve_metadata import extract


class NativeMetadataTests(unittest.TestCase):
    def test_native_utf8_metadata_is_exact_and_ambiguous_or_truncated_values_refuse(self):
        config=dict(schema='zensim-input-shaping-v1',terms=[dict(slopes=[0.,1.00000001,2.],knots=[.1,.9],gate=[0.,.5,1.])])
        text=json.dumps(config);row=dict(key='zentrain.input_shaping',kind='utf8',value_text=text,value_len=len(text.encode()))
        self.assertEqual(extract(dict(metadata=[row])),config)
        with self.assertRaisesRegex(ValueError,'one native'):extract(dict(metadata=[row,row]))
        with self.assertRaisesRegex(ValueError,'length'):extract(dict(metadata=[dict(row,value_len=row['value_len']-1)]))
        with self.assertRaisesRegex(ValueError,'encoding'):extract(dict(metadata=[dict(row,kind='bytes')]))


if __name__=='__main__':unittest.main()
