import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_state_probe import numeric_arrays,read_state


class StateProbeTests(unittest.TestCase):
    def test_subnormal_boundary_and_zero_are_distinct(self):
        rows=numeric_arrays(dict(moment=[0.,-5e-324,sys.float_info.min,1.]))
        self.assertEqual((rows[0]['length'],rows[0]['zero'],rows[0]['subnormal']),(4,1,1))
        self.assertEqual(rows[0]['min_nonzero'],5e-324)

    def test_native_header_digest_and_identity_are_checked(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            root=Path(tmp)/('a'*64);root.mkdir();p=root/'state-0.snap'
            body=json.dumps(dict(identity=root.name,epoch=3,pair=9)).encode()
            raw=(7).to_bytes(8,'little')+hashlib.sha256(body).digest()+body;p.write_bytes(raw)
            self.assertEqual(read_state(p)[:2],(7,dict(identity=root.name,epoch=3,pair=9)))
            p.write_bytes(raw[:-1]+b' ')
            with self.assertRaisesRegex(ValueError,'checksum'):read_state(p)


if __name__=='__main__':unittest.main()
