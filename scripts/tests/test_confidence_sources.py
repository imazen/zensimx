"""Invented uncertainty sources; forbidden rows contain nonnumeric sentinels."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"byconstruction"))
import confidence_sources as source


class NativeUncertainty(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ledger=self.root/"reads.json"
        self.ledger.write_text(json.dumps(dict(role="TRAIN",fitting_manifest_sha256="a"*64,reads=[])))

    def put(self,name,text):
        path=self.root/name;path.write_text(text)
        return path,source.digest(path)

    def test_kadid_variance_uses_30_and_never_parses_excluded_values(self):
        path,pin=self.put("dmos.csv","dist_img,ref_img,dmos,var\nI01.png,R01.png,unread,1.2\nI07.png,R07.png,forbidden,forbidden\n")
        result=source.kadid(path,pin,{"i01.png"},self.ledger)
        self.assertEqual(result["i01.png"]["se_native"],0.2)
        self.assertEqual(result["i01.png"]["n"],30)
        with self.assertRaises(ValueError): source.kadid(path,pin,{"i07.png"},self.ledger)
        with self.assertRaisesRegex(ValueError,"incomplete"): source.kadid(path,pin,{"missing"},self.ledger)
        with self.assertRaisesRegex(ValueError,"source changed"): source.kadid(path,"b"*64,{"i01.png"},self.ledger)

    def test_tid_already_mean_uncertainty_and_line_binding(self):
        names,n_pin=self.put("names.txt","unread I01.bmp\nforbidden I02.bmp\nunread I03.bmp\n")
        std,s_pin=self.put("std.txt","0.13\nforbidden\n0.2\n")
        result=source.tid(names,n_pin,std,s_pin,{"i01.bmp","i03.bmp"},self.ledger)
        self.assertEqual([result[k]["se_native"] for k in sorted(result)],[0.13,0.2])
        self.assertIsNone(result["i01.bmp"]["n"])
        with self.assertRaises(ValueError): source.tid(names,n_pin,std,s_pin,{"i02.bmp"},self.ledger)
        std.write_text("0.13\nforbidden\n")
        with self.assertRaisesRegex(ValueError,"row count differs"):
            source.tid(names,n_pin,std,source.digest(std),{"i01.bmp"},self.ledger)

    def test_published_sd_count_selects_before_response_parsing(self):
        path,pin=self.put("labels.tsv","item\tstd\tn_ratings\tscore\na\t1.2\t36\tunread\nforbidden\n")
        result=source.published_spread(path,pin,{0:"a"},self.ledger)
        self.assertAlmostEqual(result["a"]["se_native"],0.2)
        with self.assertRaisesRegex(ValueError,"item binding"):
            source.published_spread(path,pin,{0:"wrong"},self.ledger)
        for sd,n in (("nan","36"),("-1","36"),("1",""),("1","1.5"),("1","1")):
            path.write_text(f"item\tstd\tn_ratings\tscore\na\t{sd}\t{n}\tunread\n")
            with self.subTest(sd=sd,n=n),self.assertRaises(ValueError):
                source.published_spread(path,source.digest(path),{0:"a"},self.ledger)

    def test_source_access_requires_train_population_authority(self):
        path,pin=self.put("dmos.csv","dist_img,var\na,1\n")
        self.ledger.write_text(json.dumps(dict(role="EVAL",fitting_manifest_sha256="a"*64,reads=[])))
        with self.assertRaisesRegex(ValueError,"population authority"):
            source.kadid(path,pin,{"a"},self.ledger)

    def test_population_join_uses_frozen_keys_and_exact_native_reference(self):
        import confidence_population as population
        p,_=self.put("ids.tsv","item\tref\tscore\nd/I02_01_1.bmp\tr/I02.bmp\tforbidden\n")
        native={"kadid-10k":{},"tid2013":population.identities(p)}
        keys=[dict(pair_key="a"*64,member_set="tid2013",ref_basename="tid:I02")]
        bank=dict(ref_group="tid:I02",dist_path="d/I02_01_1.png",ref_path="r/I02.png",
                  ref_pixels_sha256="b"*64,dist_pixels_sha256="c"*64)
        rows=population.bind_human(keys,{"tid2013":{"a"*64:bank}},native)
        self.assertEqual(rows[0]["native"]["row"],0)
        self.assertEqual(rows[0]["table_row"],0)
        bank["ref_group"]="tid:I03"
        with self.assertRaisesRegex(ValueError,"reference identity"):
            population.bind_human(keys,{"tid2013":{"a"*64:bank}},native)


if __name__=="__main__":unittest.main()
