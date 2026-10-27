"""Invented uncertainty sources; forbidden rows contain nonnumeric sentinels."""
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

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

    def test_mat_selected_columns_match_scipy_without_decoding_excluded_responses(self):
        import numpy as np
        from scipy.io import savemat, loadmat
        import live_md_uncertainty as live
        values=np.array([[1.,float("nan"),3.],[4.,float("nan"),6.]],dtype=np.float64)
        for compressed in (False,True):
            stream=io.BytesIO()
            savemat(stream,{"ignored":np.array([float("nan")]),"dist_diffscores":values},do_compression=compressed)
            raw=stream.getvalue()
            oracle=loadmat(io.BytesIO(raw))["dist_diffscores"][:,[2,0]]
            original=np.frombuffer
            converted=[]
            def record(buf,*args,**kwargs):
                converted.append(len(buf));return original(buf,*args,**kwargs)
            with mock.patch.object(live.np,"frombuffer",record):
                actual=live.selected_columns(raw,"dist_diffscores",[2,0],(2,3))
            np.testing.assert_array_equal(actual,oracle)
            self.assertEqual(converted,[16,16])
            with self.assertRaisesRegex(ValueError,"nonfinite"):
                live.selected_columns(raw,"dist_diffscores",[1],(2,3))
            with self.assertRaises(ValueError): live.selected_columns(raw,"dist_diffscores",[0],(3,2))
            with self.assertRaises(ValueError): live.selected_columns(raw[:-1],"dist_diffscores",[0],(2,3))

    def test_live_observer_fit_contains_only_selected_items_and_preserves_part_observers(self):
        import numpy as np
        from scipy.io import savemat
        import confidence_audit as audit
        population=[];pins={}
        for part,nobs in (("Part 1",19),("Part 2",18)):
            directory=self.root/"live-md/labels"/part;directory.mkdir(parents=True)
            names=np.empty((225,1),dtype=object)
            for i in range(225):names[i,0]=f"item{i}.bmp"
            data=np.full((nobs,225),np.nan)
            for j in range(5):
                data[:,j]=j+np.arange(nobs)*0.01
                item=f"images/{part}/item{j}.bmp"
                population.append(dict(native=dict(item=item)))
            for filename,values in (("Imagelists.mat",dict(distimgs=names)),("Scores.mat",dict(dist_diffscores=data))):
                p=directory/filename;savemat(p,values,do_compression=True);pins[str(p)]=source.digest(p)
        with mock.patch.object(audit,"LIBRARY",self.root):
            result=audit.live_md(population,pins,self.ledger)
        self.assertEqual(len(result),10)
        self.assertEqual({r["n"] for r in result.values()},{18,19})
        self.assertTrue(all(r["se_native"]>0 for r in result.values()))

    def test_native_double_class_can_store_exact_int16_values(self):
        import struct
        import numpy as np
        from scipy.io import savemat, loadmat
        from live_md_uncertainty import selected_columns
        stream=io.BytesIO();values=np.array([[-32768,0,32767],[17,-12,300]],dtype=np.int16)
        savemat(stream,{"dist_diffscores":values},do_compression=False)
        raw=bytearray(stream.getvalue())
        # MAT's numeric storage tag is independent of the declared MATLAB class.
        struct.pack_into("<I",raw,144,6)
        oracle=loadmat(io.BytesIO(raw))["dist_diffscores"][:,[2,0]]
        actual=selected_columns(raw,"dist_diffscores",[2,0],(2,3))
        self.assertEqual(actual.dtype,np.float64)
        np.testing.assert_array_equal(actual,oracle)


if __name__=="__main__":unittest.main()
