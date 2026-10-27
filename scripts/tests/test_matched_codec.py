"""Invented metadata only; no fitting/assessment response table is opened."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
import matched_codec as codec
from subjective_training_leg import sha


class MatchingAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.table = self.root/"human.parquet"
        self.table.write_bytes(b"forbidden response sentinel")
        Path(str(self.table)+".manifest.json").write_text(json.dumps(dict(table_sha256="a"*64,keys_sha256="b"*64)))
        self.path = self.root/"matching.json"
        self.d = dict(schema="mlic-matched-codecs-v1",role="TRAIN",candidate="MlicOrder",policy=codec.POLICY,
            fold="tid2013",seed_index=0,init_seed=1101,sample_seed=100000101,
            original_root_sha256=codec.ROOT_SHA,table_sha256="a"*64,keys_sha256="b"*64)
        self.args = SimpleNamespace(matched_codec_manifest=self.path,matched_codec_sha256=None,
            strict_admission=True,train_only=True,heldout="tid2013",seed_index=0,exclude_datasets="")
        self.groups = [("human",self.table,32.5,0,"withinref,rank"),
                       ("human_development",self.root/"dev.parquet",0,1,"withinref,rank")]

    def attach(self):
        self.path.write_text(json.dumps(self.d))
        self.args.matched_codec_sha256=sha(self.path)
        original=Path.read_bytes
        def read(path):
            if path == self.table: self.fail("response payload opened during matching metadata admission")
            return original(path)
        with mock.patch.object(codec,"sha",return_value=codec.ROOT_SHA), mock.patch.object(Path,"read_bytes",read):
            return codec.attach(self.args,self.groups,self.root,1101,100000101,{"hidden":128})

    def test_only_group_name_changes_mass_and_development_are_preserved(self):
        groups,recipe=self.attach()
        self.assertEqual(groups,[(codec.GROUP,*self.groups[0][1:]),self.groups[1]])
        self.assertEqual(recipe["hidden"],128)
        self.assertEqual(codec.command_flags(recipe,strict_admission=True,head="N"),
            ["--matched-codec-manifest",str(self.path),"--matched-codec-sha256",sha(self.path),"--no-sample-coverage"])

    def test_wrong_cell_seed_policy_root_and_keys_refuse_before_labels(self):
        for key,value in (("fold","kadid"),("seed_index",1),("sample_seed",101),("role","EVAL"),
                          ("policy","different"),("original_root_sha256","x"*64),("keys_sha256","c"*64)):
            old=self.d[key];self.d[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.attach()
            self.d[key]=old

    def test_disabled_recipe_is_identical_and_alternate_route_refuses(self):
        args=SimpleNamespace(matched_codec_manifest=None,matched_codec_sha256=None)
        recipe={"hidden":128}
        self.assertEqual(codec.attach(args,self.groups,self.root,0,0,recipe),(self.groups,recipe))
        self.assertEqual(codec.command_flags(recipe,strict_admission=False,head="P"),[])
        with self.assertRaises(ValueError):
            codec.command_flags({"matched_codec":{}},strict_admission=False,head="N")
        self.args.heldout=None
        with self.assertRaisesRegex(ValueError,"exact registered omission"):self.attach()

    def test_registry_binds_all_cells_and_kadid_omission_remains_inactive(self):
        cells={}
        for seed in range(3):
            for fold in codec.FOLDS:
                if fold=="kadid": cells[f"{fold}_s{seed}"]=dict(inactive="KADID omitted; unchanged base sampler")
                else:
                    p=self.root/f"{fold}_{seed}.json"
                    p.write_text(json.dumps(dict(fold=fold,seed_index=seed,policy=codec.POLICY)))
                    cells[f"{fold}_s{seed}"]=dict(path=str(p),sha256=sha(p))
        registry=self.root/"registry.json"
        registry.write_text(json.dumps(dict(schema="mlic-codec-cells-v1",original_root_sha256=codec.ROOT_SHA,cells=cells)))
        self.assertIsNone(codec.cell_binding(registry,sha(registry),"MlicOrder","kadid",0))
        self.assertEqual(codec.cell_binding(registry,sha(registry),"MlicOrder","tid2013",2),cells["tid2013_s2"])
        with self.assertRaises(ValueError):codec.cell_binding(registry,sha(registry),"Mlic","tid2013",2)
        cells.pop("kadid_s1")
        registry.write_text(json.dumps(dict(schema="mlic-codec-cells-v1",original_root_sha256=codec.ROOT_SHA,cells=cells)))
        with self.assertRaises(ValueError):codec.cell_binding(registry,sha(registry),"MlicOrder","tid2013",2)


if __name__=="__main__":unittest.main()
