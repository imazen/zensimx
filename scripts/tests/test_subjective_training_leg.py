import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
import subjective_training_leg as leg


class SubjectiveAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.path = root/"features.parquet"
        self.path.write_bytes(b"invented target-bearing sentinel, never parsed")
        self.keys = root/"keys.json"
        self.rows = [dict(observation=f"o{i}", source="invented-source", family="invented-family",
                          reference="invented-reference", role="fit", tier="T2", row=i) for i in range(2)]
        self.decl = dict(schema="subjective-training-leg-v1",loss="within-source-rank",nominal_weight=8,
            target_orientation="quality",target_transform="negate-pointwise",target_column="human_score",
            selection="fixed-final-epoch",formula_revision=5,tier="v3",feature_ids=[0,1],
            producer_commit="b"*40,table_sha256="f"*64, rows=2)
        self.record = dict(name="subjective_rank",path=str(self.path),train_weight=16.,val_weight=0.,
            mode="withinref,rank",subjective=True,table_sha256="f"*64,keys_path=str(self.keys),
            rows=2,source_counts={"invented-source":2})
        self.cell = dict(feature_ids=[0,1],formula_revision=5,tier="v3",producer_commit="b"*40,
            sources=["invented-source"],excluded_families=[],groups=[self.record],route="full",seed_index=0,
            epochs=120,pairs_per_epoch=50000,selected_epoch=119,init_seed=1101,sample_seed=101)
        self.authority = dict(schema="subjective-training-authority-v1",state="approved",
            artifact_visibility="private",selection="fixed-final-epoch",cells={"invented-cell":self.cell})
        self.authority_path = root/"authority.json"
        self.groups = [("subjective_rank",self.path,16.,0.,"withinref,rank")]

    def save(self):
        self.keys.write_text(json.dumps(self.rows,sort_keys=True))
        declaration = Path(str(self.path)+".manifest.json")
        declaration.write_text(json.dumps(self.decl,sort_keys=True))
        self.record["manifest_sha256"] = leg.sha(declaration)
        self.record["keys_sha256"] = leg.sha(self.keys)
        self.authority_path.write_text(json.dumps(self.authority,sort_keys=True))
        return leg.sha(self.authority_path)

    def admit(self):
        return leg.admit(self.authority_path,self.save(),"invented-cell",self.groups,[0,1])

    def test_r12_linear_is_an_explicit_architecture_authorization(self):
        self.authority['schema'] = 'subjective-training-authority-v2'
        self.cell.update(width=1853, feature_ids=list(range(492)))
        recipe = dict(hidden=1, mono_all=True, derived_inputs='fx2')
        def check():
            return leg.command_binding(self.authority_path, self.save(), 'invented-cell',
                                       1101, 101, 1853, 'N', recipe)
        with self.assertRaises(ValueError):
            check()  # historical H128 authority does not authorize H1
        self.cell['r12_linear'] = dict(registration=leg.R12_REGISTRATION, arm='L1', mask_sha256=None)
        check()
        for field, value in [('hidden', 8), ('thresholds', True), ('max_pairs', 1), ('derived_inputs', 'fx1')]:
            with self.subTest(field=field):
                old = recipe.get(field)
                recipe[field] = value
                with self.assertRaises(ValueError):
                    check()
                if old is None:
                    del recipe[field]
                else:
                    recipe[field] = old
        self.cell['r12_linear']['arm'] = 'L2'
        with self.assertRaises(ValueError):
            check()  # sparse mask cannot be omitted
        self.cell['r12_linear']['mask_sha256'] = 'c' * 64
        check()

    def test_r12_l4_requires_one_pair_and_one_nonpositive_offset(self):
        c=dict(width=1853,feature_ids=list(range(492)),r12_linear=dict(
            registration="339de6b8c24b49403745cacde495ded875a3e83a",arm="L4",mask_sha256=None))
        recipe=dict(hidden=2,mono_all=True,derived_inputs='fx2',max_pairs=1,thresholds=True)
        self.assertTrue(leg.r12_head_binding(c,recipe))
        for key,value in [('thresholds',False),('max_pairs',2),('hidden',128)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                leg.r12_head_binding(c,dict(recipe,**{key:value}))

    def test_r13_width_authorization_keeps_legacy_pin_and_pair_proportion(self):
        self.authority['schema'] = 'subjective-training-authority-v2'
        self.cell.update(width=1853, feature_ids=list(range(492)))
        for width in (4, 16, 32, 64, 256, 512):
            recipe = dict(hidden=width, mono_all=True, thresholds=True,
                          derived_inputs='fx2', max_pairs=width // 4)
            self.cell.pop('r13_width', None)
            with self.assertRaises(ValueError):
                leg.command_binding(self.authority_path, self.save(), 'invented-cell',
                                    1101, 101, 1853, 'N', recipe)
            self.cell['r13_width'] = dict(registration=leg.R13_W4_REGISTRATION if width==4 else leg.R13_REGISTRATION,
                                         hidden=width, max_pairs=width // 4)
            leg.command_binding(self.authority_path, self.save(), 'invented-cell',
                                1101, 101, 1853, 'N', recipe)
            if width==4:
                self.cell['r13_width']['registration']=leg.R13_REGISTRATION
                with self.assertRaises(ValueError):leg.r13_head_binding(self.cell,recipe)
                self.cell['r13_width']['registration']=leg.R13_W4_REGISTRATION
                self.cell['seed_index']=1
                with self.assertRaises(ValueError):leg.r13_head_binding(self.cell,recipe)
                self.cell['seed_index']=0
            for key, value in [('hidden', 128), ('max_pairs', width // 4 + 1),
                               ('thresholds', False), ('derived_inputs', 'fx1')]:
                with self.subTest(width=width, key=key), self.assertRaises(ValueError):
                    leg.r13_head_binding(self.cell, dict(recipe, **{key: value}))
            self.cell['r13_width']['registration'] = 'unregistered'
            with self.assertRaises(ValueError):
                leg.r13_head_binding(self.cell, recipe)

    def test_valid_metadata_never_opens_target(self):
        with mock.patch.object(Path,"read_bytes",autospec=True,side_effect=self.read_guard):
            self.assertEqual(set(self.admit()),{"subjective_rank"})

    def read_guard(self,path):
        if path == self.path:
            self.fail("target-bearing payload opened during metadata admission")
        with path.open("rb") as stream:
            return stream.read()

    def test_role_mode_weight_producer_and_ids_refuse(self):
        for field,value in (("nominal_weight",4),("loss","pooled-rank"),("formula_revision",4),
                            ("tier","v2"),("feature_ids",[1,0]),("producer_commit","c"*40)):
            with self.subTest(field=field):
                old=self.decl[field]
                self.decl[field]=value
                with self.assertRaises(ValueError): self.admit()
                self.decl[field]=old
        self.groups[0] = ("subjective_rank",self.path,16.,1.,"withinref,rank")
        with self.assertRaises(ValueError): self.admit()

    def test_key_order_role_source_and_leak_refuse(self):
        for field,value in (("row",9),("role","dev"),("source","late-source"),("observation","o0")):
            with self.subTest(field=field):
                old=self.rows[1][field]
                self.rows[1][field]=value
                with self.assertRaises(ValueError): self.admit()
                self.rows[1][field]=old
        self.cell["excluded_families"]=["invented-family"]
        with self.assertRaises(ValueError): self.admit()

    def test_authority_is_external_and_all_groups_are_bound(self):
        pin=self.save()
        self.authority_path.write_text("{}")
        with self.assertRaises(ValueError): leg.admit(self.authority_path,pin,"invented-cell",self.groups,[0,1])
        self.groups.append(("late",self.path,1.,0.,"rank"))
        with self.assertRaises(ValueError): self.admit()
        with self.assertRaises(ValueError): leg.safe(self.path.parent/"_sealed"/"arbitrary")

    def test_actual_wrapper_binds_native_flags_and_output_ancestry(self):
        import v2_lodo_mlp as wrapper
        self.decl["table_sha256"] = self.record["table_sha256"] = leg.sha(self.path)
        self.decl["immutable_input_roots"] = [str(self.path.parent)]
        pin=self.save()
        output=self.path.parent.parent/(self.path.parent.name+"-output")
        argv=wrapper.train_command(self.groups,1101,101,1825,output/"keep.txt","N",output/"last.bin",recipe={"hidden":128},
            strict_admission=True,subjective_authority=(self.authority_path,pin,"invented-cell",[0,1]))
        self.assertIn("--subjective-authority",argv)
        self.assertEqual(argv[argv.index("--subjective-authority-sha256")+1],pin)
        self.assertIn("subjective_rank:"+str(self.path)+":16.0:0.0:withinref,rank",argv)
        with self.assertRaises(ValueError):
            wrapper.train_command(self.groups,1101,101,1825,self.path.parent/"keep.txt","N",self.path.parent/"out.bin",recipe={"hidden":128},
                strict_admission=True,subjective_authority=(self.authority_path,pin,"invented-cell",[0,1]))

    def test_wrong_seed_refuses_before_target_bearing_hash(self):
        import v2_lodo_mlp as wrapper
        pin=self.save()
        with mock.patch.object(Path,"read_bytes",autospec=True,side_effect=self.read_guard):
            with self.assertRaisesRegex(ValueError,"cell/seed"):
                wrapper.train_command(self.groups,1103,101,1825,Path("unused-keep"),"N",Path("unused-out"),
                    recipe={"hidden":128},strict_admission=True,
                    subjective_authority=(self.authority_path,pin,"invented-cell",[0,1]))


if __name__ == "__main__": unittest.main()
