import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_paired_gradient_qualification as owner
from shaping_completion import encoded,sha


class GradientQualificationTests(unittest.TestCase):
    def inputs(self,root):
        study=root/'study';study.mkdir();(study/'PLAN.json').write_text('frozen study')
        binaries=root/'binaries';binaries.mkdir();tool=binaries/'zensim_mlp_train';tool.write_text('candidate')
        (binaries/'BUILD.json').write_bytes(encoded(dict(binary_sha256={tool.name:sha(tool)})))
        archive=root/'archive';checkpoints=archive/'attempt-001/checkpoints';checkpoints.mkdir(parents=True)
        pins={}
        for epoch in range(240):
            p=checkpoints/f'ckpt_epoch{epoch:03}.bin';p.write_text('original epoch '+str(epoch))
            pins[str(p.relative_to(archive))]=sha(p)
        cases={}
        for family,cid in owner.CASES.items():
            directory=study/'runs'/cid;directory.mkdir(parents=True);(directory/'SHA256SUMS').write_text('verified archive')
            cases[family]=dict(cell=dict(id=cid,family=family,partition='inner-0-0',stage='inner',setting=0,seed=0),
                run=str(directory/'attempt-001'),source_file_sha256={},
                original_final_sha256=pins['attempt-001/checkpoints/ckpt_epoch239.bin'])
        reg=root/'original.json';reg.write_bytes(encoded(dict(schema='corrected-runtime-control-compatibility-v1',
            plan_sha256=sha(study/'PLAN.json'),epochs=240,pairs_per_epoch=50000,
            scientific_selection=False,outer_results_read=False,cases=cases)))
        return study,reg,binaries,archive,pins

    def test_registration_copies_every_original_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);study,reg,binaries,archive,pins=self.inputs(root);out=root/'new'
            with patch('shaping_checkpoint_retention.archive_inventory',return_value=(archive,pins)):
                record=owner.register(study,reg,binaries,out)
            self.assertEqual(set(record['cases']),set(owner.CASES))
            for family in owner.CASES:
                self.assertEqual(len(list((out/'references'/family).glob('*.bin'))),240)
                self.assertEqual(sha(out/'references'/family/'ckpt_epoch239.bin'),pins['attempt-001/checkpoints/ckpt_epoch239.bin'])
            self.assertFalse(record['scientific_selection'])

    def test_partial_archive_refuses_before_output_creation(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);study,reg,binaries,archive,pins=self.inputs(root);out=root/'new'
            del pins['attempt-001/checkpoints/ckpt_epoch087.bin']
            with patch('shaping_checkpoint_retention.archive_inventory',return_value=(archive,pins)):
                with self.assertRaises(KeyError):owner.register(study,reg,binaries,out)
            self.assertFalse(out.exists())

    def test_wrong_prespecified_cell_is_not_substituted(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);study,reg,binaries,archive,pins=self.inputs(root);out=root/'new'
            d=json.loads(reg.read_text());d['cases']['product_joint_identity']['cell']['seed']=2;reg.write_bytes(encoded(d))
            with self.assertRaisesRegex(ValueError,'prespecified joint-shaping case differs'):
                owner.register(study,reg,binaries,out)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
