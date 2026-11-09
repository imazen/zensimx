import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_extract_bench as owner
from shaping_completion import encoded,sha


class ExtractionBenchTests(unittest.TestCase):
    def prepare(self, root):
        study=root/'study';study.mkdir();(study/'PLAN.json').write_text('frozen plan')
        fixtures=root/'fixtures';fixtures.mkdir();pins={}
        for name in owner.CASES:
            size,threads=map(int,name.split('-'));p=fixtures/(name+'.json')
            p.write_bytes(encoded(dict(schema='shaping-extract-cost-v1',raw_ids=[1,3],
                admission_sha256=sha(study/'PLAN.json'),size=size,threads=threads)))
            pins[p.name]=sha(p)
        (fixtures/'REGISTRATION.json').write_bytes(encoded(dict(file_sha256=pins)))
        binary=root/'shaping_extract_cost';binary.write_text('pinned executable')
        (root/'BUILD.json').write_bytes(encoded(dict(binary_sha256=sha(binary))))
        launcher=root/'profile_test_cpuset.sh';launcher.write_text('verified boundary launcher')
        tool=root/'zenbench';tool.write_text('CPU boundary verifier')
        return study,fixtures,binary,launcher,tool

    def test_changed_transport_input_is_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);args=self.prepare(root);out=root/'timing'
            with patch.dict(sys.modules,tree_study=SimpleNamespace(RAW_IDS=[1,3])):
                owner.register(*args,out)
            owner.verify(out,args[0])
            (out/'inputs/zenbench').write_text('changed verifier')
            with self.assertRaisesRegex(ValueError,'registered extraction input differs'):
                owner.verify(out,args[0])

    def test_partial_or_wrong_read_set_is_refused_before_output(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);args=self.prepare(root);out=root/'timing'
            with patch.dict(sys.modules,tree_study=SimpleNamespace(RAW_IDS=[1,3,7])):
                with self.assertRaisesRegex(ValueError,'frozen extraction workload differs'):
                    owner.register(*args,out)
            self.assertFalse(out.exists())

    def test_changed_binary_is_refused_before_output(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);args=self.prepare(root);out=root/'timing';args[2].write_text('different executable')
            with patch.dict(sys.modules,tree_study=SimpleNamespace(RAW_IDS=[1,3])):
                with self.assertRaisesRegex(ValueError,'qualified extraction binary differs'):
                    owner.register(*args,out)
            self.assertFalse(out.exists())

    def test_explicit_subset_preserves_unstarted_case_identities(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch);args=self.prepare(root);out=root/'timing'
            remaining=list(owner.CASES[1:])
            with patch.dict(sys.modules,tree_study=SimpleNamespace(RAW_IDS=[1,3])):
                record=owner.register(*args,out,remaining)
                with self.assertRaisesRegex(ValueError,'distinct registered'):
                    owner.register(*args,root/'duplicate',['256-4','256-4'])
            self.assertEqual(record['cases'],remaining)
            self.assertEqual(owner.verify(out,args[0])['cases'],remaining)


if __name__=='__main__':unittest.main()
