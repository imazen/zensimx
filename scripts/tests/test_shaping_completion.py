"""Registration must reject leakage and retain paired controls/conditional work."""
import importlib.util
from pathlib import Path
import unittest
import json,sys,tempfile

path = Path(__file__).resolve().parents[1]/'rev4_featpot/shaping_completion.py'
spec = importlib.util.spec_from_file_location('shaping_completion', path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class RegistrationTests(unittest.TestCase):
    def test_worker_assignment_prevents_duplicate_host_execution(self):
        sys.path.insert(0,str(path.parent))
        from shaping_host_assignment import check
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);(root/'PLAN.json').write_text('{}')
            d=root/'worker_assignments';d.mkdir()
            row=dict(host='worker-a',cells=['assigned'],plan_sha256=study.sha(root/'PLAN.json'),build_sha256='build')
            (d/'wave.json').write_text(json.dumps(row))
            check(root,[dict(id='assigned')],'build','worker-a')
            check(root,[dict(id='unassigned')],'build','worker-b')
            with self.assertRaises(ValueError):check(root,[dict(id='assigned')],'build','worker-b')
            with self.assertRaises(ValueError):check(root,[dict(id='assigned')],'different','worker-a')

    def test_portable_protocol_requires_identical_registered_bytes(self):
        sys.path.insert(0,str(path.parent))
        from shaping_execution import load_registration
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            (root/'MATRIX.json').write_text('[]');(root/'PARTITIONS.json').write_text('{}')
            protocol=root/'PROTOCOL.md';protocol.write_text('frozen scientific protocol\n')
            plan=dict(protocol=str(root/'unavailable-host-path.md'),protocol_sha256=study.sha(protocol),
                      matrix_sha256=study.sha(root/'MATRIX.json'),partition_sha256=study.sha(root/'PARTITIONS.json'))
            (root/'PLAN.json').write_text(json.dumps(plan))
            self.assertEqual(load_registration(root),(plan,[],{}))
            protocol.write_text('changed scientific protocol\n')
            with self.assertRaisesRegex(ValueError,'protocol drift'):load_registration(root)

    def test_alias_component_and_development_leak_refuse(self):
        for component, train, test, dev in [
            (['a','a','b'],[0],[1],[2]),
            (['a','b','a'],[0],[1],[2]),
            (['a','b','c'],[0,2],[1],[2]),
            (['a','b','c'],[0],[0,1],[2]),
        ]:
            with self.assertRaises(ValueError):
                study.validate_partition(component,train,test,dev)
        study.validate_partition(['a','b','c'],[0],[1],[2])

    def test_equal_search_and_paired_seeds(self):
        cells=study.matrix()
        for partition in ('inner-0-0','inner-2-1'):
            expected={(setting,seed) for setting in range(4) for seed in range(3)}
            for family in (*study.NEURAL,'extra_trees'):
                got={(c['setting'],c['seed']) for c in cells if c['stage']=='inner'
                     and c['family']==family and c['partition']==partition}
                self.assertEqual(got,expected)
        self.assertTrue(all(c['seed'] is None for c in cells if c['family']=='shaped_additive'))
        self.assertEqual(len({c['id'] for c in cells}),len(cells))

    def test_rescue_is_equal_opportunity_and_not_silently_required(self):
        cells=study.matrix()
        for family in study.NEURAL:
            selected=[c for c in cells if c['family']==family and c['stage']=='rescue']
            self.assertEqual({c['rescue'] for c in selected},set(study.RESCUES))
            self.assertTrue(all('condition' in c for c in selected))
        pilot=[c for c in cells if c['stage']=='resource_pilot']
        self.assertEqual(len(pilot),1)
        self.assertFalse(pilot[0]['scientific_selection'])


if __name__=='__main__':unittest.main()
