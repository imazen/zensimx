"""Metadata binding fixtures, not native arithmetic qualification."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import NEURAL,encoded,sha
from shaping_cost_complexity import validate_fold


def fixture(directory,workload,cells,pins):
    directory.mkdir();w=json.loads(workload.read_text());binary=workload.parent.parent.parent/'shaping_model_inventory';binary.write_bytes(b'qualified-tool-fixture');pins[str(binary)]=sha(binary)
    output=[];native=[];by_cell={r['role']+'-'+('deterministic' if r['cell']['seed'] is None else str(r['cell']['seed'])):r for r in cells}
    for arm in w['arms']:
        row=by_cell[arm['id']];cell=row['cell'];item=dict(id=arm['id'],cell=cell,role=row['role'],resolved_family=cell['family'])
        if cell['family'] in NEURAL:
            for field,values in [('parameters',dict(trainable_scalar_entries=12,frozen_response_shape_values=0)),('training_resources',dict(directly_executed_phase_wrapper_seconds=10))]:
                path=directory/(arm['id']+'-'+field+'.json');path.write_bytes(encoded(dict(cell=cell,source_pins=copy.deepcopy(pins),**values)))
                item[field]=dict(path=path.name,sha256=sha(path))
            item.update(trainable_scalar_entries=12,frozen_response_shape_values=0,training_wrapper_seconds=10)
        elif cell['family']=='shaped_additive':
            source=workload.parent.parent.parent/('additive-fit-'+workload.parent.name+'.json');source.write_bytes(encoded(dict(coefficients=[1.,0.,2.])));pins[str(source)]=sha(source)
            item.update(coefficient_source=str(source),coefficient_source_sha256=sha(source),adjustable_basis_coefficients=3,nonzero_basis_coefficients=2)
        output.append(item);model=workload.parent/arm['model'];native.append(dict(id=arm['id'],model_sha256=arm['sha256'],model_bytes=model.stat().st_size))
    path=directory/'NATIVE.json';path.write_bytes(encoded(dict(schema='shaping-model-inventory-v1',workload_sha256=sha(workload),binary_sha256=sha(binary),models=native,timing_measured=False)))
    record=dict(schema='shaping-finalist-complexity-v1',workload_sha256=sha(workload),native_inventory_sha256=sha(path),native_inventory_binary_sha256=sha(binary),models=output,timing_measured=False,training_replayed=False)
    (directory/'SUMMARY.json').write_bytes(encoded(record));return record


class ComplexityTests(unittest.TestCase):
    def test_complete_bindings_and_changed_counts_models_roles_or_sources_refuse(self):
        from scripts.tests.test_shaping_cost_worker import fixture as workload_fixture
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            _,prepared,_,record=workload_fixture(Path(d));root=prepared/'outer-0';workload=root/'WORKLOAD.json';directory=root/'complexity'
            cells=json.loads((root/'PROVENANCE.json').read_text())['cells'];pins=json.loads((root/'PROVENANCE.json').read_text())['source_pins']
            validate_fold(directory,workload,cells,pins)
            path=directory/'SUMMARY.json';original=path.read_bytes();q=json.loads(original)
            neural=next(r for r in q['models'] if r['cell']['family'] in NEURAL);neural['trainable_scalar_entries']+=1;path.write_bytes(encoded(q))
            with self.assertRaisesRegex(ValueError,'summary differs'):validate_fold(directory,workload,cells,pins)
            path.write_bytes(original);q=json.loads(original);q['models'][0]['role']='wrong';path.write_bytes(encoded(q))
            with self.assertRaisesRegex(ValueError,'role differs'):validate_fold(directory,workload,cells,pins)
            path.write_bytes(original);q=json.loads(original);q['models'].pop();path.write_bytes(encoded(q))
            with self.assertRaisesRegex(ValueError,'sixteen'):validate_fold(directory,workload,cells,pins)
            path.write_bytes(original);changed=dict(pins);changed[next(p for p in pins if p.endswith('evidence.json'))]='changed'
            with self.assertRaisesRegex(ValueError,'unbound'):validate_fold(directory,workload,cells,changed)
            model=root/json.loads(workload.read_text())['arms'][0]['model'];model.write_bytes(b'longer changed model')
            with self.assertRaisesRegex(ValueError,'timed representation'):validate_fold(directory,workload,cells,pins)


if __name__=='__main__':unittest.main()
