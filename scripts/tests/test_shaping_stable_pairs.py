"""Qualification is explicit; original unsupported-family admission stays closed."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import max_pair_families,sha
from shaping_reexport import admit_family
from shaping_stable_pairs import families,register,frozen_sources,NAME,FAMILIES,WITNESSES


class StablePairQualificationTests(unittest.TestCase):
    def test_qualification_source_is_portable_and_changed_copies_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);source=root/'immutable-export';scripts=source/'scripts/rev4_featpot';scripts.mkdir(parents=True)
            output=root/'qualification';output.mkdir()
            for name in ('shaping_probe_trace.py','shaping_reexport.py'):(scripts/name).write_text('frozen native owner source')
            copied=frozen_sources(output,source)
            self.assertTrue(all(p.is_relative_to(output) for p in copied))
            self.assertEqual(copied,frozen_sources(output,source))
            copied[0].write_text('changed evidence')
            with self.assertRaisesRegex(ValueError,'preserve changed'):frozen_sources(output,source)
            self.assertEqual(copied[0].read_text(),'changed evidence')

    def test_unregistered_study_keeps_original_admission_and_all_gate_expectations(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study=Path(directory)
            self.assertEqual(max_pair_families(study),('relu','softplus'))
            for family in FAMILIES:
                with self.assertRaises(ValueError):admit_family(family,True,study)
                with self.assertRaises(ValueError):admit_family(family,True)

    def test_malformed_certificate_never_falls_back_to_original(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            study=Path(directory);(study/'PLAN.json').write_text('{}')
            record=dict(schema='qualified-stable-max-pair-families-v1',families=list(FAMILIES),
                plan_sha256=sha(study/'PLAN.json'),build_sha256='wrong',fitting_changed=False,
                numerical_threshold_changed=False,outer_results_read=False,original_evidence_preserved=True)
            (study/NAME).write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'scope differs'):families(study)
            record['families'].append('silu_diagnostic');(study/NAME).write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'scope differs'):families(study)

    def test_witness_and_fixed_case_coverage_required_before_certificate_creation(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';study.mkdir();qualification=root/'qualification';qualification.mkdir()
            original=[dict(cell=k,epoch=v) for k,v in WITNESSES.items()];original[0]['epoch']-=1
            (qualification/'CASES.json').write_text(json.dumps(original));(qualification/'ADDITIONAL_CASES.json').write_text('[]')
            with patch('shaping_corrected_view.require_activation'),patch('shaping_execution.load_registration',return_value=({},[],{})):
                with self.assertRaisesRegex(ValueError,'four original witnesses'):register(study,qualification,root/'source')
            self.assertFalse((study/NAME).exists())


if __name__=='__main__':unittest.main()
