import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix
from shaping_stable_pairs import FAMILIES
from shaping_stable_export_watch import watch


class FixedWaveTests(unittest.TestCase):
    def test_overlapping_prior_waves_refuse_before_any_collection_or_new_work(self):
        cells=matrix();cell=next(c for c in cells if c['stage']=='inner' and c['family'] in FAMILIES)
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);previous=[]
            for name in ('wave1','wave2'):
                p=root/name/'REGISTRATION.json';p.parent.mkdir();p.write_text(json.dumps({'jobs':[{'cell':cell}]}));previous.append(p)
            args=SimpleNamespace(output=root/'watch',study=root/'study',source=root/'source',host='r3500',previous_wave=previous)
            with patch('shaping_stable_export_watch.load_registration',return_value=({},cells,{})), \
                 patch('shaping_stable_export_watch.collect') as collect, \
                 patch('shaping_stable_export_watch.launch') as launch, \
                 patch('shaping_stable_export_watch.worker_ready') as ready:
                with self.assertRaisesRegex(ValueError,'overlapping'):watch(args)
                collect.assert_not_called();launch.assert_not_called();ready.assert_not_called()
            status=json.loads((args.output/'STATUS.json').read_text())
            self.assertEqual(status['stage'],'failed_with_preserved_evidence')


if __name__=='__main__':unittest.main()
