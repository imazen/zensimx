"""Synthetic label-parser and one-read journal controls; no corpus inputs."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('final_read_a', Path(__file__).resolve().parents[1]/'byconstruction/final_read_a.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class FinalReadControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', str(Path.home()/'tmp')))
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, name, source, label_spec):
        inputs=self.root/'inputs'/name;inputs.mkdir(parents=True)
        out=self.root/'scores'/name;out.mkdir(parents=True)
        source['label_spec']=label_spec
        owner.write(inputs/'KEYS.json',source)
        owner.write(out/'LABEL_READS.json',dict(state='authorized-before-read',key_sha256=owner.sha(inputs/'KEYS.json')))
        return type('Args',(),dict(root=self.root,set=name))(),out

    def test_key_projection_never_converts_poison_target(self):
        p=self.root/'keys.tsv';p.write_text('ref\titem\tscore\nr\td\tNOT_A_NUMBER\n')
        self.assertEqual(list(owner.key_rows(p,['ref','item'])),[(0,['r','d'])])

    def test_comma_delimited_key_projection_never_converts_poison_target(self):
        p=self.root/'dmos.csv';p.write_text('ref_img,dist_img,dmos\nr.png,d.png,NOT_A_NUMBER\n')
        self.assertEqual(list(owner.key_rows(p,['ref_img','dist_img'])),[(0,['r.png','d.png'])])

    def test_failed_parse_spends_read_and_prevents_second_open(self):
        p=self.root/'labels.tsv';p.write_text('human_score\ninvalid\n')
        a,out=self.prepare('csiq',dict(rows=[dict(pair_key='p',origin='r',source_row=0)]),
                           dict(path=str(p),sha256=owner.sha(p),target='human_score'))
        with patch.dict(owner.EXPECTED,csiq=1):
            with self.assertRaises(ValueError):owner.read(a)
            self.assertEqual(json.loads((out/'LABEL_READS.json').read_text())['state'],'failed-spent')
            with patch('builtins.open',side_effect=AssertionError('second label open')) as opened:
                with self.assertRaises(AssertionError):owner.read(a)
                opened.assert_not_called()

    def test_every_clic_question_including_tied_majority_is_retained(self):
        p=self.root/'pairs.tsv';p.write_text('n_a\tn_b\n3\t1\n1\t1\n1\t3\n')
        questions=[dict(source_row=i,ref=f'r{i}',a='a',b='b') for i in range(3)]
        a,out=self.prepare('clic22',dict(rows=[],questions=questions),dict(path=str(p),sha256=owner.sha(p)))
        with patch.dict(owner.EXPECTED,clic22=3):owner.read(a)
        rows=json.loads((out/'eval_rows.json').read_text())
        self.assertEqual([r['target'] for r in rows],[1.0,.5,0.0])
        self.assertEqual([r['source_row'] for r in rows],[0,1,2])

    def test_mcl_mean_subject_counts_and_q100_anchor(self):
        pins=self.root/'pins';pins.mkdir()
        raw=[];rows=[]
        for source in range(1,51):
            p=self.root/f'JND_image_{source:02d}.txt'
            p.write_text(''.join(f'Subject{i} {[80,50] if i<15 else [60,20]}\n' for i in range(30)))
            raw.append(dict(path=str(p),sha256=owner.sha(p)))
            rows.extend(dict(pair_key=f'{source}:{q}',origin=f'SRC{source:02d}',source=source,q=q) for q in (100,80,50))
        owner.write(pins/'MCL_RAW_SHA256.json',dict(files=raw))
        a,out=self.prepare('mcljci',dict(rows=rows),dict(target='registered chain count'))
        with patch.dict(owner.EXPECTED,mcljci=150):owner.read(a)
        got=json.loads((out/'eval_rows.json').read_text())
        self.assertEqual(len(got),150)
        self.assertEqual([r['target'] for r in got[:3]],[0.0,.5,1.5])
        receipt=json.loads((out/'LABEL_READS.json').read_text())
        self.assertEqual(receipt['rebuilt_table_sha256'],owner.sha(out/'rebuilt_jnd_dist.tsv'))
        self.assertEqual(receipt['lost_table_byte_identity'],'cannot be verified')


def panel_control(root):
    """Analytical monotone controls through all 10,000 actual Rust resamples."""
    from types import SimpleNamespace
    a=SimpleNamespace(root=Path(root)/'controls',set='synthetic')
    (a.root/'bin').mkdir(parents=True,exist_ok=True)
    (a.root/'bin/panel').symlink_to(Path(root)/'bin/panel')
    rows=[dict(a=f'p{i}',group=f'ref{i//3}',target=float(i)) for i in range(12)]
    values={name:{r['a']:sign*r['target'] for r in rows} for name,sign in [('positive',1),('negative',-1)]}
    result=owner.mos_stats(a,rows,values,'signed-reference-control')
    for name,sign in [('positive',1),('negative',-1)]:
        for stat in ('srocc_signed','krocc','plcc','within_reference_srocc'):
            cell=result['models'][name][stat]
            assert sign*cell['value']>.99, (name,stat,cell)
            assert cell['valid_draws']==10000, (name,stat,cell)
    print('native signed point and 10000-draw interval controls passed')


if __name__ == '__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--panel-control':panel_control(sys.argv[2])
    else:unittest.main()
