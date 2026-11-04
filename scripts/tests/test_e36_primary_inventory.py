"""Actual E36 owner: preserve three-seed numerics and support the owner cut."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OWNER=ROOT/'byconstruction/e36_exp56/primary.py'
FROZEN=ROOT/'tests/fixtures/e36_primary_before_r13.py'


class PrimaryInventory(unittest.TestCase):
    def test_default_matches_frozen_owner_and_cut_retains_three_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);selection=root/'selection.jsonl';pairs=root/'pairs.parquet'
            sources=[];rows=[]
            for group in range(4):
                key=hashlib.sha256(f'invented-{group}'.encode()).hexdigest()
                sources.append(dict(source=dict(source_sha256=key,identity=json.dumps(dict(origin=f'o{group}')))))
                for i in range(4):
                    rows.append(dict(reference_sha256=key,split='selection',scored=True,
                       F_s0_inv=False,F_s1_inv=i<1,F_s2_inv=i<2,
                       W32_s0_inv=i<2,W32_s1_inv=i<(group%4),W32_s2_inv=i<(3-group)))
            selection.write_text(''.join(json.dumps(s)+'\n' for s in sources))
            pq.write_table(pa.Table.from_pylist(rows),pairs)
            env=dict(os.environ,CONTROL='F');env.pop('ARM_SEEDS',None);env.pop('SEEDS',None)
            def run(script,name,extra=None):
                dest=root/f'{name}.json'
                subprocess.run([sys.executable,str(script),str(pairs),str(selection),str(dest),'F,W32'],
                               env=dict(env,**(extra or {})),capture_output=True,text=True,check=True)
                return dest.read_bytes(),json.loads(dest.read_text())
            before,_=run(FROZEN,'before')
            after,_=run(OWNER,'after')
            self.assertEqual(before,after)
            _,cut=run(OWNER,'cut',dict(ARM_SEEDS=json.dumps({'F':[0,1,2],'W32':[0]})))
            self.assertEqual(cut['F']['share'],0.25)
            self.assertEqual(cut['W32']['arm_share'],0.5)
            self.assertEqual(cut['W32']['diff'],0.25)
            self.assertEqual(cut['W32']['ci95'],[0.25,0.25])
            self.assertEqual(set(cut['F']['seeds']),{'F_s0','F_s1','F_s2'})
            self.assertEqual(set(cut['W32']['seeds']),{'W32_s0'})


if __name__=='__main__':unittest.main()
