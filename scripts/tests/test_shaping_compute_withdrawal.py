import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import sha
from shaping_compute_pool import allowed,EXCLUDED
from shaping_outer_pipeline import placements
from shaping_placement import loads


class ComputeWithdrawalTests(unittest.TestCase):
    def test_withdrawal_preserves_original_pool_and_requires_bound_direction(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            root=Path(tmp);(root/'PLAN.json').write_text('{}')
            approval=root/'approval.json';approval.write_text(json.dumps(dict(approved=True,
                all_other_known_hosts_authorized=True,excluded_hosts=sorted(EXCLUDED))))
            pool=root/'COMPUTE_POOL.json';pool.write_text(json.dumps(dict(schema='shaping-expanded-compute-pool-v1',
                approval_path=str(approval),approval_sha256=sha(approval),plan_sha256=sha(root/'PLAN.json'),
                excluded_hosts=sorted(EXCLUDED),allowed_hosts=['i270','r3800x'],scientific_protocol_unchanged=True)))
            original=pool.read_bytes();self.assertEqual(allowed(root),{'i270','r3800x'})
            direction=root/'direction.json';direction.write_text(json.dumps(dict(host='i270',new_compute_authorization_withdrawn=True)))
            p=root/'COMPUTE_WITHDRAWALS.json';p.write_text(json.dumps(dict(schema='shaping-compute-withdrawals-v1',
                plan_sha256=sha(root/'PLAN.json'),direction_path=str(direction),direction_sha256=sha(direction),hosts=['i270'])))
            self.assertEqual(allowed(root),{'r3800x'});self.assertEqual(pool.read_bytes(),original)
            direction.write_text('{}')
            with self.assertRaisesRegex(ValueError,'withdrawal identity'):allowed(root)

    def test_withdrawn_host_is_not_probed_or_selected(self):
        row=dict(occupied=0,memory_available_bytes=20*1024**3,ssd_free_bytes=100*1024**3)
        with patch('shaping_compute_pool.allowed',return_value={'r3800x'}), \
             patch('shaping_placement.observation',return_value=row) as observe:
            weights,seen=loads(Path('study'),dict(i270=20,r3800x=14))
        observe.assert_called_once_with(Path('study'),'r3800x',14)
        self.assertEqual(set(weights),{'r3800x'})
        cells=[dict(id=str(i),seed=i%3,family='relu') for i in range(6)]
        result=placements(cells,1,weights)
        self.assertEqual(len(result),6);self.assertTrue(all(host=='r3800x' for _,host in result))


if __name__=='__main__':unittest.main()
