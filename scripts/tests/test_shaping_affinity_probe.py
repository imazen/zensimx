import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_affinity_probe import probe,apply_observation


class AffinityProbeTests(unittest.TestCase):
    def test_observation_application_restores_exact_original_and_refuses_drift(self):
        import json
        from shaping_completion import encoded,sha
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';observation=root/'observation';observation.mkdir()
            record=dict(pid=31,start_ticks='same',command=['zensim_mlp_train','--out',str(study/'runs/cell/model.bin')])
            original=dict(records=[record],threads=[dict(pid=31,tid=31,original=[0,1],cpu=1)])
            (observation/'ORIGINAL.json').write_bytes(encoded(original))
            receipt=dict(host='eligible',records=[record],original_affinities_restored=True,
                         files_sha256={'ORIGINAL.json':sha(observation/'ORIGINAL.json')})
            (observation/'RESULT.json').write_bytes(encoded(receipt));affinity={31:{0,1}}
            def set_(tid,cpus):affinity[tid]=set(cpus)
            with patch('shaping_compute_pool.allowed',return_value={'eligible'}), \
                 patch('shaping_training_budget.training_limit',return_value=1), \
                 patch('shaping_affinity_probe.socket.gethostname',return_value='eligible'), \
                 patch('shaping_affinity_probe.identity',return_value=record), \
                 patch('shaping_affinity_probe.os.sched_getaffinity',side_effect=lambda tid:affinity[tid]), \
                 patch('shaping_affinity_probe.os.sched_setaffinity',side_effect=set_):
                apply_observation(study,observation,root/'applied');self.assertEqual(affinity[31],{1})
                apply_observation(study,observation,root/'restored',restore=True);self.assertEqual(affinity[31],{0,1})
                affinity[31]={0}
                with self.assertRaisesRegex(ValueError,'affinity changed'):apply_observation(study,observation,root/'refused')
            self.assertFalse((root/'refused').exists())

    def test_withdrawn_worker_is_refused_before_observing_or_mutating_trainers(self):
        with patch('shaping_compute_pool.allowed',return_value={'eligible-worker'}), \
             patch('shaping_affinity_probe.socket.gethostname',return_value='withdrawn-worker'), \
             patch('shaping_affinity_probe.identity') as identity, \
             patch('shaping_affinity_probe.os.sched_setaffinity') as change:
            with self.assertRaisesRegex(ValueError,'withdrawn'):probe(Path('/study'),[1],[1],Path('/unused'),1)
        identity.assert_not_called();change.assert_not_called()

    def test_sampling_failure_restores_all_changed_threads(self):
        import os,json
        pid=os.getpid()
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);study=root/'study';output=root/'probe'
            record=dict(pid=pid,start_ticks='same',command=['zensim_mlp_train','--out',str(study/'runs/cell/fit.bin')])
            affinity={int(p.name):{0,1} for p in (Path('/proc')/str(pid)/'task').iterdir()}
            original={t:set(v) for t,v in affinity.items()}
            def get(tid):return {0,1} if tid==0 else affinity[tid]
            def set_(tid,cpus):affinity[tid]=set(cpus)
            def sample(records,path,seconds):
                if path.name=='pinned-perf.csv':raise RuntimeError('counter failure')
                return ['perf']
            with patch('shaping_compute_pool.allowed',return_value=None), \
                 patch('shaping_training_budget.training_limit',return_value=1), \
                 patch('shaping_affinity_probe.identity',return_value=record), \
                 patch('shaping_affinity_probe.os.sched_getaffinity',side_effect=get), \
                 patch('shaping_affinity_probe.os.sched_setaffinity',side_effect=set_), \
                 patch('shaping_affinity_probe.sample',side_effect=sample), \
                 patch('shaping_affinity_probe.time.sleep'):
                with self.assertRaisesRegex(RuntimeError,'counter failure'):probe(study,[pid],[1],output,1)
            self.assertEqual(affinity,original)
            self.assertEqual(json.loads((output/'RESTORED.json').read_text())['errors'],[])


if __name__=='__main__':unittest.main()
