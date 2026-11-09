import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
import shaping_streaming_memory as owner
from shaping_completion import encoded,sha


def fixture(root):
    (root/'PLAN.json').write_text('{}');cases={};pins={}
    for family in owner.FAMILIES:
        p=root/family;p.mkdir();result=p/'RESULT.json';log=p/'service.log'
        result.write_bytes(encoded(dict(cell=dict(family=family),stage='trained',exit_code=0,epochs=480,build_sha256='build')))
        receipt='run-heavy: done rc=0 100s | peak-RSS 0.17GiB | min-avail 9000MiB | peak-load 2.00';log.write_text(receipt+'\n')
        cases[family]=dict(result=str(result),service_log=str(log),whole_worker_receipt=receipt,build_sha256='build')
        pins.update({str(p):sha(p) for p in (result,log)})
    q=dict(schema='streaming-480-memory-qualification-v1',plan_sha256=sha(root/'PLAN.json'),cap_bytes=owner.CAP,
           families=list(owner.FAMILIES),epochs=480,scientific_inputs_changed=False,native_binaries_changed=False,
           numerical_gates_changed=False,cases=cases,evidence_sha256=pins)
    (root/owner.NAME).write_bytes(encoded(q))
    return dict(study=str(root),execution_kind='outer',cell=dict(id='cell',family='relu'),build_sha256='build',
                resolution=dict(schedule=dict(phase_epochs=[480])),files={},input_bytes=0)


class StreamingMemoryTests(unittest.TestCase):
    def test_new_sealed_hint_matches_parent_cap_and_refuses_evidence_drift(self):
        from shaping_rescue_worker import memory_allowance
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);record=fixture(root)
            self.assertEqual(memory_allowance(record),4*1024**3)
            owner.attach(record,root,lambda p:('data',str(p)),Path(owner.__file__))
            self.assertEqual(memory_allowance(record),2*1024**3)
            unbound=copy.deepcopy(record);del unbound['files'][str(root/'relu/service.log')]
            with self.assertRaisesRegex(ValueError,'sealed qualification'):owner.validate(unbound)
            for wrong in [dict(record,execution_kind='extension'),dict(record,cell=dict(id='cell',family='softplus')),
                          dict(record,streaming_memory=dict(record['streaming_memory'],cap_bytes=1))]:
                with self.assertRaises(ValueError):owner.validate(wrong)
            (root/'relu/service.log').write_text('changed evidence')
            with self.assertRaisesRegex(ValueError,'evidence drift'):owner.validate(record)

    def test_unqualified_families_schedules_and_builds_keep_old_caps(self):
        from shaping_rescue_worker import memory_allowance
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);record=fixture(root)
            for changed in [dict(record,cell=dict(family='softplus')),dict(record,build_sha256='another-build'),
                            dict(record,execution_kind='extension')]:
                before=copy.deepcopy(changed);owner.attach(changed,root,lambda p:('data',str(p)),Path(owner.__file__))
                self.assertEqual(changed,before);self.assertEqual(memory_allowance(changed),4*1024**3)

    def test_native_outer_and_human_only_commands_change_only_wrapper_memory(self):
        from shaping_outer_execution import command as outer
        from shaping_variant_execution import command as variant
        groups=[(n,'/'+n,1.,0.,'withinref,both' if n=='cid22' else 'withinref,rank') for n in ('human_kadid','human_tid2013','human_konfig','human_cid22_a25','cid22')]
        original=['run-heavy','--mem','2G','--jobs','1','--','/old/zensim_mlp_train','--epochs','240',
                  '--out','/old/best.bin','--dump-checkpoints-dir','/old/checkpoints','--mse-weight','1',
                  '--init-seed','1101','--sample-seed','101','--pairs-per-epoch','50000']
        for builder,args in [(outer,(groups,)),(variant,(groups[:-1],['cid22']))]:
            old=builder(original,Path('/native'),Path('/new'),480,*args)
            new=builder(original,Path('/native'),Path('/new'),480,*args,memory_cap='2G')
            self.assertEqual(old[old.index('--mem')+1],'4G');self.assertEqual(new[new.index('--mem')+1],'2G')
            new[new.index('--mem')+1]='4G';self.assertEqual(new,old)
        with self.assertRaises(ValueError):owner.native_cap('softplus','2G')

    def test_worker_passes_the_sealed_cap_to_the_native_owner(self):
        import shaping_downstream_worker as worker
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);record=fixture(root);record.update(binary_dir='/bin',neutral_qualification='/qual',human_qualification=None)
            owner.attach(record,root,lambda p:('data',str(p)),Path(owner.__file__))
            with patch('shaping_runtime_equivalence.verify_binding'),patch.object(worker,'arguments',return_value=SimpleNamespace(outer=0)), \
                 patch.object(worker,'admitted',return_value={}),patch.object(worker,'scope',return_value=record['resolution']):
                resolution,command=worker.reproduce(record,host='r3500')
            self.assertEqual(resolution,record['resolution']);self.assertEqual(command[-2:],['--streaming-memory','2G'])

    def test_warmup_headroom_is_cleared_only_after_checkpoint_for_own_claim(self):
        from shaping_rescue_dispatch import OWNER_TOKEN
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);p=root/'cpu-admission/CLAIMS.json';p.parent.mkdir()
            rows={'cell':dict(owner=OWNER_TOKEN,warmup_until_checkpoint=True),'other':dict(owner='other',warmup_until_checkpoint=True)};p.write_bytes(encoded(rows))
            checkpoint=root/'runs/cell/attempt-001/fit/checkpoints/ckpt_epoch000.bin'
            class Process:
                calls=0
                def poll(self):
                    self.calls+=1
                    if self.calls==2:checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'checkpoint')
                    return None
                def wait(self):return 0
            def during_warmup(seconds):self.assertTrue(json.loads(p.read_text())['cell']['warmup_until_checkpoint'])
            with patch.object(owner.time,'sleep',side_effect=during_warmup):self.assertEqual(owner.wait_for_first_checkpoint(Process(),root,'cell'),0)
            after=json.loads(p.read_text());self.assertNotIn('warmup_until_checkpoint',after['cell']);self.assertEqual(after['other'],rows['other'])

    def test_warmup_cannot_change_another_owners_claim(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);p=root/'cpu-admission/CLAIMS.json';p.parent.mkdir()
            rows={'cell':dict(owner='another owner',warmup_until_checkpoint=True)};p.write_bytes(encoded(rows))
            checkpoint=root/'runs/cell/attempt-001/fit/checkpoints/ckpt_epoch000.bin';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'checkpoint')
            from types import SimpleNamespace
            process=SimpleNamespace(poll=lambda:None)
            with self.assertRaisesRegex(ValueError,'claim owner differs'):owner.wait_for_first_checkpoint(process,root,'cell')
            self.assertEqual(json.loads(p.read_text()),rows)
