import io,json,sys,tarfile,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_completion import matrix,encoded,sha
from shaping_pretraining import matching_parent
from shaping_host_assignment import parent_inputs
from shaping_worker_input_install import install


class ParentTransportTests(unittest.TestCase):
    def test_completed_matching_parent_and_exact_artifacts(self):
        cells=matrix();child=next(c for c in cells if c['stage']=='inner' and c['family']=='product_pretrained_frozen')
        parent=matching_parent(child,cells)
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp:
            study=Path(temp)/'study';run=study/'runs'/parent['id']/'attempt-001';(run/'checkpoints').mkdir(parents=True)
            (study/'PLAN.json').write_text('{}')
            for epoch in [119,239]:(run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin').write_bytes(bytes([epoch]))
            (run/'SHAPING.json').write_text('{}')
            identity=dict(cell=parent,plan_sha256=sha(study/'PLAN.json'),build_sha256='build',epochs=240,pairs_per_epoch=50000)
            (run/'RUN.json').write_bytes(encoded(identity))
            result=dict(stage='trained',exit_code=0,bake_sha256=sha(run/'checkpoints/ckpt_epoch239.bin'))
            (run/'RESULT.json').write_bytes(encoded(result))
            rows=parent_inputs(study,cells,[child,child],'build')
            self.assertEqual(len(rows),5)
            self.assertEqual({Path(r['path']).name for r in rows},{'RUN.json','RESULT.json','SHAPING.json','ckpt_epoch119.bin','ckpt_epoch239.bin'})
            with self.assertRaisesRegex(ValueError,'identity'):parent_inputs(study,cells,[child],'different')
            (run/'checkpoints/ckpt_epoch239.bin').write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError,'checkpoint'):parent_inputs(study,cells,[child],'build')

    def test_capsule_parent_paths_are_explicit_and_never_overwritten(self):
        for bad in [False,True]:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temp:
                root=Path(temp);study=root/'study';assignment=study/'worker_assignments/a.json'
                parent=study/'runs'/('a'*20)/'attempt-001'/('checkpoints/ckpt_epoch118.bin' if bad else 'checkpoints/ckpt_epoch119.bin')
                content=b'prefix';pin=__import__('hashlib').sha256(content).hexdigest();row=dict(path=str(parent),bytes=len(content),sha256=pin)
                assignment_bytes=encoded(dict(parent_inputs=[row]));arow=dict(path=str(assignment),bytes=len(assignment_bytes),sha256=__import__('hashlib').sha256(assignment_bytes).hexdigest())
                manifest=root/'INPUTS.json';manifest.write_bytes(encoded([row,arow]));bundle=root/'inputs.tar'
                with tarfile.open(bundle,'w') as tar:
                    for path,data in [(parent,content),(assignment,assignment_bytes)]:
                        info=tarfile.TarInfo(str(path).lstrip('/'));info.size=len(data);tar.addfile(info,io.BytesIO(data))
                args=(study,'inner-0-0',assignment,bundle,manifest,sha(bundle),sha(manifest))
                if bad:
                    with self.assertRaisesRegex(ValueError,'unapproved'):install(*args,root/'receipt.json')
                    self.assertFalse(parent.exists());continue
                install(*args,root/'receipt.json');self.assertEqual(parent.read_bytes(),content)
                parent.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'existing input differs'):install(*args,root/'retry.json',reuse_identical=True)
                self.assertEqual(parent.read_bytes(),b'changed')


if __name__=='__main__':unittest.main()
