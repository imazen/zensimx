import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_worker_collect import unpack,inventory,validate,pack,collect,verify_transfer,pack_reserved
from shaping_completion import encoded,sha
from shaping_rescue_execution import portable_command


class CollectionTests(unittest.TestCase):
    def test_pack_preserves_inputs_for_portable_rescue(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            study=Path(d);root=study/'runs/cell/attempt-001';root.mkdir(parents=True)
            declaration=study/'worker-declaration.json';declaration.write_bytes(b'{"inputs":[]}\n')
            pin=sha(declaration)
            command=['trainer','--derived-inputs',str(declaration)]
            cell=dict(id='cell',family='relu',stage='inner')
            (study/'PLAN.json').write_bytes(encoded(dict(minimum_ssd_free_bytes=0)))
            run=dict(cell=cell,command=command,execution_host='worker',
                     plan_sha256=sha(study/'PLAN.json'),build_sha256='build',
                     epochs=240,pairs_per_epoch=50000)
            (root/'RUN.json').write_bytes(encoded(run))
            checkpoints=root/'checkpoints';checkpoints.mkdir()
            for epoch in range(240):
                (checkpoints/f'ckpt_epoch{epoch:03}.bin').write_bytes(bytes([epoch]))
            (root/'RESULT.json').write_bytes(encoded(dict(run,stage='trained',exit_code=0,
                bake_sha256=sha(checkpoints/'ckpt_epoch239.bin'))))
            assignment=study/'assignment.json'
            assignment.write_bytes(encoded(dict(host='worker',cells=['cell'],
                plan_sha256=sha(study/'PLAN.json'),build_sha256='build')))
            before=inventory(root);bundle=study/'bundle.tar'
            with patch('shaping_execution.load_registration',return_value=({},[cell],{})), \
                 patch('shaping_worker_collect.socket.gethostname',return_value='worker'), \
                 patch('shaping_worker_collect.DECLARATION_SHA256',pin):
                receipt=pack(study,assignment,'cell','attempt-001',bundle)
                with self.assertRaisesRegex(ValueError,'changed during storage admission'):
                    pack_reserved(study,assignment,'cell','attempt-001',study/'changed.tar',{})
                self.assertFalse((study/'changed.tar').exists())
            self.assertEqual(inventory(root),before)
            collector=study/'collector';collector.mkdir();(collector/'runs').mkdir()
            (collector/'PLAN.json').write_bytes((study/'PLAN.json').read_bytes())
            with patch('shaping_execution.load_registration',return_value=(dict(minimum_ssd_free_bytes=0),[cell],{})):
                out=Path(collect(collector,assignment,bundle,receipt['bundle_sha256']))
                self.assertFalse((collector/'worker_collection'/receipt['bundle_sha256']).exists())
                with self.assertRaises(FileExistsError):
                    collect(collector,assignment,bundle,receipt['bundle_sha256'])
            self.assertEqual((out/'RUN.json').read_bytes(),(root/'RUN.json').read_bytes())
            declaration.unlink() # Original worker path is absent on the collector.
            with patch('shaping_rescue_execution.DECLARATION_SHA256',pin):
                relocated=portable_command(command,out)
                self.assertEqual(relocated[-1],str(out/'COLLECTION_DERIVED_INPUTS.json'))
                self.assertEqual(command[-1],str(declaration))
                with self.assertRaisesRegex(ValueError,'path differs'):
                    portable_command(['trainer','--derived-inputs','wrong'],out)
                (out/'COLLECTION_DERIVED_INPUTS.json').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'declaration drift'):
                    portable_command(command,out)

    def test_transfer_readback_and_changed_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);bundle=root/'bundle.tar';bundle.write_bytes(b'known producer bytes')
            assignment=root/'assignment.json';assignment.write_bytes(encoded(dict(cells=['cell'])))
            pin=sha(bundle)
            Path(str(bundle)+'.json').write_bytes(encoded(dict(bundle_sha256=pin,
                assignment_sha256=sha(assignment),cell=dict(id='cell'))))
            receipt=verify_transfer(bundle,pin,assignment)
            self.assertEqual(receipt['bundle_sha256'],pin)
            self.assertEqual(receipt['bundle_bytes'],len(b'known producer bytes'))
            with self.assertRaises(FileExistsError):verify_transfer(bundle,pin,assignment)
            bundle.write_bytes(b'changed transfer bytes')
            with self.assertRaisesRegex(ValueError,'checksum'):verify_transfer(bundle,pin,assignment)

    def test_readback_and_corruption(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);src=root/'src';src.mkdir();(src/'model.bin').write_bytes(b'original model bytes')
            files=inventory(src);bundle=root/'bundle.tar'
            with tarfile.open(bundle,'w') as tar:
                data=encoded(dict(files=files));info=tarfile.TarInfo('COLLECTION.json');info.size=len(data)
                tar.addfile(info,io.BytesIO(data));tar.add(src/'model.bin',arcname='payload/model.bin')
            unpack(bundle,root/'out',sha(bundle));self.assertEqual(inventory(root/'out'),files)
            with self.assertRaisesRegex(ValueError,'checksum'):unpack(bundle,root/'wrong','0'*64)
            with bundle.open('r+b') as f:
                data=f.read();position=data.index(b'original model bytes');f.seek(position);f.write(b'changed!')
            with self.assertRaisesRegex(ValueError,'checksum'):unpack(bundle,root/'corrupt',sha(bundle))

    def test_traversal_and_symlink_refused(self):
        for name,kind in [('payload/../escape',tarfile.REGTYPE),('payload/link',tarfile.SYMTYPE)]:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
                root=Path(d);bundle=root/'bundle.tar'
                with tarfile.open(bundle,'w') as tar:
                    data=encoded(dict(files={name[8:]:dict(bytes=0,sha256='bad')}))
                    info=tarfile.TarInfo('COLLECTION.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
                    info=tarfile.TarInfo(name);info.type=kind;info.linkname='/outside';tar.addfile(info)
                with self.assertRaises(ValueError):unpack(bundle,root/'out',sha(bundle))
                self.assertFalse((root/'escape').exists())

    def test_terminal_fit_requires_all_checkpoints_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);ck=root/'checkpoints';ck.mkdir()
            for i in range(240):(ck/f'ckpt_epoch{i:03}.bin').write_bytes(bytes([i]))
            cell=dict(id='cell',family='relu',stage='inner')
            assignment=dict(host='worker',cells=['cell'],plan_sha256='plan',build_sha256='build')
            run=dict(cell=cell,execution_host='worker',plan_sha256='plan',build_sha256='build',epochs=240,pairs_per_epoch=50000)
            result=dict(run,stage='trained',exit_code=0,bake_sha256=sha(ck/'ckpt_epoch239.bin'))
            (root/'RUN.json').write_bytes(encoded(run));(root/'RESULT.json').write_bytes(encoded(result))
            validate(root,assignment,[cell])
            (ck/'ckpt_epoch000.bin').unlink()
            with self.assertRaisesRegex(ValueError,'coverage'):validate(root,assignment,[cell])
            result['stage']='running';(root/'RESULT.json').write_bytes(encoded(result))
            with self.assertRaisesRegex(ValueError,'successful'):validate(root,assignment,[cell])


if __name__=='__main__':unittest.main()
