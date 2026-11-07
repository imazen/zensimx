import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_worker_collect import unpack,inventory,validate
from shaping_completion import encoded,sha


class CollectionTests(unittest.TestCase):
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
