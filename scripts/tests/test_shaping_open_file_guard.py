import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_open_file_guard import inspect
from shaping_checkpoint_retention import no_users


class OpenFileGuardTests(unittest.TestCase):
    def fixture(self,root):
        target=root/'state.snap';target.write_bytes(b'preserve');stat=target.stat()
        process=root/'proc/42';(process/'fdinfo').mkdir(parents=True);(process/'fd').mkdir()
        (process/'maps').write_text('')
        device=f'{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}'
        (process/'mountinfo').write_text(f'1 0 {device} / / rw - ext4 local rw\n2 1 0:999 / /unrelated rw - nfs stale rw\n')
        return target,stat,process

    def test_hardlink_or_bind_alias_descriptor_and_mapping_are_detected(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);target,stat,process=self.fixture(root);alias=root/'other-name';os.link(target,alias)
            (process/'fd/3').symlink_to(alias)
            (process/'fdinfo/3').write_text(f'mnt_id:\t1\nino:\t{stat.st_ino}\n')
            value=inspect([target],root/'proc');self.assertFalse(value['errors']);self.assertEqual(value['busy'][0]['kind'],'descriptor')
            (process/'fdinfo/3').unlink();(process/'fd/3').unlink()
            (process/'maps').write_text(f'1000-2000 r--p 00000000 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} {alias}\n')
            value=inspect([target],root/'proc');self.assertFalse(value['errors']);self.assertEqual(value['busy'][0]['kind'],'mapping')

    def test_unrelated_network_descriptor_never_gets_statted(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);target,stat,process=self.fixture(root)
            # Same inode number on another device must not cause an NFS stat.
            (process/'fdinfo/3').write_text(f'mnt_id:\t2\nino:\t{stat.st_ino}\n')
            (process/'fd/3').symlink_to('/unrelated/unavailable-file')
            original=os.stat
            def checked(path,*args,**kwargs):
                if str(path)==str(process/'fd/3'):self.fail('unrelated network descriptor was statted')
                return original(path,*args,**kwargs)
            with patch('shaping_open_file_guard.os.stat',side_effect=checked):value=inspect([target],root/'proc')
            self.assertFalse(value['busy']);self.assertFalse(value['errors'])

    def test_network_mount_uses_complete_bounded_inspection_and_busy_still_refuses(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            mount=Path(directory)/'mountinfo';mount.write_text('1 0 0:1 / /archive rw - nfs server rw\n')
            response=dict(schema='inode-open-file-inspection-v1',targets=['/state'],busy=[],errors=[],scanned_processes=1)
            with patch('shaping_checkpoint_retention.subprocess.run') as run:
                run.return_value.returncode=0;run.return_value.stderr='';run.return_value.stdout=json.dumps(response)
                no_users([dict(path='/state')],mountinfo=mount)
                self.assertEqual(run.call_args.args[0][:5],['sudo','-n','timeout','--kill-after=1s','15s'])
                self.assertEqual(json.loads(run.call_args.kwargs['input'])['targets'],['/state'])
                response['busy']=[dict(path='/state',pid=42,kind='mapping')];run.return_value.stdout=json.dumps(response)
                with self.assertRaisesRegex(ValueError,'still in use'):no_users([dict(path='/state')],mountinfo=mount)
                response.update(busy=[],errors=['permission']);run.return_value.stdout=json.dumps(response)
                with self.assertRaisesRegex(ValueError,'incomplete'):no_users([dict(path='/state')],mountinfo=mount)


if __name__=='__main__':unittest.main()
