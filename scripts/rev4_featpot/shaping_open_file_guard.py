"""Inspect local target inodes without statting unrelated network mounts."""
import json,os,sys
from pathlib import Path


def inspect(targets,proc=Path('/proc')):
    identities={};paths=list(dict.fromkeys(map(str,targets)));errors=[];busy=[];scanned=0
    for name in paths:
        try:
            value=os.stat(name);identities[(os.major(value.st_dev),os.minor(value.st_dev),value.st_ino)]=name
        except OSError as error:errors.append(dict(target=name,error=str(error)))
    inodes={key[2] for key in identities};devices={key[:2] for key in identities}
    for directory in proc.glob('[0-9]*'):
        try:
            maps=(directory/'maps').read_text()
            for line in maps.splitlines():
                fields=line.split(None,5)
                if len(fields)<5 or int(fields[4]) not in inodes:continue
                major,minor=(int(v,16) for v in fields[3].split(':'))
                key=(major,minor,int(fields[4]))
                if key in identities:busy.append(dict(path=identities[key],pid=int(directory.name),kind='mapping'))
            mounts=None
            for info in (directory/'fdinfo').iterdir():
                try:
                    fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in info.read_text().splitlines() if ':' in line}
                    # Modern kernels provide the inode directly. This rejects
                    # unrelated descriptors before any filesystem lookup.
                    if 'ino' in fields and int(fields['ino']) not in inodes:continue
                    if mounts is None:
                        mounts={line.split()[0]:tuple(map(int,line.split()[2].split(':')))
                                for line in (directory/'mountinfo').read_text().splitlines()}
                    device=mounts.get(fields.get('mnt_id'))
                    if device is not None and device not in devices:continue
                    fd=directory/'fd'/info.name
                    if device is None:
                        alias=os.readlink(fd)
                        if not alias.startswith('/'):continue  # pipe/socket/anonymous inode
                        if 'ino' not in fields:raise ValueError('regular descriptor on an unidentified mount')
                    value=os.stat(fd)
                    key=(os.major(value.st_dev),os.minor(value.st_dev),value.st_ino)
                    if key in identities:busy.append(dict(path=identities[key],pid=int(directory.name),kind='descriptor'))
                except (FileNotFoundError,ProcessLookupError):continue
            scanned+=1
        except (FileNotFoundError,ProcessLookupError):continue
        except (OSError,ValueError) as error:errors.append(dict(pid=directory.name,error=str(error)))
    if not scanned:errors.append(dict(error='no processes were inspected'))
    return dict(schema='inode-open-file-inspection-v1',targets=paths,busy=busy,errors=errors,scanned_processes=scanned)


if __name__=='__main__':
    value=inspect(json.load(sys.stdin)['targets']);print(json.dumps(value),flush=True)
    sys.exit(bool(value['errors']))
