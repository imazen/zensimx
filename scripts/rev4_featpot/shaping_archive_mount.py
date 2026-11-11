"""Refresh an explicitly owned study archive bind after an NFS stale handle."""
import argparse
import errno
import json
from pathlib import Path
import shutil
import subprocess
import time


def ensure(target, source, mount_root, subdirectory):
    try:
        shutil.disk_usage(target)
        return False
    except OSError as error:
        if error.errno != errno.ESTALE:
            raise
    if target != Path("/mnt/tower") / subdirectory or ".." in Path(subdirectory).parts:
        raise ValueError("explicit owned study bind required")
    actual = subprocess.check_output(["findmnt","-n","-o","TARGET","--mountpoint",str(target)],text=True).strip()
    if actual != str(target):
        raise ValueError("refuse replacing another archive mount")
    fresh=mount_root/("generation-"+str(time.time_ns()));fresh.mkdir(parents=True)
    subprocess.run(["sudo","-n","mount","-t","nfs4",source,str(fresh)],check=True,timeout=30)
    location=fresh/subdirectory
    if not location.is_dir():
        raise ValueError("fresh archive export lacks the registered study directory")
    subprocess.run(["sudo","-n","umount",str(target)],check=True,timeout=30)
    subprocess.run(["sudo","-n","mount","--bind",str(location),str(target)],check=True,timeout=30)
    shutil.disk_usage(target)
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("target","mount-root"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--source",required=True);p.add_argument("--subdirectory",required=True)
    a=p.parse_args();print(json.dumps(dict(recovered=ensure(a.target,a.source,a.mount_root,a.subdirectory))))
