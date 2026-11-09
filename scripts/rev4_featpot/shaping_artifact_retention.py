"""Standard-library archive identity and open-file guards shared by data owners."""
import json,re,subprocess,sys
from pathlib import Path
from shaping_completion import sha


def archive_inventory(directory):
    archive=json.loads((directory/'MIRROR_VERIFIED.json').read_text());mirror=Path(archive['destination'])
    if not archive['all_artifact_sha256_verified'] or sha(directory/'SHA256SUMS')!=sha(mirror/'SHA256SUMS'):
        raise ValueError('checkpoint archive identity differs')
    hashes={name:pin for pin,name in (line.split('  ',1) for line in (directory/'SHA256SUMS').read_text().splitlines())}
    return mirror,hashes



def open_targets(output,targets):
    return sorted({line[1:] for line in output.splitlines() if line.startswith('n') and line[1:] in targets})



def no_users(rows,*,mountinfo=Path('/proc/self/mountinfo')):
    # Enumerate descriptors/mappings once, avoiding an argument list containing
    # tens of thousands of checkpoint paths. Portal mounts are unrelated to /mnt/v.
    # Headless workers do not have the coordinator's desktop portal mounts.
    # lsof refuses a nonexistent -e target; exclude only actual mounted portals.
    mounts=mountinfo.read_text().splitlines()
    if any(line.split(' - ',1)[1].split()[0] in ('nfs','nfs4','fuse.gvfsd-fuse') for line in mounts):
        paths=[r['path'] for r in rows];helper=Path(__file__).with_name('shaping_open_file_guard.py')
        result=subprocess.run(['sudo','-n','timeout','--kill-after=1s','15s',sys.executable,str(helper)],
                              input=json.dumps(dict(targets=paths)),capture_output=True,text=True,timeout=20)
        if result.returncode or result.stderr:raise ValueError('inode open-file inspection failed: '+result.stderr+result.stdout)
        inspection=json.loads(result.stdout)
        if (inspection['schema']!='inode-open-file-inspection-v1' or set(inspection['targets'])!=set(paths)
            or inspection['errors']):raise ValueError('incomplete inode open-file inspection')
        if inspection['busy']:raise ValueError('checkpoint still in use: '+str(inspection['busy']))
        return
    command=['sudo','lsof','-nP']
    for line in mounts:
        target=line.split()[4]
        if re.fullmatch('/run/user/[0-9]+/doc',target):command+=['-e',target]
    result=subprocess.run([*command,'-Fpn'],capture_output=True,text=True)
    if result.returncode!=0 or result.stderr:raise ValueError('open-file inspection failed: '+result.stderr)
    busy=open_targets(result.stdout,{r['path'] for r in rows})
    if busy:raise ValueError('checkpoint still in use: '+str(busy))

