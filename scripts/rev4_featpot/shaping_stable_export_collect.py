"""Collect sealed scalar-export waves without replacing existing scientific evidence."""
import argparse,json,subprocess,shutil
from pathlib import Path
from shaping_completion import encoded,sha


def collect(wave,source,output):
    registration=wave/'REGISTRATION.json';q=json.loads(registration.read_text())
    if q['schema']!='fixed-qualified-supplemental-wave-v1':raise ValueError('registered scalar wave required')
    host=q['host']
    if host not in ('r3500','r3800x','r5600g','i134','i270','r7900x'):raise ValueError('ineligible collection worker')
    if output.exists():raise ValueError('preserve existing collection attempt')
    output.mkdir(parents=True)
    # This is metadata/hash validation, never a scoring or training replay.
    program='''import json,sys
from pathlib import Path
sys.path.insert(0,SOURCE+'/scripts/rev4_featpot')
sys.path.insert(0,str(Path(WAVE).parent/'python-runtime'))
from shaping_completion import sha
from shaping_rescue_selection import verified_candidate
q=json.loads((Path(WAVE)/'REGISTRATION.json').read_text());files={};directories=[]
for job in q['jobs']:
 run=Path(job['run']);study=run.parents[2];cell=job['cell']
 verified_candidate(study,cell,'research-relu-max-pairs-v1')
 for kind in ('reexports','reexport_max_pairs'):
  root=study/kind/cell['id'];directories.append(str(root))
  for p in sorted(root.rglob('*')):
   if p.is_symlink():raise ValueError('symlink in sealed scalar outputs')
   if p.is_file():files[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
if sum(x['bytes'] for x in files.values())>12*1024**3:raise ValueError('scalar collection exceeds twelve GiB')
print(json.dumps({'files':files,'directories':directories,'registration_sha256':sha(Path(WAVE)/'REGISTRATION.json')}))
'''
    owner=source/'scripts/rev4_featpot/shaping_reexport_supervision.py'
    if sha(owner)!=sha(Path(__file__).with_name(owner.name)):raise ValueError('collection owner differs')
    program='SOURCE='+repr(str(source))+'\nWAVE='+repr(str(wave))+'\n'+program
    with (output/'worker-inventory.log').open('x') as log:
        result=subprocess.run(['ssh',host,'python3 -B -'],input=program,text=True,stdout=subprocess.PIPE,stderr=log,check=True)
    manifest=json.loads(result.stdout)
    if manifest['registration_sha256']!=sha(registration):raise ValueError('worker wave differs')
    if shutil.disk_usage(output).free<sum(v['bytes'] for v in manifest['files'].values())+30*1024**3:
        raise ValueError('collection would breach the thirty GiB SSD reserve')
    (output/'MANIFEST.json').write_bytes(encoded(manifest))
    paths=output/'PATHS.txt';paths.write_bytes(b''.join(p.lstrip('/').encode()+b'\0' for p in manifest['files']))
    stage=output/'incoming';stage.mkdir()
    with (output/'transfer.log').open('x') as log:
        subprocess.run(['rsync','-rt','--omit-dir-times','--from0','--files-from='+str(paths),host+':/',str(stage)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
    for name,row in manifest['files'].items():
        path=stage/name.lstrip('/')
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('collected artifact drift: '+name)
    # Every directory is published atomically, with conversions before panels.
    # Existing directories require a separate reconciliation, never an overwrite.
    for name in manifest['directories']:
        if Path(name).exists():raise ValueError('preserve existing scalar output: '+name)
    for name in manifest['directories']:
        destination=Path(name);destination.parent.mkdir(parents=True,exist_ok=True)
        (stage/name.lstrip('/')).rename(destination)
    from shaping_rescue_selection import verified_candidate
    for job in q['jobs']:
        verified_candidate(Path(job['run']).parents[2],job['cell'],'research-relu-max-pairs-v1')
    receipt=dict(registration_sha256=sha(registration),manifest_sha256=sha(output/'MANIFEST.json'),
        models=len(q['jobs']),artifacts=len(manifest['files']),bytes=sum(v['bytes'] for v in manifest['files'].values()),
        all_transport_sha256_verified=True,all_candidate_provenance_verified=True,training_or_scoring_replayed=False)
    (output/'COLLECTION_VERIFIED.json').write_bytes(encoded(receipt))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('wave','source','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();collect(a.wave,a.source,a.output)
