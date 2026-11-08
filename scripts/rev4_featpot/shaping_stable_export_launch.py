"""Transfer, fully verify, then start each fixed scalar-export wave exactly once."""
import argparse,json,shlex,subprocess
from pathlib import Path
from shaping_completion import encoded,sha


def launch(wave):
    q=json.loads((wave/'REGISTRATION.json').read_text());host=q['host'];source=Path(q['source']);root=wave.parent
    if q['schema']!='fixed-qualified-supplemental-wave-v1':raise ValueError('registered scalar wave required')
    from shaping_reduced_scope import admit_compute
    study=root/'study';admit_compute(study,host=host)
    owner=source/'scripts/rev4_featpot/shaping_reexport_supervision.py'
    if sha(owner)!=q['source_owner_sha256']:raise ValueError('registered source differs')
    output=wave/'launch';output.mkdir(exist_ok=False)
    verify='''import json,hashlib
from pathlib import Path
p=Path(REGISTRATION);q=json.loads(p.read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==PIN
assert hashlib.sha256(Path(q['source']+'/scripts/rev4_featpot/shaping_reexport_supervision.py').read_bytes()).hexdigest()==q['source_owner_sha256']
for name,row in q['files'].items():
 p=Path(name)
 if ALLOW_MISSING and not p.exists():continue
 assert not p.is_symlink() and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],name
print('VERIFIED',len(q['files']),flush=True)
'''
    # First install the new registration only; preflight refuses divergent
    # pre-existing immutable inputs before rsync can write any of those inputs.
    with (output/'transfer.log').open('x') as log:
        subprocess.run(['ssh',host,shlex.join(['mkdir','-p',str(wave)])],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['rsync','-rt','--omit-dir-times',str(wave/'REGISTRATION.json'),host+':'+str(wave/'REGISTRATION.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        def check(allow_missing):
            program='REGISTRATION='+repr(str(wave/'REGISTRATION.json'))+'\nPIN='+repr(sha(wave/'REGISTRATION.json'))+'\nALLOW_MISSING='+repr(allow_missing)+'\n'+verify
            subprocess.run(['ssh',host,'python3 -B -'],input=program,text=True,stdout=log,stderr=subprocess.STDOUT,check=True)
        check(True)
        subprocess.run(['rsync','-rt','--omit-dir-times','--from0','--files-from='+str(wave/'INPUT_PATHS.txt'),'/',host+':/'],stdout=log,stderr=subprocess.STDOUT,check=True)
        check(False)
    (output/'INPUT_VERIFIED.json').write_bytes(encoded(dict(registration_sha256=sha(wave/'REGISTRATION.json'),all_input_sha256_verified=True)))
    records=[]
    for family in q['families']:
        jobs=[j for j in q['jobs'] if j['cell']['family']==family]
        if not jobs:continue
        directory=output/family;directory.mkdir();unit='zensim-'+wave.name+'-'+family.replace('_','-')
        command=['env','TMPDIR=/home/lilith/tmp','PYTHONPATH='+str(root/'python-runtime'),
            '/home/lilith/work/claudehints/scripts/run-heavy','--mem','1G','--jobs','1','--','python3','-B',str(owner),
            '--study',str(study),'--binary-dir',str(root/'bounded-resume-runtime-20261108/binaries-v3'),
            '--inspector',str(root/'first-weight-native-qualification/zenpredict-inspect'),
            '--family',family,'--status',str(directory/'STATUS.json'),'--reserve-slot']
        for job in jobs:command.extend(['--cell',job['cell']['id']])
        subprocess.run(['ssh',host,shlex.join(['mkdir','-p',str(directory)])],check=True)
        row=dict(host=host,unit=unit,models=len(jobs),command=command,training_replayed=False)
        (directory/'LAUNCH.json').write_bytes(encoded(row));records.append(row)
        subprocess.run(['ssh',host,shlex.join(['systemd-run','--user','--unit',unit,
            '--property=StandardOutput=append:'+str(directory/'worker.log'),
            '--property=StandardError=append:'+str(directory/'worker.log'),*command])],check=True)
        print(family,len(jobs),'started',flush=True)
    (output/'LAUNCH.json').write_bytes(encoded(dict(registration_sha256=sha(wave/'REGISTRATION.json'),rows=records)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wave',type=Path,required=True)
    launch(p.parse_args().wave)
