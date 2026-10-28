import json,shutil
from pathlib import Path
src=Path('/var/tmp/byc-chroma-20261028/b/CANDIDATES.json');d=json.loads(src.read_text());root=Path('/mnt/tower/output/zensim/byc-chroma-20261028/b');(root/'raw').mkdir(parents=True,exist_ok=True)
count=0
for s in d['sources']:
    original=Path(s['source_path'])
    if s['type']=='imazen26':
        mirror=Path('/mnt/tower/output/imazen-26-png-v3')/s['canonical_relative']
        if mirror.is_file():s['worker_raw']=str(mirror);continue
    dest=root/'raw'/(s['id']+'.png')
    if not dest.exists():shutil.copyfile(original,dest);count+=1
    s['worker_raw']=str(dest)
(root/'CANDIDATES.json').write_text(json.dumps(d,indent=2)+'\n')
print('staged',count,'sources',len(d['sources']))
