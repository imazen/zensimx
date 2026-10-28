#!/usr/bin/env python3
"""Metadata-only full TRAIN source census; canonical bytes/geometry checked later.

No labels or new feature inputs are read. CID22 fitting/development/calibration
roles remain attached even though all belong to the oracle TRAIN population.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse
import pyarrow.parquet as pq
from chroma_sources import group
from chroma_pilot import save, sha

def census(out):
    mlic=Path('/var/tmp/byc-mixes-20261025/Mlic/root/wide/main/real')
    bank=Path('/var/tmp/rev4-featbank/bank')
    canonical=Path('/home/lilith/work/zen/imazen-26-png-v3')
    pins={};sources=[];exclusions=[]
    def table(p,columns):
        pins[str(p)]=sha(p);return pq.read_table(p,columns=columns).to_pylist()
    roles={}
    for leg,rr in [('safesyn',['fit','dev']),('cid22',['fit','dev'])]:
        roles[leg]={r:set(x['ref_basename'] for x in table(mlic/f'{leg}_{r}.keys.parquet',['ref_basename'])) for r in rr}
        sets=list(roles[leg].values())
        for i,a in enumerate(sets):
            for b in sets[i+1:]:
                assert not a&b,'Mlic source role crosses partitions'
    admission_path=Path('/var/tmp/zensim-validation-2026-09-15/recovery/ADMISSION.json')
    pins[str(admission_path)]=sha(admission_path)
    assignments=json.loads(admission_path.read_text())['assignments']
    roles['cid22']['calibration']={ref for ref,role in assignments.items() if ref.startswith('cid22:') and role=='calibration'}
    # All geometry siblings inherit their base family's Mlic role.
    base=lambda s:s.split(':')[-1].split('_')[0]
    assert not {base(x) for x in roles['safesyn']['fit']}&{base(x) for x in roles['safesyn']['dev']}
    for leg,bankleg in [('safesyn','safesyn'),('cid22','cid22_train')]:
        manifest=bank/bankleg/'_MANIFEST.json'
        pins[str(manifest)]=sha(manifest)
        assert json.loads(manifest.read_text())['role']=='train'
        rows=table(bank/bankleg/'keys.parquet',['ref_group','ref_path','ref_pixels_sha256'])
        refs={}
        for r in rows:
            prior=refs.setdefault(r['ref_group'],r)
            assert prior==r,'bank source identity differs within a family'
        for ref,r in sorted(refs.items()):
            if leg=='safesyn' and (ref not in roles[leg]['fit'] or not ref.endswith('_512sq')):continue
            inherited=[role for role,items in roles[leg].items() if ref in items]
            assert len(inherited)==1,'oracle TRAIN source lacks unique Mlic role'
            path=Path(r['ref_path']);identity=('safe-'+base(ref)) if leg=='safesyn' else 'cid-'+ref.split(':')[1]
            if not path.is_file():
                exclusions.append(dict(id=identity,reference_group=ref,reason='registered available-source census: source file missing',source_path=str(path)));continue
            sources.append(dict(id=identity,type=leg,content_class=leg+'-train',class_group=leg+'-unclassified',reference_group=ref,source_path=str(path),source_pixels_sha256=r['ref_pixels_sha256'],split='train',mlic_role=inherited[0]))
    reps=canonical/'manifests/imazen26_representatives_K500_even_2026-06-18.tsv'
    sm=canonical/'manifests/split_map.tsv'
    for p in [reps,sm]:pins[str(p)]=sha(p)
    split={r['id']:r for r in csv.DictReader(sm.open(),delimiter='\t')}
    commit=subprocess.check_output(['git','--git-dir='+str(canonical/'.git'),'rev-parse','HEAD'],text=True).strip()
    representative={}
    for r in csv.DictReader(reps.open(),delimiter='\t'):
        rel=unquote(urlparse(r['url']).path).split('/imazen-26-png-v3/')[-1];identity=Path(rel).name[:4]
        if split[identity]['split']!='train':continue
        assert int(identity)%10 in [0,2,4,6,8]
        assert split[identity]['content_class']==r['content_class']
        representative.setdefault(identity,dict(row=r,relative=rel))
    substitution=json.loads(Path('/mnt/tower/output/zensim/byc-chroma-20261028/a2/SUBSTITUTION.json').read_text())
    # Preserve this accepted nearest-feature substitute, never claim cluster membership.
    for identity,r in sorted(representative.items()):
        old=identity
        if identity=='2402':
            identity='2406'
            # Resolve canonical 2406 path directly, independent of crop membership.
            candidates=subprocess.check_output(['git','--git-dir='+str(canonical/'.git'),'ls-tree','-r','--name-only',commit,'png-v3/2400-textures'],text=True).splitlines()
            if not candidates:
                candidates=subprocess.check_output(['git','--git-dir='+str(canonical/'.git'),'ls-tree','-r','--name-only',commit,'png-v3'],text=True).splitlines()
            names=[p for p in candidates if Path(p).name.startswith('2406_') and p.endswith('.sdr.png')]
            assert len(names)==1
            relative=names[0].removeprefix('png-v3/')
        else:relative=r['relative']
        pointer=subprocess.check_output(['git','--git-dir='+str(canonical/'.git'),'show',commit+':png-v3/'+relative],text=True)
        assert pointer.startswith('version https://git-lfs.github.com/spec/v1')
        expected=pointer.split('oid sha256:')[1].split()[0]
        local=canonical/'.git/lfs/objects'/expected[:2]/expected[2:4]/expected
        if not local.is_file():local=Path('/mnt/tower/output/imazen-26-png-v3')/relative
        sources.append(dict(id='im'+identity,type='imazen26',content_class=split[identity]['content_class'],class_group=group(split[identity]['content_class']),reference_group='imazen26:'+identity,source_path=str(local) if local.is_file() else None,url='https://codec-corpus.r2.imazen.org/imazen-26-png-v3/'+relative,source_sha256=expected,split='train',mlic_role='new-train',canonical_relative=relative,representative=r['row'],substitutes='im2402' if old!=identity else None))
    # One canonical origin per source family; retain all representative row provenance.
    unique={};aliases=[]
    for s in sources:
        if s['reference_group'] in unique:aliases.append(dict(alias=s,canonical=unique[s['reference_group']]['id']))
        else:unique[s['reference_group']]=s
    save(out,dict(schema='chroma-stage-b-candidates-v1',sources=list(unique.values()),candidate_count=len(sources),source_count=len(unique),exclusions=exclusions,aliases=aliases,pins=pins,canonical_imazen_commit=commit,substitution=substitution,imazen_train_representative_origins=len(representative)))
    print(json.dumps(dict(sources=len(unique),candidate_count=len(sources),exclusions=len(exclusions),types={t:sum(s['type']==t for s in unique.values()) for t in ['safesyn','cid22','imazen26']})))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();census(a.out)
