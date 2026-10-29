#!/usr/bin/env python3
"""Independent AV1 header/decode oracle. External pixels never feed training."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from chroma_pilot import save,sha

def batch_view(receipt_path,archive,oracle_base,out,golden):
    """One independent decoder/header process per whole view, no re-encode."""
    out.mkdir(parents=True)
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(oracle_base/'ffmpeg-libs')
    receipt=json.loads(receipt_path.read_text());cells=receipt['cells']
    def cas(value):return archive/value[:2]/value[2:4]/value
    files=[]
    for cell in cells:
        path=cas(cell['encoded_sha256'])
        if sha(path)!=cell['encoded_sha256']:raise ValueError('independent oracle CAS differs')
        files.append(path)
    listing=out/'files.ffconcat'
    listing.write_text('ffconcat version 1.0\n'+''.join("file '"+str(p)+"'\n" for p in files))
    ffmpeg=str(oracle_base/'bin/ffmpeg')
    trace=out/'headers.log'
    with trace.open('x') as log:
        subprocess.run([ffmpeg,'-v','verbose','-threads','1','-f','concat','-safe','0','-i',str(listing),'-map','0:v:0','-c:v','copy','-bsf:v','trace_headers','-f','null','-'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    text=trace.read_text()
    def field(chunk,key):
        values=re.findall(r'\b'+key+r'(?:\.delta_q)?\s+[^\n]*?=\s*(-?\d+)',chunk)
        if values:return sorted(set(map(int,values)))
        coded=re.findall(r'\b'+key+r'\.delta_coded\s+[^\n]*?=\s*(-?\d+)',chunk)
        return [0] if coded and set(coded)=={'0'} else []
    for key,value in [('color_primaries',9),('transfer_characteristics',16),('matrix_coefficients',6),('color_range',1),('seq_profile',0),('mono_chrome',0),('high_bitdepth',1),('separate_uv_delta_q',0)]:
        actual=field(text,key)
        if actual!=[value]:raise ValueError(f'AV1 coding {key}={actual} expected{value}')
    info=json.loads(subprocess.check_output([str(oracle_base/'bin/ffprobe'),'-v','error','-show_streams','-of','json',str(files[0])],env=env,text=True))['streams'][0]
    for key,value in dict(codec_name='av1',pix_fmt='yuv420p10le',color_primaries='bt2020',color_transfer='smpte2084',color_space='smpte170m',color_range='pc').items():
        if info.get(key)!=value:raise ValueError('independent AVIF stream coding differs: '+key)
    chunks=re.split(r'(?=\bbase_q_idx\s)',text)[1:]
    if len(chunks)!=len(cells):raise ValueError(f'AV1 frame count differs: {len(chunks)} vs {len(cells)}')
    expected=json.loads(golden.read_text())['quality_qindices']
    results=[]
    for cell,chunk in zip(cells,chunks):
        header={key:field(chunk,key) for key in ['base_q_idx','delta_q_y_dc','delta_q_u_dc','delta_q_u_ac']}
        if header['base_q_idx']!=expected[str(cell['quality'])]:raise ValueError('luma quantizer differs from frozen A3')
        if header['delta_q_y_dc']!=[0]:raise ValueError('luma DC delta changed')
        for key in ['delta_q_u_dc','delta_q_u_ac']:
            if header[key]!=[cell['chroma_q'][0]]:raise ValueError('shared chroma qindex delta differs')
        results.append(dict(encoded_sha256=cell['encoded_sha256'],quality=cell['quality'],chroma_q=cell['chroma_q'],header=header))
    with (out/'dav1d.framemd5').open('x') as log,(out/'dav1d.stderr').open('x') as err:
        subprocess.run([ffmpeg,'-v','error','-xerror','-threads','1','-c:v','libdav1d','-f','concat','-safe','0','-i',str(listing),'-map','0:v:0','-threads','1','-pix_fmt','yuv420p10le','-fps_mode','passthrough','-f','framemd5','-'],env=env,stdout=log,stderr=err,check=True)
    frames=[line for line in (out/'dav1d.framemd5').read_text().splitlines() if line and not line.startswith('#')]
    if len(frames)!=len(cells):raise ValueError('independent dav1d decode lost/added frames')
    result=dict(passed=True,cells=len(cells),receipt_sha256=sha(receipt_path),ffmpeg_sha256=sha(oracle_base/'bin/ffmpeg'),headers_sha256=sha(trace),framemd5_sha256=sha(out/'dav1d.framemd5'),records=results,oracle='FFmpeg/libdav1d 8.0.1-3ubuntu2; decode/inspect only')
    save(out/'ORACLE.json',result);print(json.dumps(dict(passed=True,cells=len(cells))),flush=True)

if '--view-receipt' in sys.argv:
    p=argparse.ArgumentParser();p.add_argument('--view-receipt',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--oracle-base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--golden',type=Path,required=True)
    a=p.parse_args();batch_view(a.view_receipt,a.archive,a.oracle_base,a.out,a.golden);sys.exit(0)

parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--quality-grid',action='store_true');parser.add_argument('--oracle-base',type=Path);args=parser.parse_args()
root=args.root;base=args.oracle_base or root.parent;env=os.environ.copy();env['LD_LIBRARY_PATH']=str(base/'ffmpeg-libs')
records=json.loads((root/'ENCODES.json').read_text());tested=[]
for r in records:
    p=Path(r['encoded_path'])
    if sha(p)!=r['encode_sha256']:raise ValueError('oracle source bytes changed')
    name=p.parent.name+'-'+p.stem
    info=json.loads(subprocess.check_output([str(base/'bin/ffprobe'),'-v','error','-show_streams','-of','json',str(p)],env=env,text=True))['streams'][0]
    for k,v in dict(codec_name='av1',pix_fmt='yuv420p10le',color_primaries='bt2020',color_transfer='smpte2084',color_space='smpte170m',color_range='pc').items():
        if info.get(k)!=v:raise ValueError(f'{p}: {k}={info.get(k)!r} != {v!r}')
    trace=root/'logs'/(name+'-headers.log')
    if not trace.exists():
        with trace.open('w') as f:subprocess.run([str(base/'bin/ffmpeg'),'-v','verbose','-threads','1','-i',str(p),'-c','copy','-bsf:v','trace_headers','-f','null','-'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    text=trace.read_text();fields={}
    for key in ['base_q_idx','delta_q_y_dc','delta_q_u_dc','delta_q_u_ac','delta_q_v_dc','delta_q_v_ac','separate_uv_delta_q']:
        values=re.findall(r'\b'+key+r'(?:\.delta_q)?\s+[^\n]*?=\s*(-?\d+)',text)
        if values:fields[key]=sorted(set(map(int,values)))
        else:
            coded=re.findall(r'\b'+key+r'\.delta_coded\s+[^\n]*?=\s*(-?\d+)',text)
            if coded and set(coded)=={'0'}:fields[key]=[0]
    decoded=root/'logs'/(name+'-dav1d.framemd5')
    if not decoded.exists():
        with decoded.open('w') as f:subprocess.run([str(base/'bin/ffmpeg'),'-v','error','-threads','1','-i',str(p),'-threads','1','-pix_fmt','yuv420p10le','-f','framemd5','-'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    if fields.get('separate_uv_delta_q')!=[0]:raise ValueError('shared UV signaling changed')
    fields['delta_q_v_dc']=fields['delta_q_u_dc'];fields['delta_q_v_ac']=fields['delta_q_u_ac']
    tested.append(dict(encoded_path=str(p),quality=r['quality'],chroma_q=r['chroma_q'],header=fields,stream={k:info.get(k) for k in ['width','height','pix_fmt','color_primaries','color_transfer','color_space','color_range']}))
base_q={tuple(r['header'].get('base_q_idx',[])) for r in tested}
if not args.quality_grid and (len(base_q)!=1 or not next(iter(base_q))):raise ValueError('luma base qindex is missing or changes')
byquality={}
for r in tested:byquality.setdefault(r['quality'],set()).add(tuple(r['header'].get('base_q_idx',[])))
for q,values in byquality.items():
    if len(values)!=1 or not next(iter(values)):raise ValueError('luma qindex varies at fixed quality')
quality_indices={int(q):list(next(iter(values))) for q,values in byquality.items()}
if args.quality_grid:
    assert len(quality_indices)==20
    ladder=[quality_indices[q][0] for q in sorted(quality_indices,reverse=True)]
    assert all(b>a for a,b in zip(ladder,ladder[1:])), 'positive control qindices must strictly increase'
for r in tested:
    delta=r['chroma_q'][0]
    # trace_headers names signed AV1 fields delta_q_u_{dc,ac} and V likewise.
    for field in ['delta_q_u_dc','delta_q_u_ac','delta_q_v_dc','delta_q_v_ac']:
        if r['header'].get(field)!=[delta]:raise ValueError(f'chroma override absent/different: {r}')
save(root/'ORACLE.json',dict(passed=True,cells=len(tested),oracle='FFmpeg/libdav1d 8.0.1-3ubuntu2',base_q_idx=sorted(list(q) for q in base_q),quality_qindices=quality_indices,checks=['all streams independently decode','10-bit PQ BT2020 full-range 420','BT601 matrix signaled, matching pinned adapter','luma base qindex fixed within each quality; positive-control indices strictly increase' if args.quality_grid else 'luma base qindex fixed','U/V chroma DC/AC deltas match requested override, shared UV flag=0'],records=tested))
print(json.dumps(dict(cells=len(tested),quality_qindices=quality_indices)),flush=True)
