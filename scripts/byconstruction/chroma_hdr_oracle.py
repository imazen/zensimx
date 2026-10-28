#!/usr/bin/env python3
"""Independent AV1 header/decode oracle. External pixels never feed training."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from chroma_pilot import save,sha
root=Path(sys.argv[1]);base=root.parent;env=os.environ.copy();env['LD_LIBRARY_PATH']=str(base/'ffmpeg-libs')
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
    tested.append(dict(encoded_path=str(p),chroma_q=r['chroma_q'],header=fields,stream={k:info.get(k) for k in ['width','height','pix_fmt','color_primaries','color_transfer','color_space','color_range']}))
base_q={tuple(r['header'].get('base_q_idx',[])) for r in tested}
if len(base_q)!=1 or not next(iter(base_q)):raise ValueError('luma base qindex is missing or changes')
for r in tested:
    delta=r['chroma_q'][0]
    # trace_headers names signed AV1 fields delta_q_u_{dc,ac} and V likewise.
    for field in ['delta_q_u_dc','delta_q_u_ac','delta_q_v_dc','delta_q_v_ac']:
        if r['header'].get(field)!=[delta]:raise ValueError(f'chroma override absent/different: {r}')
save(root/'ORACLE.json',dict(passed=True,cells=len(tested),oracle='FFmpeg/libdav1d 8.0.1-3ubuntu2',base_q_idx=list(next(iter(base_q))),checks=['all streams independently decode','10-bit PQ BT2020 full-range 420','BT601 matrix signaled, matching pinned adapter','luma base qindex fixed','U/V chroma DC/AC deltas match requested override, shared UV flag=0'],records=tested))
print(json.dumps(dict(cells=len(tested),base_q_idx=list(next(iter(base_q))))),flush=True)
