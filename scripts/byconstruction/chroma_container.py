#!/usr/bin/env python3
"""Independent major-0 framing inspection; EVIF remains the seal/decode owner."""
import json
import struct

def chunks(data):
    if data[:12]!=b'\x8bEVIF\r\n\x1a\x00\x01\x00\x00':raise ValueError('unexpected EVIF framing')
    offset=12;out={}
    while offset<len(data):
        if offset+8>len(data):raise ValueError('truncated chunk header')
        tag=data[offset:offset+4].decode('ascii');size=struct.unpack_from('<I',data,offset+4)[0];offset+=8
        if tag in out or offset+size>len(data):raise ValueError('duplicate/truncated EVIF chunk')
        out[tag]=data[offset:offset+size];offset+=size
        if tag=='SEAL' and offset!=len(data):raise ValueError('bytes after seal')
    if not {'HEAD','BITS','PNGF','DIGS','SEAL'}<=out.keys():raise ValueError('missing EVIF media')
    if out['BITS'][:4]!=b'jpeg':raise ValueError('unexpected bitstream codec')
    return out

def scores(data):return json.loads(chunks(data)['scor'])['scores']
