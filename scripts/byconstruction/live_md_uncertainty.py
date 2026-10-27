"""Read only admitted LIVE-MD observer columns from its bounded MAT-v5 sources.

Format: https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf
This adapter supports the released little-endian real-double matrices only.
Opaque decompression is not a response read; numeric decoding is column-selected.
"""
import struct
import zlib
import numpy as np

LIMIT = 2*1024*1024


def elements(data):
    offset=0
    while offset<len(data):
        if offset+8>len(data):raise ValueError("truncated MAT tag")
        tag,size=struct.unpack_from("<II",data,offset)
        small=tag>>16
        if small:
            if small>4:raise ValueError("invalid small MAT element")
            yield tag&65535,data[offset+4:offset+4+small]
            offset+=8
        else:
            end=offset+8+size
            if end>len(data):raise ValueError("truncated MAT element")
            yield tag,data[offset+8:end]
            offset=end if tag==15 else end+(-size%8)
            if offset>len(data):raise ValueError("truncated MAT alignment")


def selected_columns(raw,name,columns,shape):
    """No float conversion of omitted columns or other numeric variables."""
    if (len(raw)>LIMIT or len(raw)<128 or raw[126:128]!=b"IM"
            or raw[124:126]!=b"\x00\x01"):
        raise ValueError("bounded little-endian MAT-v5 source required")
    if not columns or len(set(columns))!=len(columns) or any(type(c) is not int or not 0<=c<shape[1] for c in columns):
        raise ValueError("unique admitted native columns required")
    found=None
    for kind,body in elements(raw[128:]):
        if kind==15:
            decoder=zlib.decompressobj()
            unpacked=decoder.decompress(body,LIMIT+1)
            if len(unpacked)>LIMIT or decoder.unconsumed_tail or decoder.unused_data or not decoder.eof:
                raise ValueError("invalid or excessive MAT compression")
            unpacked_elements=list(elements(unpacked))
            if len(unpacked_elements)!=1:raise ValueError("compressed MAT matrix count differs")
            kind,body=unpacked_elements[0]
        if kind!=14:raise ValueError("native MAT matrix required")
        parts=list(elements(body))
        if len(parts)<3 or parts[2][0]!=1:raise ValueError("MAT name declaration differs")
        if parts[2][1].decode("ascii")!=name:continue
        if found is not None:raise ValueError("duplicate native matrix")
        if (len(parts)!=4 or parts[0][0]!=6 or len(parts[0][1])!=8
                or struct.unpack("<II",parts[0][1])!=(6,0)
                or parts[1][0]!=5 or len(parts[1][1])!=8
                or struct.unpack("<ii",parts[1][1])!=shape or parts[3][0]!=9):
            raise ValueError("native real-double matrix shape/type differs")
        payload=parts[3][1]
        if len(payload)!=shape[0]*shape[1]*8:raise ValueError("native matrix byte count differs")
        selected=[]
        for col in columns:
            start=col*shape[0]*8
            selected.append(np.frombuffer(payload[start:start+shape[0]*8],dtype="<f8").copy())
        found=np.stack(selected,axis=1)
        if not np.isfinite(found).all():raise ValueError("selected native observer response is nonfinite")
    if found is None:raise ValueError("required native response matrix absent")
    return found


def sureal(R, iters=500, tol=1e-10):
    """Owned zenpapers observer_models.py at 4e81c03a; unchanged fitting arithmetic."""
    M = ~np.isnan(R)
    X = np.where(M, R, 0.0)
    psi = np.nanmean(R, axis=1)
    delta = np.zeros(R.shape[1])
    v2 = np.ones(R.shape[1])
    for _ in range(iters):
        res = np.where(M, X - psi[:, None], 0.0)
        delta_new = res.sum(0) / np.maximum(M.sum(0), 1)
        e = np.where(M, X - psi[:, None] - delta_new[None, :], 0.0)
        v2_new = np.maximum((e * e).sum(0) / np.maximum(M.sum(0), 1), 1e-6)
        w = M / v2_new[None, :]
        psi_new = (np.where(M, X - delta_new[None, :], 0.0) * w).sum(1) / np.maximum(w.sum(1), 1e-12)
        done = np.max(np.abs(psi_new - psi)) < tol
        psi, delta, v2 = psi_new, delta_new, v2_new
        if done:
            break
    se = 1 / np.sqrt(np.maximum((M / v2[None, :]).sum(1), 1e-12))
    return psi, delta, np.sqrt(v2), se
