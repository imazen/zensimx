"""Compress an immutable transport capsule with a complete raw-byte roundtrip."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def compress(source,output):
    output.parent.mkdir(parents=True,exist_ok=True);receipt=Path(str(output)+'.json')
    if output.exists() or receipt.exists():raise ValueError('preserve existing capsule or receipt')
    if source.is_symlink() or not source.is_file():raise ValueError('regular immutable source required')
    if shutil.disk_usage(output.parent).free<30*1024**3+source.stat().st_size+16*1024**2:
        raise ValueError('compressed capsule would threaten SSD reserve')
    pin=sha(source);size=source.stat().st_size
    subprocess.run(['zstd','-q','-T1','-3','-o',str(output),str(source)],check=True)
    process=subprocess.Popen(['zstd','-q','-d','-c',str(output)],stdout=subprocess.PIPE)
    decoded=hashlib.file_digest(process.stdout,'sha256').hexdigest();process.stdout.close()
    if process.wait()!=0 or decoded!=pin or sha(source)!=pin:raise ValueError('lossless capsule roundtrip differs; preserve evidence')
    record=dict(schema='lossless-transport-capsule-v1',source=str(source),raw_bytes=size,raw_sha256=pin,
        output=str(output),stored_bytes=output.stat().st_size,stored_sha256=sha(output),
        encoding='zstd-level3-single-thread',all_decompressed_bytes_identical=True)
    with receipt.open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(compress(a.source,a.output)),flush=True)
