"""Transport immutable terminal analysis/control directories with atomic import."""
import io,json,tarfile
from pathlib import Path
from shaping_completion import encoded,sha


def pack(request,bundle,validate):
    from shaping_rescue_worker import verify
    from shaping_worker_collect import inventory
    from shaping_storage import reservation
    record=json.loads(request.read_text());verify(record);source=Path(record['result_output'])
    validate(record,source);files=inventory(source);size=sum(row['bytes'] for row in files.values())
    if len(files)>1000 or size>1024**3-16*1024**2:raise ValueError('terminal result exceeds bounded transport allowance')
    receipt=dict(files=files,request_sha256=sha(request),cell=record['cell'],pack_driver_sha256=sha(__file__))
    bundle.parent.mkdir(parents=True,exist_ok=True)
    with reservation(Path(record['study']),bundle,30*1024**3,bytes_required=size+16*1024**2):
        with tarfile.open(bundle,'x') as archive:
            data=encoded(receipt);info=tarfile.TarInfo('COLLECTION.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
            for name in files:archive.add(source/name,arcname='payload/'+name,recursive=False)
        if inventory(source)!=files:raise ValueError('terminal result changed while packing')
        receipt['bundle_sha256']=sha(bundle)
        with Path(str(bundle)+'.json').open('xb') as stream:stream.write(encoded(receipt))
    return receipt


def collect(request,bundle,pin,reproduce,validate):
    from shaping_worker_collect import unpack
    from shaping_storage import reservation
    record=json.loads(request.read_text());resolution,_=reproduce(record,host=record['host'])
    if resolution!=record['resolution']:raise ValueError('coordinator cannot reproduce terminal admission')
    result=Path(record['result_output']);study=Path(record['study']);staging=study/'worker_collection'/('terminal-'+pin)
    with reservation(study,staging,30*1024**3):
        receipt=unpack(bundle,staging,pin)
        if receipt['request_sha256']!=sha(request) or receipt['cell']!=record['cell']:
            raise ValueError('terminal result request differs')
        validate(record,staging)
        (staging/'WORKER_COLLECTION.json').write_bytes(encoded(dict(request_sha256=sha(request),bundle_sha256=pin,
            source_host=record['host'],pack_driver_sha256=receipt['pack_driver_sha256'],collect_driver_sha256=sha(__file__))))
        staging.rename(result)
    return dict(result=str(result),files=len(receipt['files']),bytes=sum(row['bytes'] for row in receipt['files'].values()))
