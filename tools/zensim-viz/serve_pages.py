#!/usr/bin/env python3
"""LAN development server with the single-byte-range transport used by the bundle."""
import argparse, functools, http.server, re
from pathlib import Path

class Handler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining=None
        value=self.headers.get('Range')
        path=Path(self.translate_path(self.path))
        if not value or not path.is_file():return super().send_head()
        match=re.fullmatch(r'bytes=(\d+)-(\d*)',value)
        if not match:self.send_error(416,'Only one explicit byte range is supported');return None
        size=path.stat().st_size;start=int(match[1]);end=int(match[2]) if match[2] else size-1
        if start>=size or end<start or end>=size:self.send_error(416,'Range outside resource');return None
        stream=path.open('rb');stream.seek(start);self.remaining=end-start+1
        self.send_response(206);self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Content-Length',str(self.remaining));self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}');self.end_headers();return stream
    def copyfile(self,source,outputfile):
        if self.remaining is None:return super().copyfile(source,outputfile)
        remaining=self.remaining
        while remaining:
            chunk=source.read(min(65536,remaining))
            if not chunk:raise IOError('file changed during range response')
            outputfile.write(chunk);remaining-=len(chunk)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--port',type=int,default=3311);a=p.parse_args()
    if not 3000<=a.port<=3999:raise SystemExit('port must be 3000–3999')
    http.server.ThreadingHTTPServer(('0.0.0.0',a.port),functools.partial(Handler,directory=a.directory)).serve_forever()
