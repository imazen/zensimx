#!/usr/bin/env python3
"""Close the input lineage of a trained model over its hash-bound records.

The study drivers (byconstruction, aic_matched_continuation, ring_root, ...) bind their inputs by sha256 inside
JSON records: PLAN.json, EVIDENCE_INVENTORY.json, frozen.json, receipts, manifests. This tool starts from one or
more of those records, follows every (path, sha256) reference it finds, verifies each file against its recorded
hash, and recurses into referenced JSON. The result is the transitive closure of what a model was built from,
plus the gaps: references with no hash, files that are missing or changed, and hashes with no path.

  lineage.py close OUT START.json [START.json ...]

Writes OUT/nodes.tsv (one row per file: path, recorded sha256, bytes, status, depth, first referrer, key),
OUT/gaps.tsv (unhashed path references, bare hashes, missing or changed files) and OUT/SUMMARY.json.
Only JSON records are recursed into; other files are hashed and listed. Hashes are cached in OUT/hash_cache.tsv
keyed on (path, size, mtime_ns) so reruns are cheap.
"""
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HEX64 = re.compile(r'^[0-9a-f]{64}$')
SHA_KEYS = ('sha256', 'frozen_sha256', 'file_sha256', 'checkpoint_sha256', 'packed_sha256')


def sha256_file(path, cache):
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    cache[key] = h.hexdigest()
    return cache[key]


BASE_KEYS = ('directory', 'dir', 'root', 'source_dir', 'run', 'study')


def resolve_relative(rel, bases):
    """The first base under which a relative reference exists; None when none does."""
    for b in bases:
        if (b / rel).exists():
            return b / rel
    return None


def looks_like_path(s):
    return isinstance(s, str) and s.startswith('/') and '\n' not in s and len(s) < 4096


def references(doc, base):
    """Yield (kind, path_or_None, sha_or_None, key) for every reference in a JSON document."""
    def walk(x, key, bases=()):
        if isinstance(x, dict):
            bases = tuple(Path(x[k]) for k in BASE_KEYS if looks_like_path(x.get(k))) + bases
            path = x.get('path')
            sha = next((x[k] for k in SHA_KEYS if isinstance(x.get(k), str) and HEX64.match(x[k])), None)
            if looks_like_path(path):
                yield ('pair' if sha else 'path', path, sha, key + '.path')
            for k, v in x.items():
                sub = f'{key}.{k}' if key else k
                if isinstance(v, str) and HEX64.match(v):
                    if looks_like_path(k):
                        yield ('pair', k, v, sub)
                    elif '/' in k and not k.startswith('/') and base is not None:
                        found = resolve_relative(k, bases + tuple([base, *list(base.parents)[:3]]))
                        yield ('pair', str(found), v, sub) if found else ('relative', k, v, sub)
                    elif k not in SHA_KEYS or not looks_like_path(path):
                        yield ('bare', None, v, sub)
                elif k != 'path':
                    yield from walk(v, sub, bases)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from walk(v, f'{key}[{i}]', bases)
        elif looks_like_path(x):
            yield ('path', x, None, key)
    yield from walk(doc, '')


def close(out, starts):
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    cache_file = out / 'hash_cache.tsv'
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            p, size, mtime, sha = line.split('\t')
            cache[(p, int(size), int(mtime))] = sha
    nodes = {}
    gaps = []
    queue = collections.deque()
    for s in starts:
        p = Path(s).resolve()
        queue.append((p, sha256_file(p, cache), 0, '(start)', ''))
    while queue:
        path, claimed, depth, referrer, key = queue.popleft()
        spath = str(path)
        if spath in nodes:
            if claimed and nodes[spath]['sha256'] and claimed != nodes[spath]['sha256'] and nodes[spath]['status'] == 'ok':
                gaps.append(('conflicting-claims', spath, claimed, referrer, key))
            continue
        status, size, actual = 'ok', '', ''
        if not path.exists():
            status = 'missing'
        elif path.is_dir():
            status = 'directory'
        else:
            size = path.stat().st_size
            actual = sha256_file(path, cache)
            if claimed and actual != claimed:
                status = 'changed'
        nodes[spath] = dict(sha256=claimed or actual, actual=actual, bytes=size, status=status, depth=depth,
                            referrer=referrer, key=key)
        if status in ('missing', 'changed'):
            gaps.append((status, spath, claimed, referrer, key))
        if status != 'ok' or not spath.endswith('.json') or (size and size > 64 << 20):
            continue
        try:
            doc = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        for kind, ref, sha, k in references(doc, path.parent):
            if kind == 'bare':
                gaps.append(('bare-hash', '', sha, spath, k))
            elif kind == 'relative':
                gaps.append(('unresolved-relative', ref, sha, spath, k))
            elif kind == 'path':
                rp = Path(ref)
                if str(rp) in nodes:
                    continue
                if rp.is_dir() or not rp.exists():
                    gaps.append(('unhashed-dir' if rp.is_dir() else 'unhashed-missing', ref, '', spath, k))
                    nodes.setdefault(ref, dict(sha256='', actual='', bytes='', status='unhashed', depth=depth + 1,
                                               referrer=spath, key=k))
                else:
                    gaps.append(('unhashed-file', ref, '', spath, k))
                    queue.append((rp, None, depth + 1, spath, k))
            else:
                queue.append((Path(ref), sha, depth + 1, spath, k))
    with open(cache_file, 'w') as f:
        for (p, size, mtime), sha in cache.items():
            f.write(f'{p}\t{size}\t{mtime}\t{sha}\n')
    with open(out / 'nodes.tsv', 'w') as f:
        f.write('path\tsha256\tbytes\tstatus\tdepth\treferrer\tkey\n')
        for p, n in sorted(nodes.items()):
            f.write(f"{p}\t{n['sha256']}\t{n['bytes']}\t{n['status']}\t{n['depth']}\t{n['referrer']}\t{n['key']}\n")
    with open(out / 'gaps.tsv', 'w') as f:
        f.write('kind\tpath\tsha256\treferrer\tkey\n')
        for g in gaps:
            f.write('\t'.join(map(str, g)) + '\n')
    by_status = collections.Counter(n['status'] for n in nodes.values())
    by_gap = collections.Counter(g[0] for g in gaps)
    roots = collections.Counter()
    root_bytes = collections.Counter()
    for p, n in nodes.items():
        top = '/'.join(p.split('/')[:4])
        roots[top] += 1
        root_bytes[top] += n['bytes'] or 0
    summary = dict(starts=[str(Path(s).resolve()) for s in starts], files=len(nodes), by_status=by_status,
                   gaps=by_gap, bytes=sum(n['bytes'] or 0 for n in nodes.values()),
                   top_dirs={k: dict(files=roots[k], bytes=root_bytes[k]) for k in sorted(roots)})
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


def main(argv):
    if len(argv) < 3 or argv[0] != 'close':
        sys.exit(__doc__)
    close(Path(argv[1]), argv[2:])


if __name__ == '__main__':
    main(sys.argv[1:])
