"""Pin public policy text at study paths before freezing portable outer decisions."""
import argparse,json
from pathlib import Path
from shaping_completion import REPO,encoded,sha

FILES=('SHAPING_UNDEFINED_SETTING_POLICY.md',)


def policy_path(root,local):
    manifest=root/'policies/POLICIES.json'
    if not manifest.exists():return local
    record=json.loads(manifest.read_text())
    if (record['schema']!='shaping-public-policy-snapshot-v1' or record['outer_results_read'] is not False
        or record['plan_sha256']!=sha(root/'PLAN.json') or set(record['files'])!=set(FILES)):
        raise ValueError('frozen policy snapshot identity differs')
    if local.name not in FILES:raise ValueError('unregistered public policy')
    path=manifest.parent/local.name;pin=record['files'][local.name]
    if any(p.is_symlink() for p in [path,*path.parents]) or sha(path)!=pin or sha(local)!=pin:
        raise ValueError('frozen policy snapshot drift')
    return path


def freeze(root):
    directory=root/'policies';manifest=directory/'POLICIES.json'
    if manifest.exists():
        for name in FILES:policy_path(root,REPO/'docs'/name)
        return json.loads(manifest.read_text())
    if directory.exists():raise ValueError('preserve partial policy snapshot')
    if (list((root/'reduced_scope').glob('outer-*/SHORTLIST.json'))
        or list((root/'selection').glob('outer-*/OUTER_ADMISSION.json'))):
        raise ValueError('freeze policies before registering outer shortlists or admissions')
    files={name:sha(REPO/'docs'/name) for name in FILES}
    record=dict(schema='shaping-public-policy-snapshot-v1',plan_sha256=sha(root/'PLAN.json'),
                files=files,source_root=str(REPO),outer_results_read=False)
    directory.mkdir()
    for name in FILES:
        (directory/name).write_bytes((REPO/'docs'/name).read_bytes())
        if sha(directory/name)!=files[name]:raise ValueError('policy changed while snapshotting')
    with manifest.open('xb') as stream:stream.write(encoded(record))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    print(json.dumps(freeze(parser.parse_args().study)),flush=True)
