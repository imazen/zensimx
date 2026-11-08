"""Add a bounded, deduplicated native replay closure to numerical qualification."""
import argparse,json,shlex,shutil
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_stable_pairs import families,NAME

MAX_BYTES=1024**3


def prediction_command(log,root):
    line=log.read_text().splitlines()[0]
    if not line.startswith('$ '):raise ValueError('native prediction command log required')
    command=shlex.split(line[2:])
    if len(command)!=9 or command[1:3]!=['predict','--bake'] or command[4]!='--corpus' or command[6:8]!=['--score-units','--out']:
        raise ValueError('unexpected native prediction command')
    if any(Path(command[i]).resolve()!=Path(command[i]) or not Path(command[i]).is_relative_to(root) for i in (3,8)):
        raise ValueError('prediction bake and output must be in qualification root')
    return command


def prepare(study,root):
    families(study)
    if (root/'MIRROR_VERIFIED.json').exists() or (root/'REPLAY_CLOSURE.json').exists():
        raise ValueError('preserve existing qualification archive or replay closure')
    cases=[*json.loads((root/'CASES.json').read_text()),*json.loads((root/'ADDITIONAL_CASES.json').read_text())]
    if len(cases)!=8:raise ValueError('eight qualified native cases required')
    commands=[];copies={}
    def include(original,category,suffix):
        original=Path(original)
        if not original.is_file() or original.is_symlink():raise ValueError('canonical replay input required')
        pin=sha(original);relative=Path(category)/(pin+suffix)
        previous=copies.setdefault(str(relative),dict(original=str(original),sha256=pin,bytes=original.stat().st_size))
        if previous['sha256']!=pin:raise ValueError('replay input collision')
        return str(relative)
    stable=study.parent/'bounded-resume-runtime-20261108/binaries-v3/bake_dial_refit'
    stable_tool=include(stable,'bin','')
    script=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")"','mkdir replay','mkdir replay/scratch',
            'export TMPDIR="$PWD/replay/scratch" RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1']
    for case in cases:
        cid=case['cell'];directory=root/cid;original=include(case['original'],'original-models','.bin')
        repaired='replay/'+cid+'/repaired-from-original.bin';script.append(shlex.join(['mkdir','-p','replay/'+cid]))
        script.append(shlex.join([stable_tool,'repair-research-max-pairs','--in',original,'--out',repaired,'--pairs',cid+'/pairs.txt']))
        script.append(shlex.join(['cmp',cid+'/repaired.bin',repaired]))
        logs=[directory/'probes.log',*sorted((directory/'comparison').glob('*/*.log'))]
        logs=[p for p in logs if p.name in ('probes.log','original-runtime.log','unmarked.log','repaired.log')]
        if len(logs)!=31:raise ValueError('probe and all thirty population prediction commands required')
        for log in logs:
            command=prediction_command(log,root);binary=include(command[0],'bin','');table=include(command[5],'inputs','.parquet')
            source_output=Path(command[8]);relative_output=source_output.relative_to(root);bake=Path(command[3]).relative_to(root)
            replay_output=Path('replay')/relative_output;script.append(shlex.join(['mkdir','-p',str(replay_output.parent)]))
            replay=[binary,'predict','--bake',str(bake),'--corpus',table,'--score-units','--out',str(replay_output)]
            script.extend([shlex.join(replay),shlex.join(['cmp',str(relative_output),str(replay_output)])])
            commands.append(dict(log=str(log.relative_to(root)),input_sha256=sha(command[5]),
                                 native_command=replay,expected_output=str(relative_output),expected_sha256=sha(source_output)))
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())+sum(q['bytes'] for q in copies.values())+4*1024**2
    if total>MAX_BYTES:raise ValueError('qualification replay closure exceeds one GiB')
    for relative,row in copies.items():
        original=Path(row['original']);destination=root/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise ValueError('preserve existing replay input')
        shutil.copy2(original,destination)
        if sha(destination)!=row['sha256']:raise ValueError('replay input copy drift')
    shutil.copyfile(study/NAME,root/NAME)
    record=dict(schema='stable-pair-native-replay-closure-v1',copies=copies,predictions=commands,
                qualified_cases=8,training_replayed=False,replay_executed=False,upper_total_bytes=total,
                data_deduplicated_by_sha256=True)
    (root/'REPLAY_CLOSURE.json').write_bytes(encoded(record));(root/'REPRODUCE.sh').write_text('\n'.join(script)+'\n')
    (root/'README.md').write_text('''# Stable first-layer qualification

Eight native cases cover seven further constrained families. Original failures
remain failures. The stable representation passes unchanged probes; unmarked
runtime predictions reproduce originals exactly on all admitted population tables.
The certificate records measured score deltas and full provenance. No new training,
label calibration, model selection, outer or final-holdout evaluation occurred.

Run `bash REPRODUCE.sh` on Linux x86-64 to recreate the eight native conversions
and all 248 prediction TSVs without compiling or writing a program. It compares
every result byte for byte and requires a fresh replay directory. Binaries and
opaque TRAIN/inner-validation tables are deduplicated by SHA-256. A changed gate
or dropped source/seed is not part of this repair. This replay closure was prepared
but not rerun during archival; original native qualification and full destination
checksum verification are separate recorded evidence.
''')
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>total:
        raise ValueError('generated replay metadata exceeded its four-MiB allowance')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    from shaping_storage import reservation
    with reservation(a.study,a.root/'replay-input-admission',30*1024**3,bytes_required=MAX_BYTES):value=prepare(a.study,a.root)
    print(json.dumps(dict(cases=value['qualified_cases'],prediction_commands=len(value['predictions']),input_copies=len(value['copies']),bytes=value['upper_total_bytes'])),flush=True)
