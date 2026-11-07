"""Verify copied immutable inputs and relocate only a baseline declaration path."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from shaping_completion import encoded,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','source','frozen','baseline-inputs'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--declaration-sha256',required=True);args=p.parse_args()
    files=json.loads(args.inputs.read_text())
    for row in files:
        path=Path(row['path'])
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('worker input drift: '+str(path))
    declaration=args.source/'scripts/rev4_featpot/color_noise_fx2_declaration.json'
    if sha(declaration)!=args.declaration_sha256:raise ValueError('worker declaration differs')
    args.baseline_inputs.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.frozen/'RESULT.json',args.baseline_inputs/'RESULT.json')
    commands={}
    for arm in ('new-single','deep-relu','deep-smooth'):
        path=args.frozen/arm/'command.json';command=json.loads(path.read_text())
        if command.count('--derived-inputs')!=1:raise ValueError('ambiguous baseline declaration')
        original=command[command.index('--derived-inputs')+1]
        command[command.index('--derived-inputs')+1]=str(declaration)
        out=args.baseline_inputs/arm/'command.json';out.parent.mkdir();out.write_bytes(encoded(command))
        commands[arm]=dict(original_command_sha256=sha(path),relocated_command_sha256=sha(out),
                           original_declaration_path=original,declaration_sha256=sha(declaration))
    receipt=dict(verified_files=len(files),verified_bytes=sum(r['bytes'] for r in files),
                 input_manifest_sha256=sha(args.inputs),baseline_result_sha256=sha(args.frozen/'RESULT.json'),
                 commands=commands,scientific_fits_started=False)
    (args.baseline_inputs/'ADMISSION.json').write_bytes(encoded(receipt));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
