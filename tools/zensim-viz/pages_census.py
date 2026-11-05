#!/usr/bin/env python3
"""Conservative source census, including native tools and regression tests."""
import argparse, hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXT={'.rs','.js','.mjs','.css','.html','.py','.toml','.sh','.yml','.yaml'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--enforce',action='store_true');a=p.parse_args()
    tracked=subprocess.check_output(['jj','file','list'],cwd=ROOT,text=True).splitlines()
    files=[]
    for name in tracked:
        path=ROOT/name
        in_scope=(name.startswith(('tools/zensim-viz/','zensim-workbench/')) and not name.startswith('zensim-workbench/loop-converge/')) or name in {'zensim/src/feature_v2_viz.rs','zensim/src/research_viz.rs'}
        if in_scope and (path.suffix in EXT or name.endswith('.yml.in')):
            data=path.read_bytes();files.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    # Shared-file integration code that remains in the tree. Diff metadata,
    # context and removed lines are not code bytes in git.
    patch=subprocess.check_output(['jj','diff','--from',a.base,'--git','zensim/Cargo.toml','zensim/src/feature_v2.rs','zensim/src/research.rs','justfile'],cwd=ROOT)
    integration=b''.join(line[1:] for line in patch.splitlines(keepends=True) if line.startswith(b'+') and not line.startswith(b'+++'))
    total=sum(f['bytes'] for f in files)+len(integration)
    result=dict(code_bytes=total,limit_exclusive=500000,passed=total<500000,files=files,integration_diff_bytes=len(integration),base=a.base,scope='Tracked visualizer/workbench code, native preparation and historical tests, both metric observer modules, and shared integration additions. Excludes the separate JPEG XL loop-convergence CLI, prose, lockfiles and generated evidence/data.')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['code_bytes','passed','integration_diff_bytes']}))
    if a.enforce and not result['passed']:raise SystemExit('Visualizer source exceeds the owner budget')

if __name__=='__main__':main()
