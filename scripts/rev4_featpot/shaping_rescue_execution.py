"""Execute registered rescues after complete inner-only base selection."""
from __future__ import annotations
import argparse,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_pretraining import structural_terms
from shaping_rescue import resolve,controls,rewrite_command,admitted_families
from shaping_storage import reservation


def primary_source(root,cells,cell,setting,family=None):
    family=family or cell['family']
    candidates=[c for c in cells if c['stage']=='inner' and c['family']==family and
                c['partition']==cell['partition'] and c['seed']==cell['seed'] and c['setting']==setting]
    if len(candidates)!=1:raise ValueError('ambiguous primary fit')
    paths=[]
    for path in (root/'runs'/candidates[0]['id']).glob('attempt-*/RESULT.json'):
        result=json.loads(path.read_text())
        if result['stage']=='trained' and result['exit_code']==0:paths.append(path)
    if len(paths)!=1:raise ValueError('one completed primary fit required')
    source=paths[0].parent;run=json.loads((source/'RUN.json').read_text())
    result=json.loads(paths[0].read_text())
    if run['cell']!=candidates[0] or run['plan_sha256']!=sha(root/'PLAN.json'):
        raise ValueError('primary identity drift')
    expected=120 if '_pretrained_' in family else 240
    if run['epochs']!=expected or run['pairs_per_epoch']!=50000:raise ValueError('primary exposure differs')
    if sha(source/'checkpoints'/f"ckpt_epoch{run['epochs']-1:03}.bin")!=result['bake_sha256']:
        raise ValueError('primary model drift')
    return source,run


def phase(root,binaries,source,original,dest,epochs,rescue,groups,shaping=None):
    source_bake=source/'checkpoints'/f"ckpt_epoch{original['epochs']-1:03}.bin"
    def metadata(key):
        return subprocess.check_output([str(binaries/'bake_dial_refit'),'read-meta','--in',str(source_bake),'--key',key],text=True).strip()
    if (source/'HEAD.json').exists():
        initial=json.loads((source/'HEAD.json').read_text());stored=json.loads(metadata('zentrain.monotone_research'))
        if any(stored.get(key)!=value for key,value in initial.items()):raise ValueError('primary analytic manifest drift')
    if (source/'SHAPING.json').exists():
        if metadata('zentrain.input_shaping_initial_sha256')!=sha(source/'SHAPING.json'):
            raise ValueError('primary shaping manifest drift')
    dest.mkdir(parents=True,exist_ok=False);(dest/'checkpoints').mkdir()
    command=rewrite_command(original['command'],binaries,dest,epochs)
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('ZENSIM_MONOTONE_RESEARCH_','ZENSIM_INPUT_SHAPING_','ZENSIM_PLAIN_OPTIMIZER_')):env.pop(key)
    env.update(TMPDIR=str(Path.home()/'tmp'),ZENSIM_OPTIMIZER_DIAGNOSTICS='1',
               OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    files={}
    if (source/'HEAD.json').exists():
        head=json.loads((source/'HEAD.json').read_text());head['epochs']=epochs
        head['training']=controls(rescue,True,groups=groups)
        manifest=dest/'HEAD.json';manifest.write_bytes(encoded(head))
        env['ZENSIM_MONOTONE_RESEARCH_MANIFEST']=str(manifest);env['ZENSIM_MONOTONE_RESEARCH_SHA256']=sha(manifest)
        files[str(manifest)]=sha(manifest)
    else:
        manifest=dest/'OPTIMIZER.json';manifest.write_bytes(encoded(controls(rescue)))
        env['ZENSIM_PLAIN_OPTIMIZER_MANIFEST']=str(manifest);env['ZENSIM_PLAIN_OPTIMIZER_SHA256']=sha(manifest)
        files[str(manifest)]=sha(manifest)
    if shaping is None and (source/'SHAPING.json').exists():shaping=json.loads((source/'SHAPING.json').read_text())
    if shaping is not None:
        manifest=dest/'SHAPING.json';manifest.write_bytes(encoded(shaping))
        env['ZENSIM_INPUT_SHAPING_MANIFEST']=str(manifest);env['ZENSIM_INPUT_SHAPING_SHA256']=sha(manifest)
        files[str(manifest)]=sha(manifest)
    source_files=[source/'RUN.json',source/'RESULT.json',source/'keep.txt']
    source_files.extend(p for p in (source/'HEAD.json',source/'SHAPING.json') if p.exists())
    receipt=dict(command=command,source_run=str(source),source_run_sha256=sha(source/'RUN.json'),
                 source_file_sha256={str(p):sha(p) for p in source_files},
                 epochs=epochs,pairs_per_epoch=50000,manifest_sha256=files,build_sha256=sha(binaries/'BUILD.json'))
    with (dest/'train.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        receipt.update(pid=process.pid,stage='running',started_utc=datetime.now(timezone.utc).isoformat())
        (dest/'RUN.json').write_bytes(encoded(receipt));rc=process.wait()
    receipt.update(exit_code=rc,stage='trained' if rc==0 else 'failed',finished_utc=datetime.now(timezone.utc).isoformat())
    if rc==0:
        if len(list((dest/'checkpoints').glob('ckpt_epoch*.bin')))!=epochs:raise ValueError('phase checkpoint inventory differs')
        receipt['bake_sha256']=sha(dest/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin')
    (dest/'RESULT.json').write_bytes(encoded(receipt))
    if rc:raise RuntimeError('rescue phase failed; preserve '+str(dest))
    return receipt


def execute(args):
    plan,cells,_=load_registration(args.study);resolution=resolve(args.study,args.cell)
    cell=resolution['cell'];setting=resolution['resolved_setting']
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('rescue binary drift')
    if cell['family'] in ('softplus','silu_monotone','silu_diagnostic'):
        qualification=json.loads((args.neutral_qualification/'RESULT.json').read_text())
        if not qualification['match'] or qualification['epochs']!=240 or qualification['build_sha256']!=sha(args.binary_dir/'BUILD.json'):
            raise ValueError('full neutral analytic qualification required')
    record=json.loads((args.study/'comparators'/cell['partition']/'GROUPS.json').read_text())
    groups=admitted_families(args.study,record)
    source,original=primary_source(args.study,cells,cell,setting)
    dest=args.study/'runs'/cell['id']/f'attempt-{args.attempt:03}'
    if args.attempt<1 or dest.exists():raise ValueError('preserve existing rescue attempts')
    with reservation(args.study,dest,plan['minimum_ssd_free_bytes']):
        dest.mkdir(parents=True)
        run=dict(cell=cell,plan_sha256=sha(args.study/'PLAN.json'),build_sha256=sha(args.binary_dir/'BUILD.json'),
                 resolution=resolution,pid=os.getpid(),process_role='rescue_controller',stage='running',
                 driver_sha256=sha(__file__),epochs=120 if '_pretrained_' in cell['family'] else 240,
                 pairs_per_epoch=50000,training_exposure_pair_draws=12000000,fit_directory='fit')
        (dest/'RUN.json').write_bytes(encoded(run));phases=[];shaping=None
        try:
            if '_pretrained_' in cell['family']:
                family=cell['family'].split('_pretrained_')[0]+'_joint_identity'
                parent,parent_run=primary_source(args.study,cells,cell,setting,family)
                phases.append(phase(args.study,args.binary_dir,parent,parent_run,dest/'pretraining',120,cell['rescue'],groups))
                prefix=dest/'pretraining/checkpoints/ckpt_epoch119.bin'
                shaping=json.loads(subprocess.check_output([str(args.binary_dir/'bake_dial_refit'),'read-meta',
                    '--in',str(prefix),'--key','zentrain.input_shaping'],text=True))
                if structural_terms(shaping)!=structural_terms(json.loads((parent/'SHAPING.json').read_text())):
                    raise ValueError('rescue learned shape layout differs')
                shaping['trainable']=cell['family'].endswith('_joint')
            phases.append(phase(args.study,args.binary_dir,source,original,dest/'fit',run['epochs'],cell['rescue'],groups,shaping))
            (dest/'RESULT.json').write_bytes(encoded({**run,'phases':phases,'exit_code':0,
                'stage':'trained','bake_sha256':phases[-1]['bake_sha256'],'finished_utc':datetime.now(timezone.utc).isoformat()}))
        except Exception as error:
            (dest/'RESULT.json').write_bytes(encoded({**run,'stage':'failed','exit_code':1,
                'error':str(error),'finished_utc':datetime.now(timezone.utc).isoformat()}))
            raise



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','binary-dir','neutral-qualification'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cell',required=True);p.add_argument('--attempt',type=int,default=1)
    execute(p.parse_args())
