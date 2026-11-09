"""Restart a registered paired extension through the qualified native trainer."""
from __future__ import annotations
import argparse,copy,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import SEEDS,encoded,sha
from shaping_execution import load_registration
from shaping_extension import resolve
from shaping_pretraining import structural_terms
from shaping_rescue import controls,rewrite_command,admitted_families
from shaping_rescue_execution import primary_source,phase
from shaping_storage import reservation


def command(original,binaries,dest,epochs):
    if epochs not in (240,480):raise ValueError('unregistered extension phase length')
    result=rewrite_command(original,binaries,dest,240)
    result[result.index('--epochs')+1]=str(epochs)
    # Analytic training materializes its event sequence. The doubled phase
    # receives a bounded 4 GiB cap; its actual peak must be measured, not scaled
    # from the qualified 240-epoch replay's resource receipt.
    if epochs==480:result[result.index('--mem')+1]='4G'
    return result


def optimizer(choice,analytic,groups):
    if choice!='base':return controls(choice,analytic,groups=groups)
    result=controls('init_half',analytic,groups=groups)
    result['init_scale']=1.
    if analytic:result['bias_scale']=1.
    return result


def fresh_command(original,seed):
    result=copy.deepcopy(original)
    for flag,value in zip(('--init-seed','--sample-seed'),SEEDS[seed]):
        cmd=result['command']
        if cmd.count(flag)!=1 or cmd[cmd.index(flag)+1]!=str(value):raise ValueError('fresh control source seeds differ')
        cmd[cmd.index(flag)+1]=str(value+1000000)
    return result


def execute(args):
    plan,cells,_=load_registration(args.study);resolution=resolve(args.study,args.cell);cell=resolution['cell']
    from shaping_reduced_scope import admit_cell
    admit_cell(args.study,cell)
    for name,pin in json.loads((args.binary_dir/'BUILD.json').read_text())['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('extension native binary drift')
    analytic=cell['family'] in ('softplus','silu_monotone','silu_diagnostic')
    if analytic:
        q=json.loads((args.neutral_qualification/'RESULT.json').read_text())
        if not q['match'] or q['epochs']!=240 or q['build_sha256']!=sha(args.binary_dir/'BUILD.json'):
            raise ValueError('full analytic neutral replay required')
    record=json.loads((args.study/'comparators'/cell['partition']/'GROUPS.json').read_text())
    groups=admitted_families(args.study,record)
    source,original=primary_source(args.study,cells,cell,cell['setting'])
    if cell['role']=='fresh_control':original=fresh_command(original,cell['seed'])
    dest=args.study/'runs'/cell['id']/f'attempt-{args.attempt:03}'
    if args.attempt<1 or dest.exists():raise ValueError('preserve extension attempts')
    with reservation(args.study,dest,plan['minimum_ssd_free_bytes']):
        dest.mkdir(parents=True)
        run=dict(cell=cell,resolution=resolution,execution_host=__import__('socket').gethostname(),outer_results_read=False,plan_sha256=sha(args.study/'PLAN.json'),
                 build_sha256=sha(args.binary_dir/'BUILD.json'),driver_sha256=sha(__file__),pid=os.getpid(),
                 process_role='extension_controller',stage='running',fit_directory='fit',
                 epochs=cell['phase_epochs'][-1],pairs_per_epoch=50000,
                 training_exposure_pair_draws=cell['total_pair_draws'])
        (dest/'RUN.json').write_bytes(encoded(run));phases=[];shaping=None
        try:
            if '_pretrained_' in cell['family']:
                family=cell['family'].split('_pretrained_')[0]+'_joint_identity'
                parent,parent_run=primary_source(args.study,cells,cell,cell['setting'],family)
                phases.append(phase(args.study,args.binary_dir,parent,parent_run,dest/'pretraining',240,
                    cell['optimizer_choice'],groups,optimizer=optimizer(cell['optimizer_choice'],False,groups),command_builder=command))
                prefix=dest/'pretraining/checkpoints/ckpt_epoch239.bin'
                shaping=json.loads(subprocess.check_output([str(args.binary_dir/'bake_dial_refit'),'read-meta',
                    '--in',str(prefix),'--key','zentrain.input_shaping'],text=True))
                if structural_terms(shaping)!=structural_terms(json.loads((parent/'SHAPING.json').read_text())):
                    raise ValueError('extended shape layout differs')
                shaping['trainable']=cell['family'].endswith('_joint')
            phases.append(phase(args.study,args.binary_dir,source,original,dest/'fit',run['epochs'],
                cell['optimizer_choice'],groups,shaping,optimizer=optimizer(cell['optimizer_choice'],analytic,groups),command_builder=command))
            (dest/'RESULT.json').write_bytes(encoded({**run,'phases':phases,'stage':'trained','exit_code':0,
                'bake_sha256':phases[-1]['bake_sha256'],'finished_utc':datetime.now(timezone.utc).isoformat()}))
        except Exception as error:
            (dest/'RESULT.json').write_bytes(encoded({**run,'stage':'failed','exit_code':1,'error':str(error),
                'finished_utc':datetime.now(timezone.utc).isoformat()}))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','binary-dir','neutral-qualification'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cell',required=True);p.add_argument('--attempt',type=int,default=1);execute(p.parse_args())
