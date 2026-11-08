"""Transport corrected inner fits through their activated, source-bound owner."""
import argparse,json,socket,os
from pathlib import Path
from shaping_completion import REPO,SEEDS,sha
from shaping_corrected_view import admit_fit,admit_parent


def context(study,cell_id,binaries,*,host=None):
    from shaping_execution import load_registration,prepare_partition
    from shaping_pretraining import completed_parent
    from shaping_reduced_scope import admit_cell
    plan,cells,partitions=load_registration(study)
    found=[c for c in cells if c['id']==cell_id]
    if len(found)!=1:raise ValueError('one effective corrected cell required')
    cell=found[0];binding=admit_fit(study,cell,sha(binaries/'BUILD.json'),120)
    if binding is None:raise ValueError('only activated corrected inner fits supported')
    admit_cell(study,cell,host=host)
    if (study/'runs'/cell_id).exists():raise ValueError('preserve every prior attempt; no automatic replay')
    groups_path=study/'comparators'/cell['partition']/'GROUPS.json'
    if not groups_path.exists():raise ValueError('prepare the registered TRAIN partition before transport')
    groups=prepare_partition(study,cell['partition'],partitions[cell['partition']],plan)
    parent,source,result=completed_parent(study,cell,cells)
    identity=json.loads((source/'RUN.json').read_text())
    if (identity['cell']!=parent or identity['epochs']!=240 or identity['pairs_per_epoch']!=50000
        or identity['plan_sha256']!=sha(study/'PLAN.json') or result['cell']!=parent
        or result['bake_sha256']!=sha(source/'checkpoints/ckpt_epoch239.bin')):
        raise ValueError('matching parent identity or exposure differs')
    admit_parent(study,cell,parent,identity['build_sha256'],sha(binaries/'BUILD.json'))
    build=json.loads((binaries/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(binaries/name)!=pin:raise ValueError('corrected native binary drift')
    if any(weight<=0 or validation!=0 for _,_,weight,validation,_ in groups['groups']):
        raise ValueError('corrected fits require TRAIN-only groups')
    from shaped_linear_study import BASE
    if plan['input_pins'].get(str(BASE))!=sha(BASE):raise ValueError('registered base recipe drift')
    recipe={**json.loads(BASE.read_text())['recipe'],'shared_hidden_activation':'relu','mse_weight':groups.get('mse_weight',1)}
    owner=configure(binaries)
    command=owner.train_command([(n,Path(p),tw,vw,m) for n,p,tw,vw,m in groups['groups']],
        *(seed+1000000 for seed in SEEDS[cell['seed']]),1853,source/'keep.txt','N',
        study/'runs'/cell_id/'attempt-001/best.bin',recipe,
        strict_admission=True,data_role_decision=Path(groups['decision']))
    if Path(command[0])!=binaries/'zensim_mlp_train':raise ValueError('corrected command builder selected another binary')
    scope=dict(cell=cell,corrected_binding=binding,parent_cell=parent,source_run=str(source),
        parent_result_sha256=sha(source/'RESULT.json'),prefix_sha256=sha(source/'checkpoints/ckpt_epoch119.bin'),
        group_receipt_sha256=sha(groups_path),base_recipe_sha256=sha(BASE),command_template=command,
        epochs=120,pairs_per_epoch=50000,outer_results_read=False)
    return dict(cell=cell,scope=scope,groups=groups,parent=source)


def resolve_inputs(study,cell_id,host,binaries,qualification):
    from shaping_rescue_worker import HOSTS,reads,seal_inputs
    # Bind lazily used command/snapshot owners too; a full source export alone
    # must not let modified worker modules escape the capsule's drift check.
    import importlib
    for name in ('v2_common','v2_lodo_mlp','shaping_snapshot_retention'):
        importlib.import_module(name)
    if host not in HOSTS:raise ValueError('approved corrected-fit worker required')
    with reads() as paths:
        admitted=context(study,cell_id,binaries,host=host)
        source=admitted['parent']
        paths.add(source/'SHAPING.json')
        if qualification!=study.parent/'corrected-control-qualification-20261108':
            raise ValueError('exact registered compatibility directory required')
    paths.add(Path(__file__).absolute())
    return seal_inputs(paths,study,admitted['cell'],host,binaries,qualification,admitted['scope'],execution_kind='corrected_inner')


def reproduce(record,*,host=None):
    study=Path(record['study']);binaries=Path(record['binary_dir'])
    admitted=context(study,record['cell']['id'],binaries,host=socket.gethostname() if host is None else host)
    return admitted['scope'],['python3','-B',str(REPO/'scripts/rev4_featpot/shaping_corrected_worker.py'),
        '--study',str(study),'--binaries',str(binaries),'--cell',record['cell']['id']]


def configure(binaries):
    # The existing command owner is unchanged; these are the same module settings
    # used by its original 120-epoch pretrained/fresh execution path.
    import v2_common
    import v2_lodo_mlp as owner
    os.environ['REV4_V2_BIN_DIR']=str(binaries)
    v2_common.TRAINER=owner.TRAINER=binaries/'zensim_mlp_train'
    v2_common.FITBIN=owner.FITBIN=binaries/'bake_dial_refit'
    owner.EPOCHS=120;owner.PAIRS_PER_EPOCH=50000;owner.LOG_EVERY=1
    return owner


def fit(study,cell_id,binaries):
    from shaping_execution import run_native
    admitted=context(study,cell_id,binaries)
    configure(binaries)
    return run_native(study,binaries,admitted['cell'],admitted['groups'],1,120,'2G')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binaries',type=Path,required=True)
    p.add_argument('--cell',required=True);a=p.parse_args()
    print(json.dumps(fit(a.study,a.cell,a.binaries)),flush=True)
