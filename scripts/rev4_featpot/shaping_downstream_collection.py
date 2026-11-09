"""Collect admitted downstream phases with their complete registered exposure."""
import json
from pathlib import Path
from shaping_completion import sha


def validate(study,root,assignment,run,result):
    from shaping_downstream_worker import reproduce
    from shaping_rescue_worker import HOSTS
    if study is None:raise ValueError('downstream collection requires the original study gate')
    request=Path(assignment['worker_request_path'])
    if sha(request)!=assignment['worker_request_sha256']:raise ValueError('downstream request pin differs')
    record=json.loads(request.read_text());kind=record.get('execution_kind')
    if (record['schema']!='shaping-native-worker-v2' or kind not in ('outer','variant','extension')
        or record['host'] not in HOSTS or record['host']!=assignment['host']
        or record['study']!=str(study) or record['cell']!=run['cell']
        or record['plan_sha256']!=run['plan_sha256'] or record['build_sha256']!=run['build_sha256']):
        raise ValueError('downstream request scientific identity differs')
    resolution,_=reproduce(record,host=assignment['host'])
    if resolution!=record['resolution']:raise ValueError('downstream original gate differs')
    schedule=resolution['schedule'];cell=run['cell']
    stages={'outer':('outer','fresh_control_outer','fresh_control_outer_matched'),
            'variant':('source_curve','row_curve','shuffle','human_only'),'extension':('exposure_extension',)}
    if cell['stage'] not in stages[kind]:raise ValueError('downstream stage differs')
    family=cell['family'] if kind in ('outer','extension') else resolution['resolution']['resolved_family']
    names=['pretraining','fit'] if '_pretrained_' in family else ['fit']
    epochs=schedule['phase_epochs']
    if (len(epochs)!=len(names) or any(e not in (120,240,480) for e in epochs)
        or schedule['total_pair_draws']!=sum(epochs)*50000
        or run.get('process_role')!=kind+'_controller' or run.get('fit_directory')!='fit'
        or run['epochs']!=epochs[-1] or run['pairs_per_epoch']!=50000
        or run.get('training_exposure_pair_draws')!=schedule['total_pair_draws']
        or run.get('outer_results_read') is not False):
        raise ValueError('downstream controller exposure differs')
    expected=(dict(schedule=schedule,admission_sha256=resolution['admission_sha256'])
              if kind=='outer' else dict(resolution=resolution['resolution']))
    for key,value in {**expected,'training_exposure_pair_draws':schedule['total_pair_draws'],
                      'process_role':kind+'_controller','fit_directory':'fit','outer_results_read':False}.items():
        if run.get(key)!=value or result.get(key)!=value:raise ValueError('downstream controller identity differs: '+key)
    phases=result.get('phases',[])
    if len(phases)!=len(names):raise ValueError('downstream phase coverage differs')
    for name,epoch,phase in zip(names,epochs,phases):
        directory=root/name;initial=json.loads((directory/'RUN.json').read_text())
        if (json.loads((directory/'RESULT.json').read_text())!=phase
            or phase['stage']!='trained' or phase['exit_code']!=0 or phase['epochs']!=epoch
            or phase['pairs_per_epoch']!=50000 or phase['build_sha256']!=run['build_sha256']
            or any(phase.get(k)!=v for k,v in initial.items() if k!='stage')):
            raise ValueError('downstream native phase identity differs')
        checkpoints=directory/'checkpoints'
        if {p.name for p in checkpoints.glob('ckpt_epoch*.bin')}!={f'ckpt_epoch{i:03}.bin' for i in range(epoch)}:
            raise ValueError('downstream phase checkpoint coverage differs')
        if sha(checkpoints/f'ckpt_epoch{epoch-1:03}.bin')!=phase['bake_sha256']:
            raise ValueError('downstream phase final model differs')
    if result['bake_sha256']!=phases[-1]['bake_sha256']:raise ValueError('downstream controller final model differs')
    return run
