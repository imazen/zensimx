"""Require the entire primary outer matrix before opening prediction tables."""
from __future__ import annotations
import json
from pathlib import Path
from shaping_completion import FAMILIES,NEURAL,ROLES,seeds,sha
from shaping_execution import load_registration
from shaping_outer_admission import resolve
from shaping_prediction_evidence import read


def primary_cells(cells):
    selected=[c for c in cells if c['stage']=='outer']
    expected={(f,f'outer-{o}',s) for o in range(3) for f in FAMILIES for s in seeds(f)}
    if len(selected)!=102 or {(c['family'],c['partition'],c['seed']) for c in selected}!=expected:
        raise ValueError('complete 102-cell registered outer matrix required')
    return sorted(selected,key=lambda c:(c['partition'],c['family'],-1 if c['seed'] is None else c['seed']))


def collect(study):
    plan,cells,partitions=load_registration(study);plan_pin=sha(study/'PLAN.json')
    selected=primary_cells(cells)
    # All three full inner gates are reproduced before even checking reports.
    admissions={o:resolve(study,o) for o in range(3)}
    pending=[];pins={str(study/'PLAN.json'):plan_pin};qualifications={}
    for o,(admission,path) in admissions.items():
        pins[str(path)]=sha(path);qualifications[str(o)]=admission['qualifications']
    for cell in selected:
        o=int(cell['partition'].split('-')[1]);admission,path=admissions[o];admission_pin=sha(path)
        attempts=sorted((study/'runs'/cell['id']).glob('attempt-*'));complete=[]
        for run_dir in attempts:
            if not (run_dir/'RESULT.json').exists():raise ValueError('unfinished outer attempt: '+str(run_dir))
            result=json.loads((run_dir/'RESULT.json').read_text())
            if result['exit_code']!=0:continue
            if result['stage']!=('trained' if cell['family'] in NEURAL else 'complete'):
                raise ValueError('outer result stage differs')
            run=json.loads((run_dir/'RUN.json').read_text())
            for value in (run,result):
                if value['cell']!=cell or value['plan_sha256']!=plan_pin or value['admission_sha256']!=admission_pin:
                    raise ValueError('outer fit identity differs')
            schedule=admission['schedules'][cell['family']]
            if cell['family'] in NEURAL:
                if run['schedule']!={**schedule,'fresh':False}:raise ValueError('outer fit schedule differs')
                model=run_dir/'fit/checkpoints'/f"ckpt_epoch{run['epochs']-1:03}.bin"
                if sha(model)!=result['bake_sha256']:raise ValueError('outer trained model drift')
                pins[str(model)]=sha(model)
            elif run['selected_setting']!=schedule['setting']:raise ValueError('outer control setting differs')
            representation=admission.get('evaluation_representations',{}).get(cell['family'],{'representation':'original'})
            evaluation=(study/'reexport_max_pairs'/cell['id']/run_dir.name/'final'
                        if representation['representation']!='original' else run_dir/'evaluation')
            if not (evaluation/'COMPLETE.json').is_file():raise ValueError('outer evaluation incomplete: '+str(evaluation))
            complete.append((run_dir,evaluation,result,representation))
        if len(complete)!=1:raise ValueError('exactly one complete outer attempt required: '+cell['id'])
        pending.append((cell,admission_pin,*complete[0]))
    # No table is opened until every primary cell has a terminal fit and report.
    outputs=[]
    for cell,admission_pin,run_dir,evaluation,result,representation in pending:
        membership=partitions[cell['partition']]
        expected=sorted(set(membership['train'])|set(membership['test']))
        report,values,evidence=read(evaluation,cell,plan_pin,admission_pin,expected);pins.update(evidence)
        for name in ('RUN.json','RESULT.json'):pins[str(run_dir/name)]=sha(run_dir/name)
        if cell['family'] in NEURAL:
            if representation['representation']=='original':
                if report['checkpoint_sha256']!=result['bake_sha256']:raise ValueError('outer report model differs')
            else:
                receipt_path=Path(report['reexport_receipt']);receipt=json.loads(receipt_path.read_text())
                if (sha(receipt_path)!=report['reexport_receipt_sha256'] or receipt['original_sha256']!=result['bake_sha256']
                    or receipt['evaluation_build_sha256']!=representation['evaluation_build_sha256']
                    or report['evaluation_build_sha256']!=representation['evaluation_build_sha256']
                    or receipt['training_build_sha256']!=result['build_sha256']
                    or receipt['representation']!=representation['representation']):
                    raise ValueError('outer re-export identity differs')
                if receipt['bake'] not in receipt['file_sha256']:raise ValueError('unpinned outer re-export model')
                for name,pin in receipt['file_sha256'].items():
                    if sha(name)!=pin:raise ValueError('outer re-export artifact drift')
                    pins[name]=pin
                if sha(receipt['bake'])!=report['checkpoint_sha256']:raise ValueError('outer report re-export differs')
                pins[str(receipt_path)]=sha(receipt_path)
        outputs.append(dict(cell=cell,report=report,indices=expected,predictions=values))
    return plan,partitions,outputs,pins,qualifications


def matched_cells(cells,admissions,*,role=None,condition=None,fresh_family=None):
    """Resolve complete pairs from inner decisions and frozen matrix metadata."""
    if set(admissions)!={0,1,2}:raise ValueError('all three outer admissions required')
    if fresh_family is not None:
        if role is not None or condition is not None or fresh_family not in NEURAL or '_pretrained_' not in fresh_family:
            raise ValueError('one registered pretrained-family comparison required')
        seed_ids=(0,1,2)
    else:
        if role not in ROLES or not isinstance(condition,dict):raise ValueError('registered variant role and condition required')
        keys={'source_curve':{'stage','fraction','order_seed'},'row_curve':{'stage','fraction','order_seed'},
              'shuffle':{'stage','permutation','permutation_seed'},'human_only':{'stage'}}
        if condition.get('stage') not in keys or set(condition)!=keys[condition['stage']]:
            raise ValueError('exact registered variant condition required')
        seed_ids=tuple(seeds(role))
    def one(matches):
        if len(matches)!=1:raise ValueError('exactly one registered cell per paired arm required')
        return matches[0]
    pairs=[]
    for outer in range(3):
        admission=admissions[outer]
        if not admission['outer_evaluation_admitted']:raise ValueError('matched outer comparison not admitted')
        family=fresh_family or admission['finalist_roles'][role];partition=f'outer-{outer}'
        for seed in seed_ids:
            base=one([c for c in cells if c['stage']=='outer' and c['family']==family and c['partition']==partition and c['seed']==seed])
            if fresh_family is not None:
                control=one([v for v in admission['fresh_controls'] if fresh_family in v['paired_families'] and v['cell']['seed']==seed])
                schedule=admission['schedules'][family]
                if (control['setting'],control['optimizer_choice'],control['epochs'])!=(schedule['setting'],schedule['optimizer_choice'],schedule['phase_epochs'][-1]):
                    raise ValueError('fresh control setting, optimizer or exposure differs')
                if control['cell']['partition']!=partition or control['cell']['family']!='relu':raise ValueError('fresh control fold or family differs')
                pairs.append(dict(candidate=base,control=control['cell'],resolved_family=family))
            else:
                candidate=one([c for c in cells if c['family']==role and c['partition']==partition and c['seed']==seed and all(c.get(k)==v for k,v in condition.items())])
                pairs.append(dict(candidate=candidate,control=base,resolved_family=family))
    return pairs,seed_ids
