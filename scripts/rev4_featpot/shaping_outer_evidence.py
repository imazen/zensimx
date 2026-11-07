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


def collect_matched(study,*,role=None,condition=None,fresh_family=None):
    """Validate every matched arm before opening any prediction array."""
    from shaping_evaluation import outer_context,VARIANT_STAGES
    from shaping_variant_execution import resolve as resolve_variant
    from shaping_variant_inputs import load_variant
    plan,cells,partitions=load_registration(study);plan_pin=sha(study/'PLAN.json')
    admitted={o:resolve(study,o) for o in range(3)}
    admissions={o:a for o,(a,_) in admitted.items()}
    pairs,seed_ids=matched_cells(cells,admissions,role=role,condition=condition,fresh_family=fresh_family)
    wanted={c['id']:c for pair in pairs for c in (pair['candidate'],pair['control'])}
    pins={str(study/'PLAN.json'):plan_pin};pending=[]
    for o,(_,path) in admitted.items():pins[str(path)]=sha(path)
    for cell in wanted.values():
        outer=int(cell['partition'].split('-')[1]);admission,path=admitted[outer];admission_pin=sha(path)
        family=admission['finalist_roles'][cell['family']] if cell['stage'] in VARIANT_STAGES else cell['family']
        complete=[]
        for run_dir in sorted((study/'runs'/cell['id']).glob('attempt-*')):
            if not (run_dir/'RESULT.json').is_file():raise ValueError('unfinished matched attempt')
            result=json.loads((run_dir/'RESULT.json').read_text())
            if result['exit_code']!=0:continue
            run=json.loads((run_dir/'RUN.json').read_text())
            for value in (run,result):
                if value['cell']!=cell or value['plan_sha256']!=plan_pin:raise ValueError('matched run identity differs')
            if result['stage']!=('trained' if family in NEURAL else 'complete'):raise ValueError('matched terminal stage differs')
            if family in NEURAL:
                context=outer_context(study,run,cells)
                if context is None or context['admission_sha256']!=admission_pin:raise ValueError('matched neural admission differs')
                membership=context.get('membership',partitions[cell['partition']]);schedule=context['schedule']
                if (run['epochs']!=schedule['phase_epochs'][-1] or run['training_exposure_pair_draws']!=schedule['total_pair_draws']
                    or [p['epochs'] for p in result['phases']]!=schedule['phase_epochs']
                    or any(p['stage']!='trained' or p['exit_code']!=0 for p in result['phases'])):
                    raise ValueError('matched phase exposure or completion differs')
                model=run_dir/run['fit_directory']/'checkpoints'/f"ckpt_epoch{run['epochs']-1:03}.bin"
                if sha(model)!=result['bake_sha256']:raise ValueError('matched final model drift')
                pins[str(model)]=sha(model)
            elif cell['stage'] in VARIANT_STAGES:
                resolution=resolve_variant(study,cell['id'])
                if run['variant']!=resolution or run['admission_sha256']!=admission_pin or run['selected_setting']!=resolution['schedule']['setting']:
                    raise ValueError('matched control variant resolution differs')
                _,_,_,membership=load_variant(study,cell['id'])
            else:
                if run['admission_sha256']!=admission_pin or run['selected_setting']!=admission['schedules'][family]['setting']:
                    raise ValueError('matched control admission differs')
                membership=partitions[cell['partition']]
            if membership['test']!=partitions[cell['partition']]['test']:raise ValueError('matched outer test membership differs')
            representation=admission.get('evaluation_representations',{}).get(family,{'representation':'original'})
            evaluation=(study/'reexport_max_pairs'/cell['id']/run_dir.name/'final' if representation['representation']!='original' else run_dir/'evaluation')
            if not (evaluation/'COMPLETE.json').is_file():raise ValueError('matched evaluation incomplete')
            complete.append((run_dir,evaluation,result,representation,membership,family))
        if len(complete)!=1:raise ValueError('exactly one completed matched attempt required')
        pending.append((cell,admission_pin,*complete[0]))
    # Validate every report, model and prediction hash before the first array read.
    for cell,admission_pin,run_dir,evaluation,result,representation,membership,family in pending:
        path=evaluation/'REPORT.json';marker=evaluation/'COMPLETE.json';row=json.loads(path.read_text());done=json.loads(marker.read_text())
        if done['cell']!=cell['id'] or done['report_sha256']!=sha(path):raise ValueError('matched completion marker differs')
        if (row['cell']!=cell or row['plan_sha256']!=plan_pin or row.get('admission_sha256')!=admission_pin
            or row.get('outer_results_read') is not True or 'outer_test' not in row
            or 'exposed_development' in row or 'full_fit_validation' in row):raise ValueError('matched report scope differs')
        if sha(evaluation/'predictions.npz')!=row['predictions_sha256']:raise ValueError('matched prediction artifact drift')
        if family in NEURAL:
            if row['checkpoint_epoch']!=result['epochs']-1:raise ValueError('matched report is not scheduled final checkpoint')
            if representation['representation']=='original':
                if row['checkpoint_sha256']!=result['bake_sha256']:raise ValueError('matched report model differs')
            else:
                receipt_path=Path(row['reexport_receipt']);receipt=json.loads(receipt_path.read_text())
                if (sha(receipt_path)!=row['reexport_receipt_sha256'] or receipt['original_sha256']!=result['bake_sha256']
                    or receipt['evaluation_build_sha256']!=representation['evaluation_build_sha256']
                    or row['evaluation_build_sha256']!=representation['evaluation_build_sha256']
                    or receipt['training_build_sha256']!=result['build_sha256']
                    or receipt['representation']!=representation['representation']):raise ValueError('matched repaired identity differs')
                if receipt['bake'] not in receipt['file_sha256']:raise ValueError('matched repaired bake unpinned')
                for name,pin in receipt['file_sha256'].items():
                    if sha(name)!=pin:raise ValueError('matched repaired artifact drift')
                    pins[name]=pin
                if sha(receipt['bake'])!=row['checkpoint_sha256']:raise ValueError('matched repaired report model differs')
                pins[str(receipt_path)]=sha(receipt_path)
        else:
            model=run_dir/('model.opaque-vector.zcth' if family=='extra_trees' else 'native/model.bin')
            qualification=row['qualification'];pin=qualification['native_tree_gate']['model_sha256'] if family=='extra_trees' else qualification['model_sha256']
            if sha(model)!=pin:raise ValueError('matched native control model drift')
            pins[str(model)]=pin
        pins.update({str(p):sha(p) for p in (run_dir/'RUN.json',run_dir/'RESULT.json',path,marker,evaluation/'predictions.npz')})
    outputs={}
    for cell,admission_pin,run_dir,evaluation,result,representation,membership,family in pending:
        indices=sorted(set(membership['train'])|set(membership['test']))
        row,values,evidence=read(evaluation,cell,plan_pin,admission_pin,indices);pins.update(evidence)
        outputs[cell['id']]=dict(cell=cell,report=row,indices=indices,predictions=values)
    for name,pin in pins.items():
        if sha(name)!=pin:raise ValueError('matched evidence changed during collection')
    arms={key:[outputs[pair[key]['id']] for pair in pairs] for key in ('candidate','control')}
    qualification={str(o):a['qualifications'] for o,a in admissions.items()}
    return plan,partitions,arms,pins,qualification,seed_ids
