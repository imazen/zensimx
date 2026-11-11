"""Report observed native training resources without reruns or prorated prefixes."""
import argparse,json,math,re
from pathlib import Path
from shaping_completion import encoded,sha


def parse_log(text,epochs):
    matches=list(re.finditer(r'^[ \t]*epoch\s+(\d+)\s+\|[^\n]*\| t=([0-9.]+)s[ \t]*$',text,re.M))
    if epochs<=0 or not matches or len(matches)%epochs:raise ValueError('complete ordered epoch timing coverage required')
    blocks=len(matches)//epochs
    first=[m.group(0) for m in matches[:epochs]]
    if any([m.group(0) for m in matches[i:i+epochs]]!=first for i in range(0,len(matches),epochs)):
        raise ValueError('nonidentical repeated epoch log blocks')
    rows=[m.groups() for m in matches[:epochs]]
    if [int(e) for e,_ in rows]!=list(range(epochs)):
        raise ValueError('complete ordered epoch timing coverage required')
    times=[float(t) for _,t in rows]
    if any(not math.isfinite(t) or t<0 for t in times) or times!=sorted(times):raise ValueError('invalid cumulative training time')
    receipts=re.findall(r'^run-heavy: done rc=(\d+) (\d+)s \| peak-RSS ([0-9.]+)GiB \| min-avail (\d+)MiB \| peak-load ([0-9.]+)\s*$',text,re.M)
    if len(receipts)!=1 or receipts[0][0]!='0':raise ValueError('one successful resource receipt required')
    rc,elapsed,rss,available,load=receipts[0]
    if any(not math.isfinite(float(v)) for v in (elapsed,rss,available,load)):raise ValueError('nonfinite resource receipt')
    return dict(wrapper_elapsed_seconds=int(elapsed),wrapper_peak_rss_gib_reported=float(rss),
                wrapper_min_available_mib=int(available),wrapper_peak_load=float(load),
                wrapper_receipt_line='run-heavy: done rc='+rc+' '+elapsed+'s | peak-RSS '+rss+'GiB | min-avail '+available+'MiB | peak-load '+load,
                native_elapsed_seconds_by_epoch=times,identical_epoch_log_blocks=blocks,
                precision='Wrapper seconds are integer-reported; RSS/load retain wrapper rounding; native elapsed is reported to 0.1 seconds.')


def continuation_resources(directory, result, pins):
    root=directory.parent;proof_path=root/'CONTINUATION.json'
    proof=json.loads(proof_path.read_text())
    if (result.get('continuation_sha256')!=sha(proof_path)
        or proof['original_phase_sha256']!=sha(directory/'RUN.json')):
        raise ValueError('continued resource provenance differs')
    pins[str(proof_path)]=sha(proof_path)
    segments=[]
    for name in ('original-train.log','native-train.log'):
        path=root/'continuation-evidence'/name
        if proof['evidence_sha256'].get(str(path.relative_to(root)))!=sha(path):
            raise ValueError('continued raw resource log drift')
        pins[str(path)]=sha(path);rows={}
        for match in re.finditer(r'^[ \t]*epoch\s+(\d+)\s+\|[^\n]*\| t=([0-9.]+)s[ \t]*$',path.read_text(),re.M):
            epoch,t=int(match[1]),float(match[2])
            if not math.isfinite(t) or t<0 or epoch in rows and rows[epoch]!=t:
                raise ValueError('continued native clock differs')
            rows[epoch]=t
        if not rows:raise ValueError('continued native elapsed observations missing')
        times=[rows[e] for e in sorted(rows)]
        if times!=sorted(times):raise ValueError('nonmonotone segment clock')
        segments.append(dict(log=str(path),epoch_elapsed_seconds=rows))
    return dict(wrapper_elapsed_seconds=None,wrapper_peak_rss_gib_reported=None,
        wrapper_min_available_mib=None,wrapper_peak_load=None,native_elapsed_seconds_by_epoch=None,
        observed_native_segments=segments,completed_epochs_recomputed=proof['completed_epochs_replayed'],
        resource_coverage_complete=False,
        interpretation='Retained native segment clocks; no comparable single-wrapper elapsed time or peak RSS. No prorating, clock concatenation or memory estimate.')


def phase_record(directory,epochs,pins):
    result_path=directory/'RESULT.json';log=directory/'train.log'
    result=json.loads(result_path.read_text())
    if result['stage']!='trained' or result['exit_code']!=0 or result['epochs']!=epochs:raise ValueError('completed matching phase required')
    model=directory/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin'
    if sha(model)!=result['bake_sha256']:raise ValueError('final phase model drift')
    for p in (directory/'RUN.json',result_path,log,model):pins[str(p)]=sha(p)
    return dict(directory=str(directory),epochs=epochs,pair_draws=epochs*result['pairs_per_epoch'],
                execution_host=result.get('execution_host'),resources=(continuation_resources(directory,result,pins) if (directory.parent/'CONTINUATION.json').exists() else parse_log(log.read_text(),epochs)))


def report(run,output):
    if output.exists():raise ValueError('preserve resource report')
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    from shaping_execution import load_registration
    study=run.parents[2];_,cells,_=load_registration(study)
    if identity['cell'] not in cells or identity['plan_sha256']!=sha(study/'PLAN.json'):raise ValueError('registered resource scope required')
    if result['cell']!=identity['cell'] or result['stage']!='trained' or result['exit_code']!=0:raise ValueError('completed matching run required')
    pins={str(p):sha(p) for p in (run/'RUN.json',run/'RESULT.json',study/'PLAN.json')}
    phases=[]
    if result.get('phases'):
        if len(result['phases']) not in (1,2):raise ValueError('unsupported phase count')
        names=['pretraining','fit'] if len(result['phases'])==2 else ['fit']
        for name,phase in zip(names,result['phases']):
            if json.loads((run/name/'RESULT.json').read_text())!=phase:raise ValueError('phase result identity differs')
            phases.append(phase_record(run/name,phase['epochs'],pins))
    else:phases.append(phase_record(run/identity.get('fit_directory','.'),result['epochs'],pins))
    reused=None;provenance=run/'SHAPING.provenance.json'
    if provenance.exists():
        receipt=json.loads(provenance.read_text());parent=Path(receipt['source_run'])
        if sha(parent/'RESULT.json')!=receipt['parent_result_sha256'] or sha(parent/'checkpoints/ckpt_epoch119.bin')!=receipt['prefix_sha256'] or sha(run/'SHAPING.json')!=receipt['manifest_sha256']:
            raise ValueError('reused prefix evidence drift')
        parent_identity=json.loads((parent/'RUN.json').read_text())
        if parent_identity['cell']!=receipt['parent_cell'] or parent_identity['plan_sha256']!=identity['plan_sha256']:raise ValueError('reused parent identity differs')
        parent_cost=phase_record(parent,parent_identity['epochs'],pins)
        if receipt['pretraining_epochs']!=120 or receipt['pretraining_pair_draws']!=6000000:raise ValueError('reused prefix exposure differs')
        for p in (provenance,run/'SHAPING.json',parent/'checkpoints/ckpt_epoch119.bin'):pins[str(p)]=sha(p)
        reused=dict(provenance=str(provenance),parent=parent_cost,
                    prefix_native_elapsed_seconds=parent_cost['resources']['native_elapsed_seconds_by_epoch'][119],
                    prefix_peak_rss_gib=None,prefix_wrapper_elapsed_seconds=None,
                    interpretation='Observed cumulative trainer time at epoch 119, before that checkpoint dump. Full-parent wrapper time/RSS cover 240 epochs and are not prorated or added to child cost. Prefix-only wrapper time and peak RSS were not measured.')
    for path,pin in pins.items():
        if sha(path)!=pin:raise ValueError('resource evidence changed')
    record=dict(cell=identity['cell'],execution_host=identity.get('execution_host'),phases=phases,reused_pretraining=reused,
                directly_executed_phase_wrapper_seconds=(sum(p['resources']['wrapper_elapsed_seconds'] for p in phases) if all(p['resources']['wrapper_elapsed_seconds'] is not None for p in phases) else None),
                scientific_eligibility_asserted=False,training_replayed=False,source_pins=pins,driver_sha256=sha(__file__),
                interpretation='Observed concurrent-run resource accounting, not an isolated benchmark or cross-host speed comparison. Shared parent work remains explicit and is not treated as free.')
    output.write_bytes(encoded(record));return record


def qualify(study,output):
    from shaping_execution import load_registration
    from shaping_completion import NEURAL
    if output.exists():raise ValueError('preserve qualification')
    output.mkdir(parents=True);_,cells,_=load_registration(study);chosen=[]
    for family in NEURAL:
        chosen.append((family,next(c for c in cells if c['stage']=='inner' and c['partition']=='inner-0-0' and c['family']==family and c['setting']==0 and c['seed']==0)))
    chosen.append(('rescue',next(c for c in cells if c['stage']=='rescue' and c['family']=='relu' and c['partition']=='inner-0-0' and c['seed']==0 and c['rescue']=='consolidate_lr')))
    rows=[]
    for name,cell in chosen:
        attempts=[p.parent for p in (study/'runs'/cell['id']).glob('attempt-*/RESULT.json') if json.loads(p.read_text()).get('stage')=='trained']
        if len(attempts)!=1:raise ValueError('one completed qualification attempt required')
        record=report(attempts[0],output/(name+'.json'))
        rows.append(dict(name=name,cell=cell,wrapper_seconds=record['directly_executed_phase_wrapper_seconds'],reused_prefix=record['reused_pretraining'] is not None))
    (output/'SUMMARY.json').write_bytes(encoded(dict(rows=rows,complete=True,training_replayed=False)))
    return dict(models=len(rows),training_replayed=False)


def batch(study,output):
    """Read all completed registered neural fits; retain every refusal explicitly."""
    from shaping_execution import load_registration
    from shaping_completion import NEURAL
    if output.exists():raise ValueError('preserve batch resource evidence')
    output.mkdir(parents=True);_,cells,_=load_registration(study)
    rows=[];pending=[];refused=[]
    for cell in cells:
        # Later role-based cells resolve their concrete family in RUN.json.
        results=sorted((study/'runs'/cell['id']).glob('attempt-*/RESULT.json'))
        completed=[]
        for path in results:
            result=json.loads(path.read_text())
            identity=json.loads((path.parent/'RUN.json').read_text())
            family=identity.get('resolved_family',cell['family'])
            if family in NEURAL and result.get('stage')=='trained' and result.get('exit_code')==0:
                completed.append(path.parent)
        if not completed:
            if cell['family'] in NEURAL:pending.append(cell['id'])
            continue
        if len(completed)!=1:
            refused.append(dict(cell=cell['id'],error='ambiguous completed attempts',attempts=list(map(str,completed))))
            continue
        try:
            record=report(completed[0],output/(cell['id']+'.json'))
            rows.append(dict(cell=cell,report_sha256=sha(output/(cell['id']+'.json')),
                             directly_executed_phase_wrapper_seconds=record['directly_executed_phase_wrapper_seconds'],
                             reused_pretraining=record['reused_pretraining'] is not None))
        except (ValueError,KeyError,FileNotFoundError) as error:
            refused.append(dict(cell=cell['id'],error=str(error),run=str(completed[0])))
    summary=dict(plan_sha256=sha(study/'PLAN.json'),driver_sha256=sha(__file__),
                 rows=rows,pending_neural_cells=pending,refused_completed_runs=refused,
                 completed_fit_resource_coverage_complete=not refused,all_experiments_complete=False,
                 training_replayed=False,scope='Completed neural fits only; pending cells and nonneural controls are not measured here.')
    (output/'SUMMARY.json').write_bytes(encoded(summary))
    return dict(reported=len(rows),pending_neural=len(pending),refused_completed=len(refused),training_replayed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--run',type=Path);mode.add_argument('--qualify-study',type=Path);mode.add_argument('--batch-study',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.batch_study:print(json.dumps(batch(a.batch_study,a.output)))
    elif a.qualify_study:print(json.dumps(qualify(a.qualify_study,a.output)))
    else:
        r=report(a.run,a.output)
        print(json.dumps(dict(cell=r['cell'],phases=len(r['phases']),reused_pretraining=r['reused_pretraining'] is not None,training_replayed=False)))
