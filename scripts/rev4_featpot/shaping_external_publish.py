"""Publish scalar results from disjoint batches of the fixed external registry."""
import argparse,copy,hashlib,json,math,os
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def legacy_inventories(archives):
    inventories=[]
    for root in archives:
        receipt=json.loads((root/'MIRROR_VERIFIED.json').read_text())
        destination=Path(receipt['destination'])
        if not receipt['all_artifact_sha256_verified'] or sha(root/'SHA256SUMS')!=sha(destination/'SHA256SUMS'):
            raise ValueError('legacy external archive verification differs')
        pins={}
        for line in (root/'SHA256SUMS').read_text().splitlines():
            pin,name=line.split('  ',1);pins[name]=pin
        if pins.get('MIRROR_VERIFIED.json')!=sha(root/'MIRROR_VERIFIED.json') or sha(destination/'MIRROR_VERIFIED.json')!=pins['MIRROR_VERIFIED.json']:
            raise ValueError('legacy external archive receipt differs')
        inventories.append(pins)
    return inventories


def scatter(path,population):
    value=json.loads(path.read_text());n=population['rows']
    if any(len(value[k])!=n for k in ('indices','predicted','target','source','distortion')):
        raise ValueError('external scatter row coverage differs')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for k in ('predicted','target') for v in value[k]):
        raise ValueError('external scatter contains nonfinite responses')
    if len(set(value['source']))!=population['sources']:
        raise ValueError('external scatter source coverage differs')


def publish(root,batches,archives=()):
    registry_path=root/'REGISTRATION.json';registry=json.loads(registry_path.read_text());pin=sha(registry_path)
    registered={c['id']:c for c in registry['cells']}
    if len(registered)!=len(registry['cells']):raise ValueError('duplicate registered external model')
    inventories=legacy_inventories(archives);seen=set();results=[];coverage={};sources=[]
    for batch in batches:
        if sha(batch/'REGISTRATION.json')!=pin:raise ValueError('external batch registration differs')
        bindings_path=batch/'BINDINGS-initial.json';bindings=json.loads(bindings_path.read_text())
        if bindings['registration_sha256']!=pin:raise ValueError('external binding registration differs')
        models={row['cell']['id']:row for row in bindings['models']}
        if len(models)!=len(bindings['models']):raise ValueError('duplicate bound external model')
        for cid,model in models.items():
            if model['cell']!=registered.get(cid):raise ValueError('bound external model is outside the fixed registry')
        driver_pins={json.loads(p.read_text())['driver_sha256'] for p in batch.glob('EXECUTION_SOURCE*.json')}
        sources.append(dict(root=os.path.relpath(batch,root),bindings_sha256=sha(bindings_path)))
        for path in sorted((batch/'results').glob('*/*/RESULT.json')):
            row=json.loads(path.read_text());cell=row['cell'];cid=cell['id'];dataset=row['dataset'];key=(cid,dataset)
            if key in seen:raise ValueError('duplicate external dataset result; preserve and reconcile batches')
            if (cid not in models or path.parent.parent.name!=cid or cell!=models[cid]['cell'] or dataset not in registry['datasets']
                or row['registration_sha256']!=pin or row['schema']!='shaping-fixed-external-result-v1'
                or row['selection_permitted'] is not False or row['training_cells_added']!=0):
                raise ValueError('external result identity, exposure or selection scope differs')
            if any(row[name]!=models[cid][name] for name in ('bake_sha256','report_sha256')):
                raise ValueError('external result differs from the previously bound model')
            result_pin=sha(path)
            if 'driver_sha256' in row:
                if row['driver_sha256'] not in driver_pins:raise ValueError('undeclared external scoring driver')
            elif not any(pins.get(path.relative_to(batch).as_posix())==result_pin for pins in inventories):
                raise ValueError('legacy result requires its verified immutable archive binding')
            if sha(path.parent/'native.tsv')!=row['prediction_sha256']:
                raise ValueError('external native predictions differ')
            expected={'all','clean10'} if dataset=='live' else {'all'}
            if set(row['populations'])!=expected:raise ValueError('external population coverage differs')
            item=copy.deepcopy(row)
            for name,population in item['populations'].items():
                expected_rows=267 if name=='clean10' else registry['datasets'][dataset]['rows']
                statistics=population['statistics']
                if (population['rows']!=expected_rows or statistics['n']!=expected_rows or statistics['n_dropped']!=0
                    or (name=='clean10' and population['sources']!=10)):
                    raise ValueError('external native population denominator differs')
                pairs=Path(population['scatter'])
                if pairs.is_symlink() or not pairs.resolve().is_relative_to(path.parent.resolve()):
                    raise ValueError('external scatter is outside its result directory')
                scatter(pairs,population)
                population.update(scatter_url=os.path.relpath(pairs,root),scatter_sha256=sha(pairs))
            item.update(source_result_url=os.path.relpath(path,root),source_result_sha256=result_pin,
                        source_bindings_sha256=sha(bindings_path))
            results.append(item);seen.add(key);coverage.setdefault(cid,set()).add(dataset)
    complete={cid for cid,datasets in coverage.items() if datasets==set(registry['datasets'])}
    value=dict(schema='shaping-external-dashboard-v1',registration_sha256=pin,results=results,
               completed_models=len(complete),pending=[c for c in registry['cells'] if c['id'] not in complete],
               batches=sources,publication_driver_sha256=sha(Path(__file__)),selection_permitted=False)
    temporary=root/'INDEX.json.writing';temporary.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n')
    temporary.replace(root/'INDEX.json')
    return dict(completed_models=len(complete),dataset_reports=len(results),pending=len(value['pending']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--batch-root',type=Path,action='append',required=True)
    parser.add_argument('--legacy-archive',type=Path,action='append',default=[])
    args=parser.parse_args();print(json.dumps(publish(args.root,args.batch_root,args.legacy_archive)),flush=True)
