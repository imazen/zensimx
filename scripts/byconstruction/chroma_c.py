#!/usr/bin/env python3
"""Stage C scientific controls and joins; zenfleet owns distributed execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct
from collections import defaultdict
import sys
import subprocess
import os

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from chroma_pilot import save, sha
from chroma_pilot import admission
from chroma_b import PINS
from chroma_evif import specs
from chroma_b import cas

NAS=Path('/mnt/tower/output/zensim/byc-chroma-20261028')
PARENT=Path('/var/tmp/byc-mixes-20261025/Mlic/root')
BANK=Path('/mnt/v/output/zensim/byconstruction-2026-10-22/rev7raw/bank')
EXTRACTOR_SHA='ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8'
BUILD='b77883be18db5933a9db4b9e2670a2ba88a9729b'
FEATURE_ID='basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af'
IDS=list(range(228))+list(range(372,720))


def run_controls(binary, controls, out):
    assert sha(binary)==EXTRACTOR_SHA
    out.mkdir();output=out/'features.csv'
    env=os.environ.copy();env.update(ZENSIM_FORMULA_REV='7',ZENSIM_ROOT_FORM='sqrt',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env.pop('ZENSIM_MAX_TIER',None)
    command=[str(binary),'--corpus','pairs-tsv','--path',str(controls/'pairs.tsv'),'--out',str(output),
             '--restore-cuts','basic,peaks,v2','--input-contract','legacy-rgb8','--era-label','rev7_rowlanes',
             '--force-tier','native','--audit-jsonl',str(output)+'.audit.jsonl']
    with (out/'extract.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    verify_controls(controls,output,out/'PROOF.json')
    save(out/'COMMAND.json',dict(argv=command,extractor_sha256=EXTRACTOR_SHA,environment={k:env[k] for k in ['ZENSIM_FORMULA_REV','ZENSIM_ROOT_FORM','RAYON_NUM_THREADS','OMP_NUM_THREADS']}))


def prepare_controls(out):
    out.mkdir();(out/'src').mkdir();records=[];expected=[];mlic=[];pins={}
    for name,member in [('safesyn','safesyn'),('cid22','cid22_train')]:
        kp=PARENT/'wide/main/real'/f'{name}_fit.keys.parquet'
        selected=pq.read_table(kp).slice(0,6).to_pylist()
        bank=BANK/member
        manifest=json.loads((bank/'_MANIFEST.json').read_text())
        assert manifest['binary_sha256']==EXTRACTOR_SHA and manifest['feature_set_id']==FEATURE_ID
        assert manifest['build_commit']==BUILD and manifest['tier_request']=='native'
        keys=pq.read_table(bank/'keys.parquet').to_pylist()
        table=pq.ParquetFile(bank/'features.parquet');root=pq.ParquetFile(PARENT/'wide/main/real'/f'{name}_fit.parquet')
        rootrows=next(root.iter_batches(batch_size=6,columns=[f'f{i}' for i in IDS])).to_pylist()
        cache={};offsets=np.cumsum([0]+[table.metadata.row_group(i).num_rows for i in range(table.num_row_groups)])
        for index,item in enumerate(selected):
            rowid=item['source_row_id'];key=keys[rowid];assert item['pair_key']==key['pair_key']
            rg=int(np.searchsorted(offsets,rowid,side='right')-1)
            if rg not in cache:cache[rg]=table.read_row_group(rg,columns=[f'f{i}' for i in IDS])
            values=cache[rg].slice(rowid-int(offsets[rg]),1).to_pylist()[0]
            record=dict(set=name,source_row_id=rowid,pair_key=key['pair_key'],root_row=index)
            for prefix in ['ref','dist']:
                source=Path(key[prefix+'_path']);digest=sha(source);dest=out/'src'/(digest+source.suffix)
                if not dest.exists():shutil.copyfile(source,dest)
                assert sha(dest)==digest
                record[prefix+'_path']=str(dest);record[prefix+'_sha256']=digest
                record[prefix+'_pixels_sha256']=key[prefix+'_pixels_sha256']
            record['row_id']=len(records);records.append(record)
            expected.append(dict(row_id=record['row_id'],**values));mlic.append(dict(row_id=record['row_id'],**rootrows[index]))
        for path in [kp,bank/'_MANIFEST.json',bank/'keys.parquet']:
            pins[str(path)]=sha(path)
        pins[str(bank/'features.parquet')]=manifest['features_parquet_sha256']
    pq.write_table(pa.Table.from_pylist(expected),out/'EXPECTED_BANK_F64.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(mlic),out/'EXPECTED_MLIC_F32.parquet',compression='zstd')
    with (out/'pairs.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['ref_path','dist_path','human_score','row_id'])
        for r in records:writer.writerow([r['ref_path'],r['dist_path'],0,r['row_id']])
    save(out/'CONTROLS.json',dict(extractor_sha256=EXTRACTOR_SHA,build_commit=BUILD,feature_set_id=FEATURE_ID,
        feature_ids=IDS,records=records,parent_root=str(PARENT),parent_frozen_sha256=sha(PARENT/'wide/frozen.json'),input_pins=pins))


def verify_controls(controls, output, receipt):
    assert not receipt.exists()
    specification=json.loads((controls/'CONTROLS.json').read_text())
    manifest=json.loads(Path(str(output)+'.manifest.json').read_text())
    assert manifest['feature_set_id']==FEATURE_ID and str(manifest['formula_revision'])=='7'
    assert manifest['era_label']=='rev7_rowlanes'
    columns=['row_id']+[f'f{i}' for i in IDS]
    actual=pacsv.read_csv(output,convert_options=pacsv.ConvertOptions(include_columns=columns,column_types={'row_id':pa.int64(),**{f'f{i}':pa.float64() for i in IDS}}))
    actual=actual.sort_by('row_id');expected=pq.read_table(controls/'EXPECTED_BANK_F64.parquet').sort_by('row_id')
    root=pq.read_table(controls/'EXPECTED_MLIC_F32.parquet').sort_by('row_id')
    assert actual['row_id'].to_pylist()==expected['row_id'].to_pylist()==list(range(12))
    failures=[]
    for i in IDS:
        name=f'f{i}';a=actual[name].to_numpy();e=expected[name].to_numpy();r=root[name].to_numpy()
        assert np.isfinite(a).all() and np.isfinite(e).all()
        bad=np.flatnonzero(a.view('u8')!=e.view('u8'))
        cast=np.flatnonzero(a.astype('f4').view('u4')!=r.astype('f4').view('u4'))
        if len(bad) or len(cast):failures.append(dict(feature=i,bank_rows=bad.tolist(),mlic_rows=cast.tolist()))
    audit=[json.loads(line) for line in Path(str(output)+'.audit.jsonl').read_text().splitlines()]
    assert len(audit)==12
    for record,original in zip(audit,specification['records']):
        assert record['reference']==original['ref_path'] and record['distorted']==original['dist_path']
        for actual_key,expected_key in [('reference_pixels_sha256','ref_pixels_sha256'),('distorted_pixels_sha256','dist_pixels_sha256')]:
            assert record[actual_key]==original[expected_key]
    save(receipt,dict(passed=not failures,rows=12,measured_features=576,feature_width=1825,
        extractor_sha256=EXTRACTOR_SHA,build_commit=BUILD,feature_set_id=FEATURE_ID,
        bank_f64='bit-identical' if not failures else 'failed',mlic_f32='bit-identical' if not failures else 'failed',
        output_sha256=sha(output),controls_sha256=sha(controls/'CONTROLS.json'),failures=failures))
    assert not failures,'frozen Mlic extractor identity failed; feature declaration forbidden'


def prepare_band_selection(ledger, archive, out):
    """Schedule features from authenticated target receipts during media readback.

    This is not admission: the merged bank still needs the complete independent
    media readback and an exact kept-row reconciliation before training.
    """
    from concurrent.futures import ThreadPoolExecutor
    out.mkdir();views={v['path']:v for v in json.loads((NAS/'b/full/VIEWS.json').read_text())}
    rows=pq.read_table(ledger).to_pylist()
    assert len(rows)==len(views)==9162 and all(r['status']=='done' for r in rows)
    assert {r['image_path'] for r in rows}==set(views)
    def load(row):
        data=cas(archive,row['output_sha']).read_bytes();assert hashlib.sha256(data).hexdigest()==row['output_sha']
        return row,json.loads(data)
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(rows),32):
            for row,result in pool.map(load,rows[start:start+32]):
                v=views[row['image_path']]
                assert result['image_path']==row['image_path'] and result['knob_tuple_json']==row['knob_tuple_json']
                assert result['source_sha256']==v['sha256'] and result['reference_pixels_sha256']==v['pixels_sha256']
                assert result['runtime']['canonical_teacher_helper']==PINS[1] and result['runtime']['diagnostics']=='none'
                cells=sorted(result['cells'],key=lambda c:c['rung']);assert len(cells)==16
                assert all(c['family']=='band-ac' for c in cells)
                gate=admission([c['teacher_ssim2'] for c in cells])
                for c in cells:
                    records.append(dict(**c,view_id=v['id'],source=v['source'],image_path=v['path'],reference_group=v['reference_group'],
                        mlic_role=v['mlic_role'],content_class=v['content_class'],source_type=v['type'],width=v['width'],height=v['height'],
                        kept=c['rung'] in gate['kept_indices']))
            save(out/'PROGRESS.json',dict(views=min(start+32,len(rows)),required_views=len(rows),stage='selection only; full media admission still required'))
    pq.write_table(pa.Table.from_pylist(records),out/'CELLS.parquet',compression='zstd')
    save(out/'_MANIFEST.json',dict(qualified_for_leg=False,selection_only=True,rows=len(records),ledger_sha256=sha(ledger)))


def feature_spec(cells, proof, out, exclude, band_ac=False):
    """Freeze a disjoint wave; full B readback must qualify every row before leg assembly."""
    qualification=json.loads(proof.read_text())
    assert qualification['passed'] and qualification['extractor_sha256']==EXTRACTOR_SHA
    assert qualification['bank_f64']==qualification['mlic_f32']=='bit-identical'
    out.mkdir();views={v['path']:v for v in json.loads((NAS/'b/full/VIEWS.json').read_text())}
    excluded=set()
    for wave in exclude:
        excluded.update(pq.read_table(wave/'SELECTION.parquet',columns=['image_path'])['image_path'].to_pylist())
    grouped=defaultdict(list)
    for batch in pq.ParquetFile(cells).iter_batches():
        for row in batch.to_pylist():
            if row['image_path'] not in excluded:grouped[row['image_path']].append(row)
    selections=[];sources=[];families={k:v for k,v in specs().items() if (k=='band-ac' if band_ac else not k.startswith('band-'))}
    for image,rows in sorted(grouped.items()):
        assert image in views;view=views[image]
        assert len(rows)==(16 if band_ac else 76)
        by_family=defaultdict(list)
        for r in rows:
            assert struct.pack('>d',r['teacher_ssim2']).hex()==r['teacher_bits']
            assert r['reference_pixels_sha256']==view['pixels_sha256']
            assert r['mlic_role']==view['mlic_role'] and r['reference_group']==view['reference_group']
            by_family[r['family']].append(r)
        assert set(by_family)==set(families),'only the explicitly declared family set may enter this feature wave'
        selected=[]
        for family,(_,_,params) in families.items():
            ladder=sorted(by_family[family],key=lambda r:r['rung']);assert len(ladder)==len(params)
            gate=admission([r['teacher_ssim2'] for r in ladder])
            for index,r in enumerate(ladder):
                assert r['rung']==index and json.loads(r['params_json'])==params[index]
                assert r['kept']==(index in gate['kept_indices'])
                if r['kept']:selected.append(r)
        if not selected:continue
        evifs=sorted({r['evif_sha256'] for r in selected})
        source=dict(image_path=image,source_sha256=view['sha256'],pixels_sha256=view['pixels_sha256'],
                    identity=json.dumps({k:view[k] for k in ['id','reference_group','mlic_role','type','content_class']},sort_keys=True),
                    width=view['width'],height=view['height'])
        sources.append(dict(source=source,evif_shas=evifs));selections.extend(selected)
    assert sources and selections
    pq.write_table(pa.Table.from_pylist(selections),out/'SELECTION.parquet',compression='zstd')
    save(out/'spec.json',dict(sources=sources,implementation_shas=PINS+[EXTRACTOR_SHA],peak_mem_bytes=512*1024*1024))
    save(out/'_MANIFEST.json',dict(schema='chroma-stage-c-feature-wave-v1',qualified_for_leg=False,
        feature_set_id=FEATURE_ID,extractor_sha256=EXTRACTOR_SHA,views=len(sources),pairs=len(selections),
        source_cells=str(cells),source_cells_sha256=sha(cells),identity_proof=str(proof),identity_proof_sha256=sha(proof),
        selection_sha256=sha(out/'SELECTION.parquet'),excluded_waves=[str(p) for p in exclude],
        data_roles='all admitted pairs extracted; only fit/new-train may enter training; dev/calibration stay diagnostic',
        family_set='band-ac' if band_ac else 'core',
        final_gate='exact selection reconciliation against the fully verified family admission before leg/root assembly'))


def harvest_features(ledger, wave, archive, out, qualification=False):
    """Reuse the bank owner's CSV ingress and verify exact pixels/feature digests."""
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
    from rev5_bank import read_feature_csv
    out.mkdir();selected=defaultdict(list)
    for r in pq.read_table(wave/'SELECTION.parquet').to_pylist():selected[r['image_path']].append(r)
    rows=pq.read_table(ledger).to_pylist()
    if qualification:
        assert len(rows)==1 and out.name.startswith('qualification-')
        selected={rows[0]['image_path']:selected[rows[0]['image_path']]}
    assert len(rows)==len(selected) and all(r['status']=='done' for r in rows)
    assert len({r['image_path'] for r in rows})==len(rows) and {r['image_path'] for r in rows}==set(selected)
    bank_writer=None;key_writer=None;buffers=[];keys=[];count=0;verified=set();classes=defaultdict(int);roles=defaultdict(int)
    work=out/'readback';work.mkdir()
    def read(digest):
        path=cas(archive,digest);data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        verified.add(digest);return data
    try:
        for index,row in enumerate(sorted(rows,key=lambda r:r['image_path'])):
            result=json.loads(read(row['output_sha']))
            assert result['schema']=='evif-rev7-features-v1' and result['feature_set_id']==FEATURE_ID
            assert result['extractor_sha256']==EXTRACTOR_SHA and result['build_commit']==BUILD
            assert result['feature_width']==1825 and result['measured_feature_ids']==IDS
            assert result['image_path']==row['image_path'] and result['knob_tuple_json']==row['knob_tuple_json']
            expected=sorted(selected[row['image_path']],key=lambda r:(r['family'],r['rung']))
            actual={c['evif_sha256']:c for c in result['cells']};assert len(actual)==len(result['cells'])
            assert set(actual)=={r['evif_sha256'] for r in expected}
            assert set(actual)==set(json.loads(row['knob_tuple_json'])['evif_shas'])
            assert result['source_sha256']==json.loads(row['knob_tuple_json'])['source_sha256']
            assert all(r['reference_pixels_sha256']==result['reference_pixels_sha256'] for r in expected)
            for digest in result['sidecars'].values():
                if digest not in verified:read(digest)
            feature_csv=work/'features.csv';feature_csv.write_bytes(read(result['features_csv_sha256']))
            feature_ids=sorted({c['row_id'] for c in actual.values()})
            assert feature_ids==list(range(len(feature_ids)))
            raw,columns=read_feature_csv(feature_csv,feature_ids,1825)
            audits=[json.loads(line) for line in read(result['sidecars']['features.audit.jsonl']).splitlines()]
            audit_by_png={Path(r['distorted']).stem:r for r in audits};assert len(audits)==len(audit_by_png)==len(feature_ids)
            raw_f32=np.column_stack([raw[f'f{i}'].to_numpy() for i in range(1825)]).astype('<f4')
            digest_by_row={}
            for cell in actual.values():
                audit=audit_by_png[cell['png_sha256']]
                assert not audit['model_inputs'] and audit['reference']==row['image_path']
                assert audit['reference_pixels_sha256']==result['reference_pixels_sha256']
                assert audit['distorted_pixels_sha256']==cell['distorted_pixels_sha256']
                rid=cell['row_id']
                if rid not in digest_by_row:
                    digest_by_row[rid]=hashlib.sha256(raw_f32[rid].tobytes()).hexdigest()
                assert audit['canonical_features_f32_le_sha256']==digest_by_row[rid]
            view_keys=[];take=[]
            for selection in expected:
                cell=actual[selection['evif_sha256']]
                assert cell['jpeg_sha256']==selection['jpeg_sha256']
                assert cell['distorted_pixels_sha256']==selection['distorted_pixels_sha256']
                assert struct.pack('>d',selection['teacher_ssim2']).hex()==selection['teacher_bits']
                identity=json.dumps([selection['view_id'],selection['family'],selection['rung'],selection['evif_sha256']],separators=(',',':'))
                pair_key=hashlib.sha256(identity.encode()).hexdigest();global_row=count+len(view_keys)
                view_keys.append(dict(**selection,pair_key=pair_key,row_id=global_row,png_sha256=cell['png_sha256'],
                    canonical_features_f32_le_sha256=digest_by_row[cell['row_id']],feature_primary_sha256=row['output_sha']))
                take.append(cell['row_id']);classes[selection['content_class']]+=1;roles[selection['mlic_role']]+=1
            features=pa.table(columns).take(pa.array(take))
            features=features.add_column(0,'row_id',pa.array([r['row_id'] for r in view_keys],type=pa.int64()))
            features=features.add_column(0,'pair_key',pa.array([r['pair_key'] for r in view_keys]))
            buffers.append(features);keys.extend(view_keys);count+=len(view_keys)
            if len(keys)>=5000 or index+1==len(rows):
                bank=pa.concat_tables(buffers);key_table=pa.Table.from_pylist(keys)
                if bank_writer is None:
                    bank_writer=pq.ParquetWriter(out/'features.parquet',bank.schema,compression='zstd',compression_level=3)
                    key_writer=pq.ParquetWriter(out/'keys.parquet',key_table.schema,compression='zstd')
                bank_writer.write_table(bank,row_group_size=5000);key_writer.write_table(key_table,row_group_size=5000)
                buffers=[];keys=[]
                save(out/'PROGRESS.json',dict(stage='feature readback',views=index+1,required_views=len(rows),pairs=count))
    finally:
        if bank_writer:bank_writer.close();key_writer.close()
    assert count==sum(len(r) for r in selected.values())
    assert pq.ParquetFile(out/'features.parquet').metadata.num_rows==pq.ParquetFile(out/'keys.parquet').metadata.num_rows==count
    save(out/'_MANIFEST.json',dict(schema='rev5-featbank-v1',set='chroma',stage_c_wave=True,qualified_for_leg=False,
        qualification_only=qualification,rows=count,views=len(rows),feature_width=1825,dtype='float64',absent_slots='NaN',
        requested_slot_ranges=[[0,228],[372,720]],feature_set_id=FEATURE_ID,build_commit=BUILD,binary_sha256=EXTRACTOR_SHA,
        features_parquet_sha256=sha(out/'features.parquet'),keys_sha256=sha(out/'keys.parquet'),source_selection_sha256=sha(wave/'SELECTION.parquet'),
        ledger_sha256=sha(ledger),unique_objects_verified=len(verified),per_class=dict(classes),per_role=dict(roles),
        final_gate='full family selection reconciliation required before leg assembly; every target/pixel/feature digest exact'))
    save(out/'PROGRESS.json',dict(stage='complete',views=len(rows),pairs=count))


def merge_banks(banks, out, build_commit, ac_admission=None):
    """Reconcile disjoint waves with every fully verified B kept row."""
    out.mkdir();admission=NAS/'b/full/admission-r1';summary=json.loads((admission/'SUMMARY.json').read_text())
    assert summary['qualified_full_grid'] and summary['cells']==696312
    expected={}
    identity=['image_path','family','rung']
    bindings=['evif_sha256','jpeg_sha256','teacher_bits','reference_pixels_sha256','distorted_pixels_sha256','mlic_role','reference_group']
    admissions=[admission];required=251685;fit_required=232888;diagnostic_required=18797
    if ac_admission is not None:
        ac_summary=json.loads((ac_admission/'SUMMARY.json').read_text())
        assert ac_summary['qualified_full_grid'] and ac_summary['cells']==146592 and ac_summary['views']==9162
        assert ac_summary['training_allowed'] and set(ac_summary['totals'])=={'band-ac_ladders','band-ac_admitted','band-ac_kept'}
        required+=ac_summary['totals']['band-ac_kept']
        fit_required+=sum(ac_summary['per_role'][r]['band-ac_kept'] for r in ['fit','new-train'])
        diagnostic_required+=sum(ac_summary['per_role'][r]['band-ac_kept'] for r in ['dev','calibration'])
        admissions.append(ac_admission)
    for directory in admissions:
        for batch in pq.ParquetFile(directory/'CELLS.parquet').iter_batches(columns=identity+bindings+['kept']):
            for r in batch.to_pylist():
                if r['kept']:
                    key=tuple(r[k] for k in identity);assert key not in expected;expected[key]=tuple(r[k] for k in bindings)
    assert len(expected)==required
    bank_writer=None;key_writer=None;total=0;role_counts=defaultdict(int);class_counts=defaultdict(int);inputs={};pair_keys=set();measured=set(IDS)
    try:
        for bank in banks:
            manifest=json.loads((bank/'_MANIFEST.json').read_text())
            assert manifest['stage_c_wave'] and not manifest['qualification_only']
            assert manifest['feature_set_id']==FEATURE_ID and manifest['binary_sha256']==EXTRACTOR_SHA and manifest['build_commit']==BUILD
            assert sha(bank/'features.parquet')==manifest['features_parquet_sha256'] and sha(bank/'keys.parquet')==manifest['keys_sha256']
            inputs[str(bank/'_MANIFEST.json')]=sha(bank/'_MANIFEST.json')
            features=pq.ParquetFile(bank/'features.parquet');keys=pq.ParquetFile(bank/'keys.parquet');local=0
            for fb,kb in zip(features.iter_batches(batch_size=1000),keys.iter_batches(batch_size=1000),strict=True):
                assert fb.num_rows==kb.num_rows
                f=pa.Table.from_batches([fb]);k=pa.Table.from_batches([kb]);records=k.to_pylist()
                assert f['pair_key'].to_pylist()==k['pair_key'].to_pylist()
                assert f['row_id'].to_pylist()==k['row_id'].to_pylist()==list(range(local,local+len(records)))
                for i in range(1825):
                    a=f[f'f{i}'].to_numpy();assert f[f'f{i}'].type==pa.float64()
                    assert (np.isfinite(a).all() if i in measured else np.isnan(a).all())
                for record in records:
                    key=tuple(record[n] for n in identity)
                    assert expected.pop(key)==tuple(record[n] for n in bindings),'feature wave differs from full B admission'
                    assert record['pair_key'] not in pair_keys;pair_keys.add(record['pair_key'])
                    assert struct.pack('>d',record['teacher_ssim2']).hex()==record['teacher_bits']
                    role_counts[record['mlic_role']]+=1;class_counts[record['content_class']]+=1
                ids=pa.array(range(total,total+len(records)),type=pa.int64())
                f=f.set_column(f.schema.get_field_index('row_id'),'row_id',ids)
                k=k.set_column(k.schema.get_field_index('row_id'),'row_id',ids)
                if bank_writer is None:
                    bank_writer=pq.ParquetWriter(out/'features.parquet',f.schema,compression='zstd',compression_level=3)
                    key_writer=pq.ParquetWriter(out/'keys.parquet',k.schema,compression='zstd')
                bank_writer.write_table(f);key_writer.write_table(k);total+=len(records);local+=len(records)
                save(out/'PROGRESS.json',dict(stage='full B/feature reconciliation',pairs=total,required_pairs=required))
            assert local==manifest['rows']
    finally:
        if bank_writer:bank_writer.close();key_writer.close()
    assert not expected and total==required and role_counts['fit']+role_counts['new-train']==fit_required
    assert role_counts['dev']+role_counts['calibration']==diagnostic_required
    save(out/'_MANIFEST.json',dict(schema='rev5-featbank-v1',set='chroma',qualified_for_leg=ac_admission is not None,qualified_core=True,rows=total,
        feature_width=1825,dtype='float64',absent_slots='NaN',requested_slot_ranges=[[0,228],[372,720]],
        feature_set_id=FEATURE_ID,binary_sha256=EXTRACTOR_SHA,build_commit=BUILD,assembly_build_commit=build_commit,
        features_parquet_sha256=sha(out/'features.parquet'),keys_sha256=sha(out/'keys.parquet'),
        band_ac_included=ac_admission is not None,core_rows=251685,source_banks=inputs,per_role=dict(role_counts),per_class=dict(class_counts),
        full_ac_cells_sha256=sha(ac_admission/'CELLS.parquet') if ac_admission else None,
        ac_admission_summary_path=str(ac_admission/'SUMMARY.json') if ac_admission else None,
        ac_admission_summary_sha256=sha(ac_admission/'SUMMARY.json') if ac_admission else None,
        full_b_cells_sha256=sha(admission/'CELLS.parquet'),full_b_admission_summary_sha256=sha(admission/'SUMMARY.json'),
        checks=['all kept B identities exactly once','pixel identities exact','every native f32 digest exact in wave readback',
                'all576 f64 slots finite;1249 absent slotsNaN','target doubles preserve original bits','development/calibration excluded from fitting']))
    save(out/'PROGRESS.json',dict(stage='complete',pairs=total,qualified_for_leg=ac_admission is not None))


def seal_stage_c(out, build_commit, bank, roots, feature_audit, diagnostic_audit, diagnostic_result, leg_checks):
    """Freeze the identities of the verified stage-C closure; every input must already have passed its own gate."""
    assert not out.exists()
    manifest=json.loads((bank/'_MANIFEST.json').read_text())
    assert manifest['qualified_for_leg'] and manifest['qualified_core'] and manifest['band_ac_included'] and manifest['rows']==317008
    roles=manifest['per_role']
    assert roles['fit']+roles['new-train']==293318 and roles['dev']+roles['calibration']==23690
    for audit in (feature_audit,diagnostic_audit):
        summary=json.loads((audit/'SUMMARY.json').read_text());assert summary['conflicting_scientific_jobs']==0
    result=json.loads((diagnostic_result/'SUMMARY.json').read_text())
    assert result['passed'] and result['qualified_full_grid'] and result['targets_unchanged'] and result['admission_unchanged']
    assert result['selected_cells']==20858 and result['views']==8319
    checks=[json.loads((c).read_text()) for c in leg_checks];assert all(c['passed'] for c in checks)
    assert all(c['fitting_rows']==293318 and c['diagnostic_rows']==23690 for c in checks)
    files=[bank/'_MANIFEST.json',bank/'features.parquet',bank/'keys.parquet',feature_audit/'SUMMARY.json',diagnostic_audit/'SUMMARY.json',
           diagnostic_result/'SUMMARY.json',diagnostic_result/'DIAGNOSTICS.parquet',*leg_checks,*[r/'wide/frozen.json' for r in roots]]
    save(out,dict(schema='chroma-stage-c-ready-v1',build_commit=build_commit,observed_utc=__import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat(),totals=dict(pairs=manifest['rows'],fitting=293318,diagnostic=23690),
        per_role=roles,files={str(f):dict(sha256=sha(f),bytes=f.stat().st_size) for f in files},
        roots={str(r):sha(r/'wide/frozen.json') for r in roots},
        repeat_audits=dict(feature_extra_completions=334,diagnostic_extra_completions=185,conflicts=0),
        note='full-grid hash/seal validation, not independent decoder coverage; HDR legs not included'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare-controls');p.add_argument('out',type=Path)
    p=sub.add_parser('run-controls');p.add_argument('binary',type=Path);p.add_argument('controls',type=Path);p.add_argument('out',type=Path)
    p=sub.add_parser('verify-controls');p.add_argument('controls',type=Path);p.add_argument('output',type=Path);p.add_argument('receipt',type=Path)
    p=sub.add_parser('prepare-band-selection');p.add_argument('ledger',type=Path);p.add_argument('archive',type=Path);p.add_argument('out',type=Path)
    p=sub.add_parser('feature-spec');p.add_argument('cells',type=Path);p.add_argument('proof',type=Path);p.add_argument('out',type=Path);p.add_argument('--exclude-wave',action='append',type=Path,default=[]);p.add_argument('--band-ac',action='store_true')
    for name in ['harvest-features','qualify-features']:
        p=sub.add_parser(name);p.add_argument('ledger',type=Path);p.add_argument('wave',type=Path);p.add_argument('archive',type=Path);p.add_argument('out',type=Path)
    p=sub.add_parser('merge-banks');p.add_argument('out',type=Path);p.add_argument('build_commit');p.add_argument('banks',nargs='+',type=Path);p.add_argument('--ac-admission',type=Path)
    p=sub.add_parser('seal');p.add_argument('out',type=Path);p.add_argument('build_commit');p.add_argument('bank',type=Path)
    p.add_argument('--root',action='append',type=Path,required=True);p.add_argument('--feature-audit',type=Path,required=True)
    p.add_argument('--diagnostic-audit',type=Path,required=True);p.add_argument('--diagnostic-result',type=Path,required=True)
    p.add_argument('--leg-check',action='append',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='seal':seal_stage_c(args.out,args.build_commit,args.bank,args.root,args.feature_audit,args.diagnostic_audit,args.diagnostic_result,args.leg_check)
    elif args.command=='prepare-controls':prepare_controls(args.out)
    elif args.command=='run-controls':run_controls(args.binary,args.controls,args.out)
    elif args.command=='verify-controls':verify_controls(args.controls,args.output,args.receipt)
    elif args.command=='prepare-band-selection':prepare_band_selection(args.ledger,args.archive,args.out)
    elif args.command=='feature-spec':feature_spec(args.cells,args.proof,args.out,args.exclude_wave,args.band_ac)
    elif args.command=='merge-banks':merge_banks(args.banks,args.out,args.build_commit,args.ac_admission)
    else:harvest_features(args.ledger,args.wave,args.archive,args.out,qualification=args.command=='qualify-features')
