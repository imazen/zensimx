#!/usr/bin/env python3
"""Render measured A2 receipts; never invent missing results or stage adoption."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics

def read(root,name):return json.loads((root/name).read_text())
def report(root,destination,mirror_state):
    families=['ycc444','ycc420','xyb-capped','xyb-extended','band-dc','band-ac']
    summaries={f:read(root,f+'/SUMMARY.json') for f in families}
    ladders={f:read(root,f+'/LADDERS.json') for f in families}
    base=root.parent;spec=read(base,'pilot-run04/SOURCES.json');sources={s['id']:s for s in spec['sources']}
    expected={'ycc444':(162,138,1649),'ycc420':(162,133,1535),'xyb-capped':(162,95,646),'xyb-extended':(162,95,1484),'band-dc':(12,0,0),'band-ac':(12,12,105)}
    for name,(views,passed,kept) in expected.items():
        measured=summaries[name]
        if (measured['views'],measured['passed'],measured['kept'])!=(views,passed,kept):raise ValueError('different pilot needs a reviewed report')
    raw=ladders['xyb-extended'];capped={x['image']:x for x in ladders['xyb-capped']};scales=summaries['xyb-extended']['scales']
    shared=[(c,capped[x['image']]['cells'][i]) for x in raw for i,c in enumerate(x['cells'][:14])]
    beyond=sum(i>=14 for x in raw for i in x['admission']['kept_indices'])
    stops=Counter(scales[x['admission']['stop_index']] for x in raw if x['admission']['stop_index'] is not None)
    third=Counter(scales[x['admission']['kept_indices'][2]] for x in raw if x['admission']['kept_indices'])
    lines=['# Chroma ladders — Stage A2 report','',
    '**MISSING for a usable HDR training family:** 4:4:4 is unsupported by the pinned SVT adapter; 0/4 measured HDR ladders pass the unchanged admission rule. No HDR rung is admitted. Stage B, Stage C and training remain unstarted and require their respective approvals. A2 pilot generation, scoring and read-back are complete.','',
    f'Archive verification: **{mirror_state}**. Data: `/var/tmp/byc-chroma-20261028/a2/` on i270; mirror: `/mnt/tower/output/zensim/byc-chroma-20261028/a2/`. Stage A remains accepted; its full report is preserved in `report-stage-A-accepted.md`.','',
    '## Origin of the <1,000,000-pixel cap','',
    'The owner said on **2026-09-19**, as supplied in the coordinator’s corrected brief §8.3: **“No need to go huge (hard to compute) like above 1MP”**, and **“we just don\'t want to mess scales up by doing tiny thumbnails too much”**, with Mitchell preferred over Lanczos. During Stage A the user also directly said **“keep images under 1mp, scale or crop if needed”**. That live message was my immediate source for the strict `<1,000,000 pixels` condition and the 960-long-edge proposal. The cap was owner scope; im2402’s separate source-decoder guard did not establish it. No 1024–2048 size experiment ran.','',
    '## EVIF pilot versus the sweep pilot','',
    '| Family | EVIF ladders admitted | EVIF kept rungs | Original Stage A ladders |','|---|---:|---:|---:|']
    prior={'ycc444':'140/162','ycc420':'140/162','xyb-capped':'80/162','xyb-extended':'—','band-dc':'—','band-ac':'—'}
    for f in families:
        s=summaries[f];lines.append(f"| {f} | {s['passed']}/{s['views']} | {s['kept']} | {prior[f]} |")
    lines += ['',
    'The main rerun preserves the original 162 views and their TRAIN provenance. EVIF writes 12,312 main-family cells and 384 band-subset cells: **12,696 scored containers**, all with BITS retained and JPEG SHA256 CAS copies. YCbCr uses quality 95, unit Y-table multipliers, stepped Cb/Cr multipliers, zero-bias scales 1, AQ on, deringing/trellis off, in 444/420. These are unit multipliers on jpegli’s base Y table; Stage A’s all-ones arm used an absolute unit DQT. The encoder controls therefore differ.','',
    'EVIF pins zenjpeg `a2a128ab0bc36f01338225f61c285f462ae67ca6`. Stage A used the cached worker executable `37c3bc112d8bdc86b949005f8fcafd419510b88b7dd7c7a8938ba59251c4e50f` from `exec-cvvdp-0a61830d`; its path-dependency zenjpeg source revision is not recoverable from the image metadata/Cargo lock. I did not assume identical encoder pins: the entire admission was rerun with EVIF. Current EVIF generator SHA256: `232434619478f3ce076e895f3b4342b00f86e8b633a0da5123867d2e45dab800`.','',
    'Band subset uses the first 12 frozen views (six CID22 TRAIN origins, each at 256/512). DC-only changes admit 0/12; AC-only admit 12/12, keeping 105 rungs. This demonstrates a distinct subset result, not broad class qualification. Neither band family is added to the full-grid proposal before review.','',
    '## Extended XYB','',
    f"Raw X/B table multipliers 0.1–1024×, Y fixed, admit **95/162** ladders and keep **1,484** rungs. **{beyond}** kept rungs are above 5×. Every ladder reaches its first non-drop; latest first stop is **{max(stops)}×**. Capped `zenjpeg-color` admits the same 95 views and keeps 646 rungs. The raw prefix through 5× keeps 690, so the 838 difference between families is not a pure range-extension gain.", '',
    f"Shared-range encodes match byte for byte in {sum(c['encode_sha256']==d['encode_sha256'] for c,d in shared)}/{len(shared)} cells. All 162 views match at multiplier 1.0. Maximum shared-range score difference is {max(abs(c['teacher_ssim2']-d['teacher_ssim2']) for c,d in shared):.15g}. Raw table multiplication and the distance API use different frequency mappings: zenjpeg’s `distance_to_scale` is nonlinear above its threshold. Numeric multipliers are therefore separately labelled; the baseline parity gate does not assert equality at every non-unit multiplier.", '',
    'Multiplier at which each admitted raw ladder has its third kept rung: '+', '.join(f'{s:g}×: {n}' for s,n in sorted(third.items()))+'.', '',
    'First-stop distribution, including dropped ladders: '+', '.join(f'{s:g}×: {n}' for s,n in sorted(stops.items()))+'.','',
    'The full-grid proposal keeps both separately labelled XYB families pending owner confirmation. Late endpoint damage never rescues a ladder rejected by an earlier tie/increase.','',
    '## Teacher identity and container contract','',
    'Every scored SDR container stores `ssim2/safesyn-teacher` with implementation pin `frozen-safe-syn-executable-sha256:7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87`. That preserved executable scores persisted PNG pairs. Its incidental Rev3 feature CSV is not a Stage C feature bank.','']
    replay=read(root,'judge-replay/REPLAY.json')
    lines += [f"EVIF fast-ssim2 `54fabca9` replays **{replay['exact_replays']}/4** stored SafeSyn controls exactly. All four decoded JPEG pixel hashes match the old inputs, and the frozen teacher reproduces all four old scores on those same pixels. EVIF’s current judge remains diagnostic.", '',
    '| Control | Frozen target | EVIF fast-ssim2 diagnostic |','|---|---:|---:|']
    for i,r in enumerate(replay['controls']):lines.append(f"| {i} | {r['expected_safesyn_score']:.17g} | {r['evif_judge']['value']:.17g} |")
    lines += ['',
    'Render writes legacy major-0 containers; `wire append` requires major 1. The new `score-import` command writes a new legacy container, binds both raster digests, preserves existing media/recipes/unknown metadata, refuses duplicate implementation scores and existing outputs. Independent read-back verified all 12,696 f64 scores/pins and unchanged media, recipes and digests. Pillow 12.3.0 independently decoded every JPEG, checked all CAS identities, every scored PNG’s RGB hash against the frozen teacher input, and fixed Y DQT within every ladder. There are 10,031 unique main/subset JPEG SHA256s.','',
    '## HDR pilot','']
    hdr=read(root,'hdr/SUMMARY.json');hl=read(root,'hdr/LADDERS.json')
    lines += [
    'The small tested `evif-probe/examples/chroma_hdr.rs` driver uses zenavif `a7638f6e` / SVT `0f63ca13`. Four canonical imazen-26 TRAIN origins also belong to E26’s HDRTEACH TRAIN population: 1066, 1070, 1230, 1508. Original 384×512 / 512×384 references are RGB16 PQ with BT.709 or Display-P3 primaries. The native PNG codec decoder supplies CICP; zenpixels-convert converts their primaries to BT.2020 before encoding. No RGB8 narrowing or HDR-to-SDR rendering enters the target.','',
    'There are **68** 10-bit 420 AVIFs: quality 95, speed 6, one thread, chroma delta-q `[u,v]=[s,s]` for s=0,4,…,60,63. These are signed qindex offsets, not absolute chroma qualities. Independent FFmpeg/libdav1d 8.0.1-3ubuntu2 read every stream: PQ, BT.2020, 10-bit 420, full range, **MC=6 (BT.601)** as the pinned adapter signals. Luma base qindex stays **12**, luma DC delta 0; shared U/V DC/AC deltas match every requested override. This is not labelled BT.2020-NCL matrix coding.','',
    'The E26 teacher is the exact preserved zenmetrics source `348bde5bf96d635383965d80d9adb8f9c195fd14` binary SHA256 `1278c6935ad4ebbb624e6213c56ad672602b05433b288ea1f9274dbf02dee721`. HDR-VDP-3.0.7 target = 10×q_jod. E26 control replays **9.579838179794153** exactly. Conditions: 60 ppd, quality task, common absolute RGB BT.709 nits ingress, LED LCD sRGB emission, age 24, surround none, with E26’s frozen defaults. The binary’s common-native ingress accepts PQ PNG/JXL only. Each AVIF is decoded through the pinned imazen decoder into an exact RGB16 PQ BT.2020 PNG; this lossless rendition feeds the unchanged teacher. All 68 AVIF encodes reproduced byte for byte on a second run.','',
    '| Origin | First score | Last score | Stop delta-q | Kept |','|---|---:|---:|---:|---:|']
    for r in hl:
        a=r['admission'];stop='end of range' if a['stop_index'] is None else r['cells'][a['stop_index']]['chroma_q'][0]
        lines.append(f"| {r['source']} | {r['cells'][0]['teacher_score']:.12f} | {r['cells'][-1]['teacher_score']:.12f} | {stop} | {len(a['kept_indices'])} |")
    lines += ['',
    '**0/4 ladders admit:** none achieves even one ≥1-point kept step. The measured q95 / delta≤63 combination does not qualify a teacher leg. Two curves additionally stop at small reversals. No threshold was relaxed. 444 produces an explicit unsupported-backend error; implementing a new SVT 444 seam is outside this small driver. PU-SSIMULACRA2 was not used as a target.','',
    'EVIF render/probe currently owns an RGB8 image path. HDR adoption needs a native tagged pixel input, native PQ rendition I/O and hashes binding that signaling, plus HDR judges with frozen viewing conditions. Major-1’s HDR design/carrier work is prior infrastructure, not an already-enabled HDR render path. HDR pilot AVIFs, exact PQ16 renditions, recipes and teacher parquet remain files.','',
    '## Size axis and substitution','']
    size=read(root,'size/TIMING.json')
    lines += [f"The cost pilot has **{size['views']}** views from im9040, im5044 and the substitute im2406: Mitchell long-edge fits 32,64,128,192,256,384,512,640,768,960 and native 256/512 crops. Only **{size['tiny_views']}/36** are 32–64 thumbnails. All six crops match independent integer crop pixels exactly. im2406 crops are bottom-right windows inside the selected source region; the other two use center crops. Largest actual raster: **{size['max_pixels']:,} pixels**. No upscaling or 1024–2048 size was used.", '',
    f"Preparation took {size['preparation_seconds']:.6f} seconds. Each size row below measures 36 EVIF cells: three sources × three registered settings × four families, including process startup and PNG/container I/O. These are measured pilot costs, not a projected full-grid runtime. The 432 cost-control containers retain BITS and CAS copies; they are unscored diagnostics, not training candidates.", '',
    '| View | EVIF wall seconds / 36 cells |','|---|---:|']
    for i,row in enumerate(size['families']['ycc444']):lines.append(f"| {row['view']} | {sum(v[i]['seconds'] for v in size['families'].values()):.6f} |")
    sub=read(root,'SUBSTITUTION.json');selected=sub['selected']
    lines += ['',
    f"**im2402 substitute:** im{selected['origin']}/{selected['crop_label']}, distance **{selected['distance']:.15g}** from im2402/c50_bl in the existing **{sub['content_feature_count']}-feature** z-scored selection space, with the same GEOM exclusion. Target feature row exists. Candidates are same-content-class TRAIN units; original-source eligibility was tested with the unchanged frozen preparation binary. 2400/2404/2406/2408 pass; 2402 still returns `LimitExceeded(pixel count exceeds limit)`. Canonical LFS hashes were checked for staged sources. This is a **nearest-feature substitute, not a certified cluster-mate**. Original K500 replay matched **457/500** representatives; only the representatives TSV was retained, per coordinator. Reconstruction stopped when the coordinator supplied the deterministic fallback.", '',
    'Selection parquet SHA256 `aab7c7d2f06687a069283f0e43b9fc31087f216a3d6120b12231d0ef619af774`; existing selection owner SHA256 `a0d3591dbf85ff1a8d2d60f4a4db39e9d02da9946a89829c4834718825e5b72b`. These historical features select sources only and never enter the new metric’s feature bank.','',
    'Proposal for Stage B: keep the <1MP cap, Mitchell and native crops; use the measured 32–960 axis with 32–64 views a minority after feasibility/deduplication, and preserve source-family grouping. Keep both labelled XYB controls; keep band shaping as reviewed pilot evidence. Full source census, geometry and split qualification still precede declaration. No full grid or feature/root/training job was launched.','',
    '## Pins, checks and artifacts','',
    'EVIF main commits (ancestry verified): `797753648aa7a7a8e2f1c7a18d5bd060fd2fdaba` raw XYB and bound score import; `29eb4a77936a` preserves the frozen zensim commit while resolving it from zensimx after the repository split; `a78d54901724` transport helpers; `4bd849d93684` native PNG CICP; `a3b5d98a6f8a0f62cb3b98571c390f581e1747df` native AVIF renditions/rectangular crops. No codec dependency revision was changed.','',
    'Worker gates passed: requested XYB param roundtrip, baseline byte parity at qualities 45/75/95 for both B subsamplings, monotone bytes on the textured fixture; external-score media preservation/refusal checks; native PQ16 PNG transport, primary conversion and SVT 444 refusal. Full release all-target tests with `possession`, doctests and clippy all-targets with `possession` passed on i270. Python admission tests and `just lint-scripts` passed. CI status is recorded separately at final reporting; pending CI is not called green.','',
    'run-heavy evidence: main SDR `rc=0 977s | peak-RSS 0.54GiB | min-avail 26576MiB | peak-load 3.44`; HDR encode/native decode/E26 labels `rc=0 168s | peak-RSS 0.24GiB | min-avail 26524MiB | peak-load 1.66`; size cost `rc=0 37s | peak-RSS 0.48GiB | min-avail 26088MiB | peak-load 1.21`; full EVIF gates `rc=0 57s | peak-RSS 1.05GiB | min-avail 23357MiB | peak-load 4.84`; SDR/crop/readback `rc=0 58s | peak-RSS 0.43GiB | min-avail 26624MiB | peak-load 1.37`. Jobs were serialized on i270; no other run’s workers were touched.','',
    'Receipts: `_MANIFEST.json`, `SOURCE_IDENTITIES.json`, `ORACLE.json`, `judge-replay/REPLAY.json`, family `LADDERS.json`/`SUMMARY.json`, `hdr/CONTROL.json`/`ORACLE.json`/`scores.parquet`, `size/TIMING.json`, `SUBSTITUTION.json`, `CLUSTER_REPLAY.json`. Failed orchestration attempts remain preserved and are distinguished from successful receipts. The frozen teacher’s incidental feature outputs are excluded from any Stage C claim.','',
    '## Learnings','',
    '- Container score import must bind both raster identities, and the render container’s wire version matters.\n- A matching library name/version does not establish a matching teacher: EVIF’s current fast-ssim2 differs on all four exact-pixel controls.\n- Unit raw-XYB parity does not equate a table multiplier with the distance API at every rung. Extension adds usable severity on already-admitted sources; early tie rejection remains.\n- The E26 teacher’s native ingress is narrower than general HDR decode dispatch. Exact PQ16 PNG transport preserves its build and viewing conditions.\n- The available HDR chroma-delta range at fixed q95 did not reach the admission margin; unsupported 444 and zero HDR admissions remain explicit.\n- A representative list does not certify cluster membership. The authorized nearest-feature replacement is separately labelled.','',
    '## Per-class admission (main 162-view rerun)','',
    '| Source class | 444 | 420 | capped XYB | raw XYB |','|---|---:|---:|---:|---:|']
    counts={f:defaultdict(lambda:[0,0]) for f in families[:4]}
    for f in families[:4]:
        for l in ladders[f]:
            identity=Path(l['image']).stem.rsplit('_m',1)[0];s=sources[identity]
            label=s['type']+'/'+(s['content_class'] if s['type']=='imazen26' else s['class_group'])
            counts[f][label][1]+=1;counts[f][label][0]+=bool(l['admission']['kept_indices'])
    for label in sorted(counts['ycc444']):lines.append('| '+label+' | '+' | '.join(f"{counts[f][label][0]}/{counts[f][label][1]}" for f in families[:4])+' |')
    lines += ['','## Full pilot ladders','', 'All scores below are the frozen target. Rungs after the first non-drop remain artifacts but are excluded from admission.','']
    for f in families:
        lines += ['### '+f,'','| View | Scores in rung order | Kept (index:score) | Stop |','|---|---|---|---|']
        for l in ladders[f]:
            cells=l['cells'];a=l['admission'];vals='; '.join(f"{c['teacher_ssim2']:.9f}" for c in cells)
            kept='; '.join(f"{i}:{cells[i]['teacher_ssim2']:.9f}" for i in a['kept_indices']) or 'dropped (<3 kept)'
            stop='end of range' if a['stop_index'] is None else str(a['stop_index'])
            lines.append(f"| {Path(l['image']).stem} | {vals} | {kept} | {stop} |")
    lines += ['','### HDR score ladders','']
    for l in hl:lines.append(f"- {l['source']}: "+'; '.join(f"{c['chroma_q'][0]}:{c['teacher_score']:.12f}" for c in l['cells']))
    destination.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('report',type=Path);p.add_argument('--mirror-state',choices=['pending','verified'],required=True);a=p.parse_args();report(a.root,a.report,a.mirror_state)
