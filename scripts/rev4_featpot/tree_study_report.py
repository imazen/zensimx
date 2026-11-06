#!/usr/bin/env python3
"""Exact-key peer/source joins and canonical panels for the registered tree study."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tree_study import load_data, panels, sha, write


def unique_index(frame):
    result = frame.set_index('pair_key')
    if not result.index.is_unique:
        raise ValueError('duplicate pair key')
    return result


def metadata(base_path, run, output):
    base = json.loads(base_path.read_text())
    prepared = json.loads((Path(base['output']) / 'PREPARED.json').read_text())
    plan = json.loads((run / 'PLAN.json').read_text())
    if sha(base_path) != plan['baseline_plan_sha256']:
        raise ValueError('plan mismatch')
    data = load_data(run / 'data')
    frames, pins, bank_cache, peer_cache = [], {}, {}, {}
    bank_root = Path('/mnt/v/output/zensim/byconstruction-2026-10-22/rev7raw/bank')
    peer_root = Path('/var/tmp/rev4-featbank-r4/peer_gmsd')
    for g in prepared['groups']:
        path = Path(g['path']).with_suffix('.keys.parquet')
        if sha(path) != g['keys_sha256']:
            raise ValueError('input key drift')
        keys = pq.read_table(path).to_pandas()
        keys['position'] = np.arange(len(keys))
        chunks = []
        for member in keys.member_set.unique():
            if member not in bank_cache:
                bp = bank_root / member / 'keys.parquet'
                if str(bp) not in base['inputs'] or sha(bp) != base['inputs'][str(bp)]['sha256']:
                    raise ValueError('unregistered bank keys: ' + str(bp))
                pins[str(bp)] = sha(bp)
                bank_cache[member] = unique_index(pq.read_table(bp).to_pandas())
                pp = peer_root / (member + '.parquet')
                pins[str(pp)] = sha(pp)
                peer = pq.read_table(pp).to_pandas()
                if peer.groupby('pair_key')[['gmsd', 'gmsm', 'ref_sha_new', 'dist_sha_new', 'digest_match']].nunique().max().max() > 1:
                    raise ValueError('ambiguous duplicate peer scores')
                peer_cache[member] = unique_index(peer.drop_duplicates('pair_key'))
            selected = keys[keys.member_set == member]
            b = bank_cache[member].loc[selected.pair_key].reset_index()
            p = peer_cache[member].loc[selected.pair_key].reset_index()
            if not np.array_equal(b.ref_pixels_sha256, p.ref_sha_new) or not np.array_equal(b.dist_pixels_sha256, p.dist_sha_new) or not (p.digest_match == 1).all():
                raise ValueError('peer pixel digest mismatch')
            b['position'] = selected.position.to_numpy()
            b['gmsd_quality'] = -p.gmsd.to_numpy()
            b['gmsm'] = p.gmsm.to_numpy()
            chunks.append(b)
        frame = pd.concat(chunks).sort_values('position')
        if frame.pair_key.tolist() != keys.pair_key.tolist():
            raise ValueError('metadata reordered rows')
        frames.append(frame)
    joined = pd.concat(frames, ignore_index=True)
    joined['population'], joined['reference'], joined['development'], joined['fold'] = data['pop'], data['refs'], data['dev'], data['fold']
    # Exact decoded reference identity, independent of filenames and dataset names.
    for digest, group in joined.groupby('ref_pixels_sha256'):
        if group.development.nunique() > 1 or (group[~group.development].fold.nunique() > 1):
            raise ValueError('decoded source crosses split/folds: ' + digest)
    peer_results = {}
    e21 = Path('/mnt/v/output/zensim/e21-peers-20261028')
    for name, file, col, sign in [('ssim2', 'judge_ssim2.tsv', 'ssim2', 1),
                                  ('butter3', 'judge_butteraugli.tsv', 'butteraugli_pnorm3', -1)]:
        path = e21 / file
        pins[str(path)] = sha(path)
        peer = pd.read_csv(path, sep='\t', usecols=['pair_key', 'ref_path', 'dist_path', col])
        if peer.groupby('pair_key')[col].nunique().max() != 1:
            raise ValueError('ambiguous E21 peer scores')
        peer = unique_index(peer.drop_duplicates('pair_key'))
        human = joined.population != 'cid22_ssim2'
        missing = sorted(set(joined.loc[human, 'pair_key']) - set(peer.index))
        if missing:
            raise ValueError(f'{name}: missing {len(missing)} admitted human pairs')
        rows = peer.loc[joined.loc[human, 'pair_key']]
        # Keys bind the original pair; mismatched path spellings need explicit review.
        if rows.ref_path.tolist() != joined.loc[human, 'ref_path'].tolist() or rows.dist_path.tolist() != joined.loc[human, 'dist_path'].tolist():
            raise ValueError('peer key/path inconsistency')
        joined[name] = np.nan
        joined.loc[human, name] = sign * rows[col].to_numpy()
        peer_results[name] = dict(human_rows=int(human.sum()), missing=0)
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pandas(joined, preserve_index=False), output / 'metadata.parquet', compression='zstd')
    summaries = {}
    for name in ('ssim2', 'butter3', 'gmsd_quality', 'gmsm'):
        summaries[name] = {}
        for split, mask in [('fit', ~data['dev']), ('development', data['dev'])]:
            ids = np.flatnonzero(mask & (data['pop'] != 'cid22_ssim2'))
            summaries[name][split] = panels(output / name / split, joined[name].to_numpy()[ids], ids, data, plan['panel_bin'])
    write(output / 'PEERS.json', dict(inputs=pins, metrics=summaries, coverage=peer_results,
        decoded_reference_split_audit=True, source_hashes=joined.ref_pixels_sha256.nunique(),
        metadata_sha256=sha(output / 'metadata.parquet'), peer_labels_used_for_fit=False))
    print(json.dumps(summaries, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-plan', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    metadata(args.baseline_plan, args.run, args.output)


if __name__ == '__main__':
    main()
