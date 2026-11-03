"""E36 revised plan R2 (method review M3): human fit rows chosen by distortion type instead of by dataset.

Writes `--human-rows` keep lists over the design-released human fit table. Each row's distortion type comes from the
E36 bank's keys (`codec`), joined on (member_set, pair_key). Tiers, from the datasets' own distortion lists:
  C  compression: KADID-10k 09 JPEG 2000, 10 JPEG; TID2013 10 JPEG, 11 JPEG 2000, 21 lossy compression of noisy
     images; KonFiG-IQA jpeg2000; every CID22-A25 codec.
  N  artifacts an encoder or its filters produce: KADID 01 Gaussian blur, 06 colour quantization, 15 denoise,
     21 pixelate, 22 quantization; TID2013 07 quantization noise, 08 Gaussian blur, 09 image denoising, 22 colour
     quantization with dither.
  S  everything else (noise, colour shifts, brightness, contrast, jitter, transmission errors, ...).
The trainer draws a reference uniformly and then two of its rows, so every kept reference keeps an equal share; an arm
changes which rows a reference's pairs compare. A reference with no row of the arm's types leaves the leg (listed in the
summary: KonFiG's part-B references hold only motion blur); one left with a single row is refused.

    m3_rows.py HUMAN_FIT_TABLE BANK_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TIER_C = {'kadid_09', 'kadid_10', 'tid_10', 'tid_11', 'tid_21', 'jpeg2000',
          'aom', 'cld_avif', 'cld_heic', 'cld_jp2', 'cld_webp', 'libjxl', 'mozjpeg', 'vis_avif'}
TIER_N = {'kadid_01', 'kadid_06', 'kadid_15', 'kadid_21', 'kadid_22', 'tid_07', 'tid_08', 'tid_09', 'tid_22'}
ARMS = {'m3c': TIER_C, 'm3cn': TIER_C | TIER_N}


def main(table, bank, out):
    keys = pq.read_table(Path(table).with_suffix('.keys.parquet')).to_pandas()
    bk = pd.concat([pq.read_table(Path(bank) / s / 'keys.parquet', columns=['pair_key', 'codec']).to_pandas()
                    .assign(member_set=s) for s in sorted(keys.member_set.unique())])
    j = keys.reset_index(names='row').merge(bk, on=['member_set', 'pair_key'], how='left', validate='many_to_one')
    if j.codec.isna().any() or len(j) != len(keys):
        raise ValueError('every human fit row needs exactly one bank distortion type')
    tier = j.codec.map(lambda c: 'C' if c in TIER_C else 'N' if c in TIER_N else 'S')
    summary = {}
    for arm, kept in ARMS.items():
        m = j.codec.isin(kept)
        per_ref = j[m].groupby(['member_set', 'ref_basename']).size()
        refs = j.groupby(['member_set', 'ref_basename']).size()
        if per_ref.min() < 2:
            raise ValueError(f'{arm}: a reference keeps a single row')
        left = sorted('/'.join(r) for r in refs.index.difference(per_ref.index))
        keep = j.row[m].astype(int).tolist()
        summary[arm] = dict(rows=len(keep), refs=len(per_ref), refs_left=left, min_rows_per_ref=int(per_ref.min()),
                            by_member={k: int(v) for k, v in j[m].member_set.value_counts().sort_index().items()})
        (Path(out) / f'{arm}.json').write_text(json.dumps(dict(
            label=arm, keep=keep, dropped=len(j) - len(keep), rule=f'R2 tiers {sorted(set(tier[m]))}',
            types=sorted(kept), table=str(table))) + '\n')
    summary['tiers'] = {t: int(n) for t, n in tier.value_counts().sort_index().items()}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
