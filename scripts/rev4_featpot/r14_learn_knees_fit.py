"""Conditional R14 T2 over the unchanged canonical A fit owners.

The initial table and release decision are TRAIN-side artifacts. This wrapper
adds native knee learning and its authority, without changing the T1 wrapper.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--r14-arm',choices=('T2',),required=True)
    ap.add_argument('--r14-table',type=Path,required=True)
    ap.add_argument('--r14-knee-bounds',type=Path,required=True)
    ap.add_argument('--r14-development',type=Path,required=True)
    args,remainder=ap.parse_known_args()
    import subjective_training_leg as admission
    import v2_lodo_mlp as trainer
    import r14_transforms_fit as fixed
    decision=admission.document(args.r14_development)
    best=decision['best_arm']
    if best not in ('T1a','T1b','T1c'):
        raise ValueError('unknown R14 TRAIN-development winner')
    expected=min(('T1a','T1b','T1c'),key=lambda a:decision['pooled'][f'{a}_s0'])
    if (best!=expected or decision['T2_feasible'] is not True
            or decision['pooled'][f'{best}_s0']>decision['pooled']['F_s0']
            or not decision['trainable_soft_clip_inputs']):
        raise ValueError('R14 T2 release rule not met')
    screen=admission.document(args.r14_knee_bounds.with_name('SUMMARY.json'))
    manifest=admission.document(args.r14_knee_bounds.with_name('_MANIFEST.json'))
    table_sha=admission.sha(args.r14_table)
    if (table_sha!=screen['tables'][best]['sha256']
            or admission.sha(args.r14_knee_bounds)!=manifest['input_records_sha256']):
        raise ValueError('T2 requires the frozen best T1 table and TRAIN bounds')
    binding=dict(registration=admission.R14_REGISTRATION,
        bounds_sha256=admission.sha(args.r14_knee_bounds),
        initial_table_path=str(args.r14_table.resolve()),best_arm=best,
        feasibility_path=str(args.r14_development.resolve()),feasibility_sha256=admission.sha(args.r14_development))
    original_groups=admission.dataset_groups
    original_command=trainer.train_command

    def groups(*a,**kw):
        expanded,authority_binding=original_groups(*a,**kw)
        if authority_binding is None:raise ValueError('T2 requires dataset authority')
        path,_,cell,selected=authority_binding
        authority=admission.document(path)
        authority['cells'][cell]['r14_knee_learning']=binding
        path.write_text(json.dumps(authority,sort_keys=True,separators=(',',':'))+'\n')
        return expanded,(path,admission.sha(path),cell,selected)

    def command(*a,**kw):
        if (admission.sha(args.r14_knee_bounds)!=binding['bounds_sha256']
                or admission.sha(args.r14_development)!=binding['feasibility_sha256']
                or admission.sha(args.r14_table)!=table_sha):
            raise ValueError('T2 frozen preprocessing or release decision changed')
        return original_command(*a,**kw)+['--r14-knee-bounds',str(args.r14_knee_bounds)]

    admission.dataset_groups=groups
    trainer.train_command=command
    sys.argv=[sys.argv[0],'--r14-arm','T2','--r14-table',str(args.r14_table),*remainder]
    fixed.main()


if __name__=='__main__':main()
