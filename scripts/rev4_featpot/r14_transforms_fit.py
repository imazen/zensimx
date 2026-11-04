"""R14 transform-table authority over A's unchanged canonical fit/pack owners."""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--r14-arm',choices=('T1a','T1b','T1c','T2'),required=True)
    ap.add_argument('--r14-table',type=Path,required=True)
    args,remainder=ap.parse_known_args()
    sys.argv=[sys.argv[0],*remainder]
    import v2_confirm_fit as owner
    import v2_lodo_mlp as trainer
    import v2_production_pack as packing
    import subjective_training_leg as admission
    original_groups=admission.dataset_groups
    original_command=trainer.train_command
    original_pack=packing.pack_production
    table_sha=admission.sha(args.r14_table)

    def groups(*a,**kw):
        expanded,binding=original_groups(*a,**kw)
        if binding is None:raise ValueError('R14 requires dataset authority')
        path,_,cell,selected=binding
        authority=admission.document(path)
        authority['cells'][cell]['r14_transforms']=dict(registration=admission.R14_REGISTRATION,
            arm=args.r14_arm,table_sha256=table_sha)
        path.write_text(json.dumps(authority,sort_keys=True,separators=(',',':'))+'\n')
        return expanded,(path,admission.sha(path),cell,selected)

    def command(*a,**kw):
        if admission.sha(args.r14_table)!=table_sha:raise ValueError('R14 frozen screen table changed')
        cmd=original_command(*a,**kw)
        expected={'--hidden':'128','--max-pairs':'32'}
        if any(flag not in cmd or cmd[cmd.index(flag)+1]!=value for flag,value in expected.items()):
            raise ValueError('R14 must preserve A head')
        cmd+=['--r14-transforms-registration',admission.R14_REGISTRATION]
        if '--init-from' not in cmd:
            cmd+=['--auto-transforms',str(args.r14_table),'--auto-transforms-min-lift','0.005']
        return cmd

    def pack(*a,**kw):
        kw['no_prune']=True
        return original_pack(*a,**kw)

    admission.dataset_groups=groups
    trainer.train_command=command
    packing.pack_production=pack
    owner.main()


if __name__=='__main__':main()
