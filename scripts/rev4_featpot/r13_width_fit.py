"""R13 proportional-width authorization over A's canonical fit and pack owners."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--r13-width', type=int, choices=(4, 16, 32, 64, 256, 512), required=True)
    args, remainder = ap.parse_known_args()
    sys.argv = [sys.argv[0], *remainder]
    if args.r13_width == 4:
        binaries=Path('/var/tmp/byc-width-20261104/w4-bin')
        if not (binaries/'zensim_mlp_train').is_file():
            raise ValueError('W4 requires its separately pinned admission binary')
        os.environ['REV4_V2_BIN_DIR']=str(binaries)
        os.environ['ZEN_PANEL_BIN']=str(binaries/'panel')
    import v2_confirm_fit as owner
    import v2_lodo_mlp as trainer
    import v2_production_pack as packing
    import subjective_training_leg as admission
    canonical_groups = admission.dataset_groups
    canonical_recipe = owner.recipe_of
    canonical_command = trainer.train_command
    canonical_pack = packing.pack_production

    def recipe(spec):
        if args.r13_width == 4:
            return canonical_recipe(spec,r13_w4_registration=admission.R13_W4_REGISTRATION)
        return canonical_recipe(spec)

    def groups(*a, **kw):
        expanded, binding = canonical_groups(*a, **kw)
        if binding is None:
            raise ValueError('R13 requires its dataset authority')
        path, _, cell, selected = binding
        authority = admission.document(path)
        authority['cells'][cell]['r13_width'] = dict(
            registration=admission.R13_W4_REGISTRATION if args.r13_width == 4 else admission.R13_REGISTRATION,
            hidden=args.r13_width, max_pairs=args.r13_width // 4)
        path.write_text(json.dumps(authority, sort_keys=True, separators=(',', ':')) + '\n')
        return expanded, (path, admission.sha(path), cell, selected)

    def command(*a, **kw):
        cmd = canonical_command(*a, **kw)
        expected = {'--hidden': str(args.r13_width), '--max-pairs': str(args.r13_width // 4)}
        if any(flag not in cmd or cmd[cmd.index(flag) + 1] != value for flag, value in expected.items()):
            raise ValueError('R13 width/pair allocation differs')
        if any(flag not in cmd for flag in ('--monotone-cbc', '--nonneg-thresholds')):
            raise ValueError('R13 requires A\'s constrained threshold head')
        return cmd

    def pack(*a, **kw):
        kw['no_prune'] = True
        return canonical_pack(*a, **kw)

    admission.dataset_groups = groups
    owner.recipe_of = recipe
    trainer.recipe_of = recipe
    trainer.train_command = command
    packing.pack_production = pack
    owner.main()


if __name__ == '__main__':
    main()
