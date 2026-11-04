"""R13 proportional-width authorization over A's canonical fit and pack owners."""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--r13-width', type=int, choices=(16, 32, 64, 256, 512), required=True)
    args, remainder = ap.parse_known_args()
    sys.argv = [sys.argv[0], *remainder]
    import v2_confirm_fit as owner
    import v2_lodo_mlp as trainer
    import v2_production_pack as packing
    import subjective_training_leg as admission
    canonical_groups = admission.dataset_groups
    canonical_command = trainer.train_command
    canonical_pack = packing.pack_production

    def groups(*a, **kw):
        expanded, binding = canonical_groups(*a, **kw)
        if binding is None:
            raise ValueError('R13 requires its dataset authority')
        path, _, cell, selected = binding
        authority = admission.document(path)
        authority['cells'][cell]['r13_width'] = dict(
            registration=admission.R13_REGISTRATION,
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
    trainer.train_command = command
    packing.pack_production = pack
    owner.main()


if __name__ == '__main__':
    main()
