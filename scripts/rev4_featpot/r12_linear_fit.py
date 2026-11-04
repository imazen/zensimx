"""R12 linear/max-pair head orchestration over the canonical full-data fit and pack owners.

All ordinary fitter arguments pass through. Optional --r12-mask supplies the
registered sparse-family mask in logical (raw + appended product) coordinates.
No objective, sampler, admission or inference arithmetic is reimplemented here.
"""
import argparse
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--r12-mask', type=Path)
    ap.add_argument('--r12-arm', choices=('L1', 'L2', 'L3', 'L4'), required=True)
    args, remainder = ap.parse_known_args()
    sys.argv = [sys.argv[0], *remainder]
    import v2_confirm_fit as owner
    import v2_lodo_mlp as trainer
    import v2_production_pack as packing
    import json
    import subjective_training_leg as admission
    canonical_groups = admission.dataset_groups

    def groups(*a, **kw):
        expanded, binding = canonical_groups(*a, **kw)
        if binding is None:
            raise ValueError('R12 requires a dataset authority')
        path, _, cell, selected = binding
        authority = admission.document(path)
        authority['cells'][cell]['r12_linear'] = dict(
            registration="339de6b8c24b49403745cacde495ded875a3e83a" if args.r12_arm == "L4" else admission.R12_REGISTRATION, arm=args.r12_arm,
            mask_sha256=admission.sha(args.r12_mask) if args.r12_mask else None)
        path.write_text(json.dumps(authority, sort_keys=True, separators=(',', ':')) + '\n')
        return expanded, (path, admission.sha(path), cell, selected)

    admission.dataset_groups = groups
    canonical_command = trainer.train_command
    canonical_pack = packing.pack_production

    def command(*a, **kw):
        cmd = canonical_command(*a, **kw)
        if cmd[cmd.index('--hidden') + 1] != ('2' if args.r12_arm == 'L4' else '1') or '--monotone-cbc' not in cmd:
            raise ValueError('R12 requires its registered H1/H2 monotone head')
        if any(flag in cmd for flag in (('--monotone-gated-products',) if args.r12_arm == 'L4' else ('--nonneg-thresholds', '--max-pairs', '--monotone-gated-products'))):
            raise ValueError('R12 head contains an unregistered threshold/product option')
        if args.r12_mask:
            cmd += ['--monotone-feature-mask', str(args.r12_mask.resolve()), '--monotone-strict']
        return cmd

    def pack(*a, **kw):
        kw['no_prune'] = True  # derived positions remain bound, including zero coefficients
        return canonical_pack(*a, **kw)

    trainer.train_command = command
    packing.pack_production = pack
    owner.main()


if __name__ == '__main__':
    main()
