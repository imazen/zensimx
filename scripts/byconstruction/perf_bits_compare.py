#!/usr/bin/env python3
"""Compare two perf_bits.sh output dirs: every dump must be byte-identical for a bit-identical change; for a
value-changing candidate, report per dump how many values moved and by how much (relative to max(|a|, 1e-30)).

  perf_bits_compare.py BASE_DIR CAND_DIR [--allow-diff]
  perf_bits_compare.py --tiers DIR

Exit status 1 when any dump differs (or is missing) unless --allow-diff. `--tiers` checks one dump set: every
tier and thread count of the same bake and pair must be byte-identical (`<bake>_<pair>_<tier>_t<threads>.hex`). Record: benchmarks/byconstruction_2026-10-22.md.
"""
import argparse
import struct
import sys
from pathlib import Path


def values(path):
    """The dump's values (score first, then each emitted feature), from their hex bit patterns."""
    return [struct.unpack("<d", int(line, 16).to_bytes(8, "little"))[0] for line in path.read_text().split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("cand", nargs="?")
    ap.add_argument("--allow-diff", action="store_true")
    ap.add_argument("--tiers", action="store_true", help="compare tiers within BASE instead of two dirs")
    a = ap.parse_args()
    if a.tiers:
        groups = {}
        for f in sorted(Path(a.base).glob("*.hex")):
            bake, pair = f.stem.split("_")[0], "_".join(f.stem.split("_")[1:-2])
            groups.setdefault((bake, pair), []).append(f)
        bad = 0
        for (bake, pair), files in groups.items():
            ref = files[0].read_bytes()
            for f in files[1:]:
                if f.read_bytes() != ref:
                    va, vb = values(files[0]), values(f)
                    moved = sum(1 for x, y in zip(va, vb) if struct.pack("<d", x) != struct.pack("<d", y))
                    print(f"TIER DIFF {f.name} vs {files[0].name}: {moved} values")
                    bad += 1
        print(f"{len(groups)} bake/pair groups, {sum(len(v) for v in groups.values())} dumps, {bad} tier mismatches")
        sys.exit(1 if bad else 0)
    base, cand = Path(a.base), Path(a.cand)
    names = sorted(p.name for p in base.glob("*.hex"))
    if not names:
        sys.exit(f"no dumps in {base}")
    bad, worst = 0, 0.0
    for n in names:
        c = cand / n
        if not c.exists():
            print(f"MISSING {n}")
            bad += 1
            continue
        if c.read_bytes() == (base / n).read_bytes():
            continue
        va, vb = values(base / n), values(c)
        if len(va) != len(vb):
            print(f"LENGTH {n}: {len(va)} vs {len(vb)}")
            bad += 1
            continue
        moved = [(i, x, y) for i, (x, y) in enumerate(zip(va, vb)) if struct.pack("<d", x) != struct.pack("<d", y)]
        rel = max(abs(y - x) / max(abs(x), 1e-30) for _, x, y in moved)
        worst = max(worst, rel)
        score = " (score moved)" if moved[0][0] == 0 else ""
        print(f"DIFF {n}: {len(moved)} of {len(va)} values, max rel {rel:.3e}{score}, first index {moved[0][0]}")
        bad += 1
    print(f"{len(names) - bad} of {len(names)} dumps identical; worst relative change {worst:.3e}")
    if bad and not a.allow_diff:
        sys.exit(1)


if __name__ == "__main__":
    main()
