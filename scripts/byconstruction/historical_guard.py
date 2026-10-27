"""Fail closed before dated research entry points perform work.

Frozen as-run copies remain immutable. This guard governs new invocations of
current-checkout launchers, not supported old models or regression tests.
"""
from pathlib import Path
import socket
import sys

ENTRIES = {
    "serve_parity.py": "silently restamps research bakes to Rev6",
    "candidate_registry.py": "dated registry with obsolete checkout/model paths",
    "perf_grid.sh": "historical Rev5/Rev6 performance grid",
    "perf_bits.sh": "historical B/Tga model defaults",
    "map_study.sh": "fixed historical map models and output directory",
    "jxl_steer_rd.sh": "historical models, placements and judge protocol",
    "jxl_reach.sh": "historical Rev6 reachability study",
    "dataset_loop.sh": "historical Mdata loop and fixed control artifacts",
}
BENCHMARKS = {"perf_grid.sh", "map_study.sh", "jxl_steer_rd.sh", "jxl_reach.sh", "dataset_loop.sh"}


def require_replay(argv, entry):
    """Consume an explicit first argument; refuse before any experiment I/O."""
    if entry not in ENTRIES:
        raise ValueError(f"unregistered historical entry point: {entry}")
    if not argv or argv[0] != "--historical-replay":
        raise ValueError(
            f"OBSOLETE ENTRY POINT: {entry}: {ENTRIES[entry]}. "
            "No work started. Use the registered current study; see "
            "SESSION-RESUME.md and just mlic-priority-progress. "
            "Intentional historical reproduction requires --historical-replay "
            "as the first argument, frozen inputs and the original science contract.")
    host = socket.gethostname().split(".")[0]
    if host in {"r7900x", "7900x"} or (host == "dev" and entry in BENCHMARKS):
        raise ValueError(f"HOST RESTRICTION: {entry} cannot run on {host}; historical replay does not override host rules")
    return argv[1:]


def guard_or_exit(argv, entry):
    try:
        return require_replay(argv, entry)
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(64) from None


if __name__ == "__main__":
    # Shell launchers retain their positional arguments and shift the one flag.
    if len(sys.argv) < 2:
        raise SystemExit("usage: historical_guard.py ENTRY [--historical-replay ...]")
    guard_or_exit(sys.argv[2:], Path(sys.argv[1]).name)
