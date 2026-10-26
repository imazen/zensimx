"""Linux process/lock evidence shared by experiment progress readers.

Extracted from the AIC continuation supervisor at 87864f94; threaded runners
need every task's child list, since Linux attributes forks to the calling task.
"""
import os
from pathlib import Path


def process_snapshot(pid):
    """Read one Linux process identity; a vanished process is not live evidence."""
    try:
        root = Path(f"/proc/{pid}")
        argv = root.joinpath("cmdline").read_bytes().split(b"\0")
        fields = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, parent_pid=int(fields[1]), state=fields[0],
            start_ticks=int(fields[19]), cpu_ticks=int(fields[11]) + int(fields[12]),
            argv=[a.decode(errors="replace") for a in argv if a])
    except (OSError, ValueError, IndexError):
        return None


def process_tree(pid):
    result, pending, seen = [], [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        value = process_snapshot(current)
        if value is None:
            continue
        result.append(value)
        try:
            for task in Path(f"/proc/{current}/task").iterdir():
                try:
                    pending.extend(int(p) for p in (task/"children").read_text().split())
                except (OSError, ValueError):
                    pass
        except (OSError, ValueError):
            pass
    return result


def compute_lock(pid):
    """Resolve the actual flock inode and whether its holder is a worker ancestor."""
    path = Path.home() / "tmp/heavy.lock"
    try:
        file = path.stat()
        key = (os.major(file.st_dev), os.minor(file.st_dev), file.st_ino)
        holders = []
        for line in Path("/proc/locks").read_text().splitlines():
            fields = line.split()
            if len(fields) < 6 or fields[1] != "FLOCK":
                continue  # In particular, blocked -> waiters do not hold the lock.
            major, minor, inode = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == key:
                holders.append(int(fields[4]))
        ancestors = set()
        while isinstance(pid, int) and pid > 0 and pid not in ancestors:
            ancestors.add(pid)
            parent = process_snapshot(pid)
            if parent is None:
                break
            pid = parent["parent_pid"]
        return dict(path=str(path), holder_pids=holders,
            held_by_worker_ancestor=any(p in ancestors for p in holders))
    except (OSError, ValueError, IndexError) as error:
        return dict(path=str(path), error=str(error), held_by_worker_ancestor=None)



def bound_worker(pid, start_ticks):
    """Reject exited, zombie and recycled process IDs."""
    value = process_snapshot(pid) if type(pid) is int and pid > 0 else None
    return value if value and value["state"] not in ("Z", "X") and value["start_ticks"] == start_ticks else None
