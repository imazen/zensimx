"""Short admission transaction for pending writes; never serializes training."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl,json,os,shutil,time
from pathlib import Path
from shaping_completion import encoded

GIB=1024**3
# Conservative admission allowances, not predictions of measured output sizes.
JOB_ALLOWANCE=GIB
AUXILIARY_ALLOWANCE=3*GIB


def required_free(minimum,pending):
    return minimum+AUXILIARY_ALLOWANCE+JOB_ALLOWANCE*pending


def pending_runs(root):
    # Retain unfinished/dead attempts as reservations until explicitly resolved.
    return {str(p.parent.resolve()) for p in root.glob('runs/*/attempt-*/RUN.json')
            if not (p.parent/'RESULT.json').exists()}


@contextmanager
def reservation(root,dest,minimum,*,bytes_required=JOB_ALLOWANCE):
    if not isinstance(bytes_required,int) or not 0<bytes_required<=JOB_ALLOWANCE:
        raise ValueError('reservation must be positive and within one-job allowance')
    root=Path(root);dest=Path(dest).resolve();key=str(dest)
    directory=root/'storage-admission';directory.mkdir(exist_ok=True)
    ledger=directory/'RESERVATIONS.json';status=directory/(dest.parent.name+'-'+dest.name+'.json')
    while True:
        with (directory/'transaction.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            entries=json.loads(ledger.read_text()) if ledger.exists() else {}
            # Do not silently erase stale reservations: unexpected death needs review.
            budgets={name:JOB_ALLOWANCE for name in pending_runs(root)}
            for name,entry in entries.items():
                budgets[name]=max(budgets.get(name,0),entry.get('job_allowance_bytes',JOB_ALLOWANCE))
            budgets[key]=max(budgets.get(key,0),bytes_required)
            free=shutil.disk_usage(root).free;needed=minimum+AUXILIARY_ALLOWANCE+sum(budgets.values())
            record=dict(run=key,pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),
                        free_bytes=free,required_free_bytes=needed,pending_runs=len(budgets),
                        stage='admitted' if free>=needed else 'waiting_for_archival',
                        job_allowance_bytes=budgets[key],auxiliary_allowance_bytes=AUXILIARY_ALLOWANCE)
            status.write_bytes(encoded(record))
            if free>=needed:
                entries[key]=record;ledger.write_bytes(encoded(entries));break
        # No compute/global lock is held while waiting or training.
        time.sleep(30)
    try:yield
    finally:
        with (directory/'transaction.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            entries=json.loads(ledger.read_text());entries.pop(key,None);ledger.write_bytes(encoded(entries))
