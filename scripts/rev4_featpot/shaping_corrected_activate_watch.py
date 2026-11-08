"""Activate only after all registered native proofs have verified archives."""
import argparse,fcntl,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_corrected_view import activate,CONTROL_CLASSES
from shaping_checkpoint_retention import archive_inventory


def sweep(study,proofs,archives):
    if set(proofs)!=set(CONTROL_CLASSES) or set(archives)!=set(CONTROL_CLASSES):
        raise ValueError('all six explicit proof and archive roots required')
    waiting=[]
    for family in CONTROL_CLASSES:
        root=archives[family];proof=proofs[family]
        if root.resolve()!=root or proof.resolve()!=proof or not proof.is_relative_to(root):
            raise ValueError('proof must be inside its canonical archived evidence root')
        if any(not p.is_file() for p in (proof,root/'SHA256SUMS',root/'MIRROR_VERIFIED.json')):
            waiting.append(family)
    if waiting:return dict(stage='waiting_for_archived_native_compatibility',waiting=waiting,scientific_fits_started=False)
    pins={}
    for family in CONTROL_CLASSES:
        root=archives[family];proof=proofs[family];mirror,inventory=archive_inventory(root)
        relative=proof.relative_to(root).as_posix();pin=sha(proof)
        if inventory.get(relative)!=pin or sha(mirror/relative)!=pin:
            raise ValueError('archived native proof changed: '+family)
        pins[family]=dict(proof_sha256=pin,archive_manifest_sha256=sha(root/'SHA256SUMS'),
                          archive_receipt_sha256=sha(root/'MIRROR_VERIFIED.json'))
    # The original activation owner reproduces all six full trajectories,
    # original-result identities, frozen registrations and matrix boundaries.
    result=activate(study,proofs)
    return dict(stage='corrected_activated',activation=result,archived_proof_identity=pins,
                scientific_fits_started=False,outer_results_read=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--proof',action='append',required=True);p.add_argument('--archive',action='append',required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args();tables=[]
    for values in (a.proof,a.archive):
        table={}
        for item in values:
            family,path=item.split('=',1)
            if family in table:p.error('duplicate compatibility class')
            table[family]=Path(path)
        tables.append(table)
    root=a.study/'corrected_thresholds';status=root/'ACTIVATION_STATUS.json'
    with (root/'ACTIVATION_MONITOR.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'ACTIVATION.json').exists() or (root/'COMPATIBILITY.json').exists():
            raise ValueError('preserve prior activation or partial certification')
        while True:
            identity=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),driver_sha256=sha(__file__))
            try:value=sweep(a.study,*tables)
            except Exception as error:
                status.write_bytes(encoded(dict(identity,stage='failed_preserved_for_reconciliation',error=str(error))))
                raise
            temporary=status.with_suffix('.writing');temporary.write_bytes(encoded(dict(identity,**value)));temporary.replace(status)
            if value['stage']=='corrected_activated' or not a.watch:print(json.dumps(value),flush=True);return
            time.sleep(60)


if __name__=='__main__':main()
