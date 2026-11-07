"""Reclaim four inactive Cargo incremental caches, retaining all final binaries."""
import json,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded


def main():
    base=Path('/srv/zen-research/zensim-a-repro-20261106')
    output=Path('/mnt/v/output/zensim/shaping-completion-20261107/RECLAIMED_INCREMENTAL_0752.json')
    if output.exists():raise ValueError('preserve reclamation receipt')
    # The recorded run reclaimed compiler caches on the root volume, not the
    # experiment SSD. Never repeat that as an experiment-space intervention.
    if base.stat().st_dev!=output.parent.stat().st_dev:
        raise ValueError('compiler cache is on a different filesystem from study outputs')
    paths=[base/p for p in ('target-wasm/debug/incremental','target-wasm/wasm32-wasip1/debug/incremental',
                           'target-msrv/debug/incremental','target-perms/debug/incremental')]
    entries=[]
    for path in paths:
        if not (base/path.relative_to(base).parts[0]/'CACHEDIR.TAG').is_file():raise ValueError('Cargo cache marker missing')
        inventory=list(path.rglob('*'))
        if any(p.is_symlink() for p in inventory):raise ValueError('symlink in incremental cache')
        files=[p for p in inventory if p.is_file()]
        if not files or any(p.suffix not in ('.o','.bin','.lock','.rmeta','.partial','.bc') for p in files):
            raise ValueError('unexpected incremental compiler inventory')
        entries.append(dict(path=str(path),files=len(files),logical_bytes=sum(p.stat().st_size for p in files),
                            allocated_bytes=sum(p.stat().st_blocks*512 for p in files)))
    def no_users():
        for path in paths:
            # Exclude only unrelated desktop portal filesystems which lsof
            # cannot stat. The inspected /srv compiler caches are disk-backed.
            result=subprocess.run(['sudo','lsof','-nP','-e','/run/user/107/doc','-e','/run/user/1000/doc','+D',str(path)],capture_output=True,text=True)
            if result.returncode!=1 or result.stdout or result.stderr:
                raise ValueError('open-file check not clean: '+result.stdout+result.stderr)
    no_users()
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),stage='verified_before_removal',paths=entries,
        scope='Cargo incremental compiler intermediates only; all final binaries and experiment artifacts retained',
        initial_refusals=['LLVM pre-lto.bc inventory recognition','unrelated desktop portal lsof warnings'],
        filesystem_free_before_bytes=shutil.disk_usage(base).free)
    with output.open('xb') as stream:stream.write(encoded(receipt))
    no_users()
    for path in paths:shutil.rmtree(path)
    receipt.update(stage='inactive_incremental_removed',filesystem_free_after_bytes=shutil.disk_usage(base).free)
    output.write_bytes(encoded(receipt));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
