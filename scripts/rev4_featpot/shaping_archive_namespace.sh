#!/usr/bin/env bash
# Read durable bytes in a private read-only mount namespace with a finite cap.
set -euo pipefail
if [[ ${1-} == --inside ]]; then
    shift
    snapshot_mount_source=$1
    snapshot_mount_target=$2
    snapshot_actor=$3
    shift 3
    mkdir -p "$snapshot_mount_target"
    mount -t nfs -o ro,nosuid,nodev,noexec,vers=4.2 \
        "$snapshot_mount_source" "$snapshot_mount_target"
    exec /usr/sbin/runuser -u "$snapshot_actor" -- /usr/bin/env \
        PATH="$PATH" TMPDIR="${TMPDIR:?}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:?}" \
        RAYON_NUM_THREADS="${RAYON_NUM_THREADS:?}" \
        CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:?}" "$@"
fi
if [[ $# -lt 3 ]]; then
    echo 'usage: shaping_archive_namespace.sh NFS_SOURCE MOUNT_TARGET COMMAND [ARGS...]' >&2
    exit 2
fi
snapshot_source=$1
snapshot_target=$2
shift 2
snapshot_scope=$(awk -F/ '$1 == "0::" {print $NF}' /proc/self/cgroup)
case "$snapshot_scope" in
    run-*.scope) ;;
    *) echo 'run this command inside run-heavy on systemd Linux' >&2; exit 2 ;;
esac
snapshot_memory=$(systemctl --user show "$snapshot_scope" --property=MemoryMax --value)
case "$snapshot_memory" in
    ''|*[!0-9]*) echo 'finite caller MemoryMax required' >&2; exit 2 ;;
esac
snapshot_user=$(id -un)
snapshot_script=$(realpath "$0")
exec sudo -n systemd-run --scope --quiet \
    --property="MemoryMax=$snapshot_memory" --property=MemorySwapMax=0 \
    nice -n19 ionice -c3 unshare --mount --propagation private /usr/bin/env \
    PATH="$PATH" TMPDIR="${TMPDIR:?}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:?}" \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS:?}" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:?}" \
    /bin/bash "$snapshot_script" --inside "$snapshot_source" "$snapshot_target" "$snapshot_user" "$@"
