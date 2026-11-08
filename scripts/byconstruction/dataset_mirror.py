#!/usr/bin/env python3
"""Checksum and mirror completed feature or training artifacts without deletion.

Run under the resource wrapper with an explicit CPU/RAM budget. The caller supplies the
physical array destination and capacity-check mount paths, or a mounted
destination for copying and verification on the client. Existing mirrors
are preserved. No dataset source is modified. Explicit evaluation snapshots may
include opaque stimulus files; feature/training modes retain their image refusal.
The remote evaluation transport materializes file symlinks and refuses directory
or broken symlinks.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import socket
import subprocess
import tarfile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def readback_names(digests):
    if not digests:
        raise ValueError("an archive must contain at least one artifact")
    return random.SystemRandom().sample(list(digests), min(3, len(digests)))


def cold_sha(path):
    """Request Linux page-cache eviction before and after a client readback."""
    with path.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return digest


def verify_existing(root, destination, output, cold=False, destination_only=False):
    """Re-read both immutable copies without altering either archive."""
    assert root.is_dir() and destination.is_dir()
    assert not output.exists(), "preserve prior verification receipts"
    assert not output.resolve().is_relative_to(root.resolve())
    assert not output.resolve().is_relative_to(destination.resolve())
    manifest = root / "SHA256SUMS"
    pin = sha(manifest)
    assert sha(destination / manifest.name) == pin, "mirror manifest differs"
    digest = cold_sha if cold else sha
    count = total = 0
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert len(expected) == 64 and all(c in "0123456789abcdef" for c in expected)
        for base in ((destination,) if destination_only else (root, destination)):
            path = base / relative
            assert path.is_file() and not path.is_symlink(), str(path)
            assert digest(path) == expected, str(path)
        count += 1
        total += (destination / relative).stat().st_size
        if count % 1000 == 0:
            print(json.dumps({"verified_files": count, "root": str(root)}), flush=True)
    assert count and sha(manifest) == pin and sha(destination / manifest.name) == pin
    result = dict(schema="existing-artifact-verification-v1",
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  root=str(root), destination=str(destination), files=count,
                  bytes_per_copy=total, manifest_sha256=pin,
                  both_copies_verified=not destination_only,
                  destination_verified=True, local_manifest_verified=True,
                  verification_scope='destination artifacts only' if destination_only else 'both artifact copies',
                  cold_readback=cold)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--host")
    ap.add_argument("--nfs", action="store_true", help="copy and checksum on this host through a mounted filesystem; no remote processes")
    ap.add_argument("--cold-readback", action="store_true", help="with --nfs on Linux: fsync copies and request page-cache eviction for readback")
    ap.add_argument("--destination", required=True)
    ap.add_argument("--capacity-path", nargs="+")
    ap.add_argument("--artifact-kind", choices=("features", "training", "evaluation"), default="features")
    ap.add_argument("--verify-only", action="store_true", help="re-read existing local and mounted mirror manifests without copying")
    ap.add_argument("--destination-only", action="store_true", help="with --verify-only: reread every destination artifact, retaining only the local manifest as a pin")
    ap.add_argument("--verification-output", type=Path, help="new receipt outside both immutable archive roots")
    ap.add_argument(
        "--image",
        default="iqa-organize:5",
        help="resident checksum image; never pulled",
    )
    args = ap.parse_args()
    if args.destination_only and not args.verify_only:
        ap.error('--destination-only requires --verify-only')
    if args.nfs:
        assert args.host is None, "mounted-filesystem transport never starts remote processes"
    else:
        assert args.host and args.capacity_path and not args.cold_readback
    assert args.root.is_dir() and args.destination.startswith("/")
    if args.verify_only:
        assert args.nfs and args.verification_output is not None
        return verify_existing(args.root, Path(args.destination), args.verification_output, args.cold_readback, args.destination_only)
    assert args.verification_output is None
    receipt_path = args.root / "MIRROR_VERIFIED.json"
    sums = args.root / "SHA256SUMS"
    assert not receipt_path.exists() and not sums.exists(), (
        "preserve verified snapshots"
    )
    if not args.nfs:
        return remote_main(args, receipt_path, sums)
    destination = Path(args.destination)
    assert not destination.exists(), "preserve existing mirrors"
    assert not destination.resolve().is_relative_to(args.root.resolve())
    assert not args.root.resolve().is_relative_to(destination.resolve())
    files = sorted(p for p in args.root.rglob("*") if p.is_file())
    assert not any(p.is_symlink() for p in args.root.rglob("*"))
    if args.artifact_kind != "evaluation":
        assert not any(p.suffix.lower() in {".png", ".jpg", ".bmp", ".tif", ".imijset"} for p in files)
    total = sum(p.stat().st_size for p in files)
    assert shutil.disk_usage(destination.parent).free > total
    digests = {p.relative_to(args.root).as_posix(): sha(p) for p in files}
    destination.mkdir()
    readback_sha = cold_sha if args.cold_readback else sha

    def copy(path):
        target = destination / path.relative_to(args.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        # copyfile preserves bytes only. Replay helpers also need their original
        # execute bits; leave data-file permissions and special bits unchanged.
        executable = path.stat().st_mode & 0o111
        current = target.stat().st_mode
        if current & 0o111 != executable:
            target.chmod((current & 0o777 & ~0o111) | executable)
        assert target.stat().st_mode & 0o111 == executable, str(target)
        if args.cold_readback:
            with target.open("rb") as stream:
                os.fsync(stream.fileno())

    for path in files:
        copy(path)
    for name, digest in digests.items():
        assert readback_sha(destination / name) == digest, name
    readbacks = []
    for name in readback_names(digests):
        assert readback_sha(destination / name) == digests[name], name
        readbacks.append({"relative_path": name, "sha256": digests[name]})
    receipt = {
        "schema": f"dataset-{args.artifact_kind}-mirror-v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "destination": str(destination),
        "transport": "mounted-filesystem",
        "checksum_compute_host": socket.gethostname(),
        "remote_processes_started": False,
        "artifact_files_verified_before_receipt": len(files),
        "artifact_bytes": total,
        "all_artifact_sha256_verified": True,
        "executable_bits_preserved": True,
        "random_readbacks": readbacks,
        "readback_method": "fsync, POSIX_FADV_DONTNEED and client SHA-256" if args.cold_readback else "client SHA-256",
    }
    if args.artifact_kind == "features":
        receipt["response_tables_opened"] = False
        receipt["readme_example_exposure_receipt"] = "README_EXPOSURE.json"
    else:
        receipt["label_values_interpreted_by_mirror"] = False
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    digests[receipt_path.name] = sha(receipt_path)
    sums.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items())))
    copy(receipt_path)
    copy(sums)
    assert readback_sha(destination / sums.name) == sha(sums)
    assert readback_sha(destination / receipt_path.name) == sha(receipt_path)
    print(json.dumps({"receipt_sha256": sha(receipt_path), "sha256sums_sha256": sha(sums),
                      "files_in_final_inventory": len(digests), **receipt}), flush=True)


def remote_main(args, receipt_path, sums):
    subprocess.run(
        ["ssh", args.host, "df -h " + " ".join(map(shlex.quote, args.capacity_path))],
        check=True,
    )
    subprocess.run(["ssh", args.host, "uptime"], check=True)
    subprocess.run(
        ["ssh", args.host, "docker ps --format '{{.Names}} {{.Image}}'"], check=True
    )
    image_id = subprocess.check_output(
        [
            "ssh",
            args.host,
            shlex.join(
                ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"]
            ),
        ],
        text=True,
    ).strip()
    assert image_id.startswith("sha256:")
    files = sorted(p for p in args.root.rglob("*") if p.is_file())
    if args.artifact_kind == "evaluation":
        assert not any(p.is_symlink() and not p.is_file() for p in args.root.rglob("*")), (
            "evaluation snapshots refuse directory or broken symlinks"
        )
    else:
        assert not any(
            p.suffix.lower() in {".png", ".jpg", ".bmp", ".tif", ".imijset"} for p in files
        )
    digests = {}
    for index, path in enumerate(files, 1):
        digests[path.relative_to(args.root).as_posix()] = sha(path)
        if index % 1000 == 0:
            print(f"checksum inventory {index}/{len(files)}", flush=True)
    sums.write_text("".join(f"{digest}  {name}\n" for name, digest in digests.items()))
    destination = args.destination.rstrip("/")
    if args.artifact_kind == "evaluation":
        subprocess.run(["ssh", args.host, "test ! -e " + shlex.quote(destination)], check=True)
    container = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cpuset-cpus=0-23",
        "--cpu-shares=256",
        "--memory=1g",
        "--blkio-weight=100",
    ]

    def copy(paths):
        command = [
            *container,
            "-i",
            "-v",
            destination + ":/artifacts:rw",
            "--entrypoint=/usr/bin/tar",
            image_id,
            "--extract",
            "--file=-",
            "--directory=/artifacts",
            "--no-same-owner",
            "--backup=numbered",
        ]
        with subprocess.Popen(
            ["ssh", args.host, shlex.join(command)], stdin=subprocess.PIPE
        ) as proc:
            assert proc.stdin is not None
            with tarfile.open(fileobj=proc.stdin, mode="w|", dereference=args.artifact_kind == "evaluation") as archive:
                for index, path in enumerate(paths, 1):
                    archive.add(
                        path,
                        arcname=path.relative_to(args.root).as_posix(),
                        recursive=False,
                    )
                    if index % 1000 == 0:
                        print(f"mirroring {index}/{len(paths)}", flush=True)
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("containerized array copy failed")

    copy([*files, sums])
    checksum_command = [
        *container,
        "-v",
        destination + ":/artifacts:ro",
        "--workdir=/artifacts",
        "--entrypoint=/usr/bin/sha256sum",
        image_id,
    ]
    verify = shlex.join([*checksum_command, "-c", "SHA256SUMS"])
    subprocess.run(["ssh", args.host, verify], check=True)
    readbacks = []
    for name in readback_names(digests):
        result = subprocess.check_output(
            ["ssh", args.host, shlex.join([*checksum_command, name])],
            text=True,
        )
        assert result.split()[0] == digests[name], name
        readbacks.append({"relative_path": name, "sha256": digests[name]})
    receipt = {
        "schema": f"dataset-{args.artifact_kind}-mirror-v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "destination": args.host + ":" + destination,
        "artifact_files_verified_before_receipt": len(files),
        "artifact_bytes": sum(p.stat().st_size for p in files),
        "all_artifact_sha256_verified": True,
        "random_readbacks": readbacks,
        "checksum_container_image_id": image_id,
    }
    if args.artifact_kind == "features":
        receipt["response_tables_opened"] = False
        receipt["readme_example_exposure_receipt"] = "README_EXPOSURE.json"
    else:
        receipt["label_values_interpreted_by_mirror"] = False
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    # Final inventory includes the receipt; no inventory hash in the receipt
    # means there is no circular digest dependency.
    digests[receipt_path.name] = sha(receipt_path)
    sums.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items()))
    )
    copy([receipt_path, sums])
    subprocess.run(["ssh", args.host, verify], check=True)
    print(
        json.dumps(
            {
                "receipt_sha256": sha(receipt_path),
                "sha256sums_sha256": sha(sums),
                "files_in_final_inventory": len(digests),
                **receipt,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
