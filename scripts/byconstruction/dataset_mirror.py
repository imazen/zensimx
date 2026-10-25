#!/usr/bin/env python3
"""Checksum and mirror completed feature or training artifacts without deletion.

Run under the shared heavy lock and resource wrapper. The caller supplies the
physical array destination and capacity-check mount paths, or a mounted
destination for copying and verification on the client. Existing mirrors
are preserved. No dataset source is modified.
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


def cold_sha(path):
    """Request Linux page-cache eviction before and after a client readback."""
    with path.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--host")
    ap.add_argument("--nfs", action="store_true", help="copy and checksum on this host through a mounted filesystem; no remote processes")
    ap.add_argument("--cold-readback", action="store_true", help="with --nfs on Linux: fsync copies and request page-cache eviction for readback")
    ap.add_argument("--destination", required=True)
    ap.add_argument("--capacity-path", nargs="+")
    ap.add_argument("--artifact-kind", choices=("features", "training"), default="features")
    ap.add_argument(
        "--image",
        default="iqa-organize:5",
        help="resident checksum image; never pulled",
    )
    args = ap.parse_args()
    if args.nfs:
        assert args.host is None, "mounted-filesystem transport never starts remote processes"
    else:
        assert args.host and args.capacity_path and not args.cold_readback
    assert args.root.is_dir() and args.destination.startswith("/")
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
        if args.cold_readback:
            with target.open("rb") as stream:
                os.fsync(stream.fileno())

    for path in files:
        copy(path)
    for name, digest in digests.items():
        assert readback_sha(destination / name) == digest, name
    readbacks = []
    for name in random.SystemRandom().sample(list(digests), 3):
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
    assert not any(
        p.suffix.lower() in {".png", ".jpg", ".bmp", ".tif", ".imijset"} for p in files
    )
    digests = {p.relative_to(args.root).as_posix(): sha(p) for p in files}
    sums.write_text("".join(f"{digest}  {name}\n" for name, digest in digests.items()))
    destination = args.destination.rstrip("/")
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
            with tarfile.open(fileobj=proc.stdin, mode="w|") as archive:
                for path in paths:
                    archive.add(
                        path,
                        arcname=path.relative_to(args.root).as_posix(),
                        recursive=False,
                    )
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
    for name in random.SystemRandom().sample(list(digests), 3):
        result = subprocess.check_output(
            ["ssh", args.host, shlex.join([*checksum_command, name])],
            text=True,
        )
        assert result.split()[0] == digests[name], name
        readbacks.append({"relative_path": name, "sha256": digests[name]})
    receipt = {
        "schema": "dataset-features-mirror-v1" if args.artifact_kind == "features" else "dataset-training-mirror-v1",
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
