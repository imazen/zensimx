#!/usr/bin/env python3
"""Checksum and mirror a completed label-free feature root without deletion.

Run under the shared heavy lock and resource wrapper. The caller supplies the
physical array destination and capacity-check mount paths. Existing files are
backed up, never removed. No dataset source is modified.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--destination", required=True)
    ap.add_argument("--capacity-path", nargs="+", required=True)
    ap.add_argument(
        "--image",
        default="iqa-organize:5",
        help="resident checksum image; never pulled",
    )
    args = ap.parse_args()
    assert args.root.is_dir() and args.destination.startswith("/")
    receipt_path = args.root / "MIRROR_VERIFIED.json"
    sums = args.root / "SHA256SUMS"
    assert not receipt_path.exists() and not sums.exists(), (
        "preserve verified snapshots"
    )
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
    remote = args.host + ":" + destination + "/"
    subprocess.run(
        [
            "rsync",
            "-a",
            "--no-owner",
            "--no-group",
            "--backup",
            "--suffix=.bak",
            "--mkpath",
            str(args.root) + "/",
            remote,
        ],
        check=True,
    )
    checksum_command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cpuset-cpus=0-23",
        "--cpu-shares=256",
        "--memory=1g",
        "--blkio-weight=100",
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
        "schema": "dataset-features-mirror-v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "destination": args.host + ":" + destination,
        "artifact_files_verified_before_receipt": len(files),
        "artifact_bytes": sum(p.stat().st_size for p in files),
        "all_artifact_sha256_verified": True,
        "random_readbacks": readbacks,
        "response_tables_opened": False,
        "readme_example_exposure_receipt": "README_EXPOSURE.json",
        "checksum_container_image_id": image_id,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    # Final inventory includes the receipt; no inventory hash in the receipt
    # means there is no circular digest dependency.
    digests[receipt_path.name] = sha(receipt_path)
    sums.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items()))
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--no-owner",
            "--no-group",
            str(receipt_path),
            str(sums),
            remote,
        ],
        check=True,
    )
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
