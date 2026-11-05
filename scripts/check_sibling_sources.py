#!/usr/bin/env python3
"""Check requested Cargo roots for external paths and shared source identity.

Requires Python 3.11+ (tomllib). Cargo resolves each checked root with its
existing lock; this never updates locks or reads source from sibling checkouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = (
    "Cargo.toml",
    "zensim-target/Cargo.toml",
    "zensim-picker-prep/Cargo.toml",
    "zensim-ultrahdr-probe/Cargo.toml",
    "zensim-bench/Cargo.toml",
    "tools/zensim-viz/Cargo.toml",
    "tools/shipgates-target/Cargo.toml",
)
CORE = {
    "zencodec", "zencodec-io", "zencodec-coroutine", "zencodec-media",
    "zencodec-testkit", "zencodec-media-testkit", "zenmeta",
}
PIXELS = {"zenpixels", "zenpixels-convert"}
SIBLINGS = {
    "zenjpeg": "zenjpeg", "zenpng": "zenpng", "zenwebp": "zenwebp",
    "zenavif": "zenavif", "ultrahdr-core": "ultrahdr",
    "ultrahdr-rs": "ultrahdr", "zenanalyze": "zenanalyze",
    "zenanalyze-api": "zenanalyze", "zenpredict": "zenanalyze",
    "zenpredict-bake": "zenanalyze",
}


def check_paths(manifest: Path, value: object, trail: tuple[str, ...] = ()) -> int:
    count = 0
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path" and isinstance(item, str):
                target = (manifest.parent / item).resolve()
                if not target.is_relative_to(ROOT):
                    raise ValueError(f"{manifest.relative_to(ROOT)} {'.'.join(trail)}: external path {item}")
                count += 1
            else:
                count += check_paths(manifest, item, (*trail, key))
    elif isinstance(value, list):
        for item in value:
            count += check_paths(manifest, item, trail)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-only", action="store_true", help="explicit static pre-build check")
    parser.add_argument("--manifest", action="append", choices=WORKSPACES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    record = json.loads((ROOT / "scripts/sibling-cohort.json").read_text())
    revisions = record["revisions"]
    result = {"schema": "sibling-source-check-v1", "static_manifests": [], "workspaces": []}
    for name in record["manifests"]:
        manifest = ROOT / name
        data = tomllib.loads(manifest.read_text())
        count = check_paths(manifest, data)
        result["static_manifests"].append({"manifest": name, "internal_paths": count})
    if not args.paths_only:
        for name in args.manifest or WORKSPACES:
            manifest = ROOT / name
            lock = manifest.parent / "Cargo.lock"
            packages = tomllib.loads(lock.read_text())["package"]
            seen: dict[str, list[dict]] = {}
            for package in packages:
                seen.setdefault(package["name"], []).append(package)
            shared = {}
            expected = {**{n: "zencodec" for n in CORE}, **{n: "zenpixels" for n in PIXELS}, **SIBLINGS}
            for package, repo in expected.items():
                entries = seen.get(package, [])
                if not entries:
                    continue
                if len(entries) != 1:
                    raise ValueError(f"{name}: {package} has {len(entries)} lock sources: {entries}")
                source = entries[0].get("source", "")
                revision = revisions[repo]
                if name == "zensim-ultrahdr-probe/Cargo.toml" and package == "zenpredict":
                    revision = record["public_zensim_F_zenpredict"]
                if not source.startswith("git+") or not source.endswith("#" + revision):
                    raise ValueError(f"{name}: {package} has unexpected source {source}")
                shared[package] = source
            try:
                proc = subprocess.run(
                    ["cargo", "metadata", "--locked", "--all-features", "--format-version", "1", "--manifest-path", str(manifest)],
                    check=True, capture_output=True, text=True, cwd=ROOT,
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"{name}: Cargo metadata failed:\n{error.stderr}") from error
            metadata = json.loads(proc.stdout)
            for package in metadata["packages"]:
                if package["source"] is None:
                    path = Path(package["manifest_path"]).resolve()
                    if not path.is_relative_to(ROOT):
                        raise ValueError(f"{name}: local package outside this repository: {path}")
            result["workspaces"].append({"manifest": name, "shared_sources": shared, "packages": len(metadata["packages"])})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
