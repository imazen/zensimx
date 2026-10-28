"""Read files from a git commit without touching any checkout, and build permalinks.

zenanalyze is read from a named commit (default `origin/main`) rather than from
whatever its working copy holds. zensim links point at the newest pushed
ancestor of the build commit, so line anchors keep pointing at the text that
was read.
"""
from __future__ import annotations

import fnmatch
import hashlib
import re
import subprocess
from pathlib import Path

from .mdparse import SourceShapeError

GITHUB = {"zensim": "https://github.com/imazen/zensimx", "zenanalyze": "https://github.com/imazen/zenanalyze"}


def _git(git_dir: Path, *args: str, binary: bool = False):
    r = subprocess.run(["git", f"--git-dir={git_dir}", *args], capture_output=True, check=False)
    if r.returncode != 0:
        raise SourceShapeError(f"git {' '.join(args)} in {git_dir.name or git_dir}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode()


def git_dir_of(checkout: Path) -> Path:
    """The git directory behind a checkout: `.git` for colocated repos, `jj git root` for secondary jj workspaces."""
    if (checkout / ".git").exists():
        return checkout / ".git"
    try:
        out = subprocess.run(["jj", "git", "root"], cwd=checkout, capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return Path(out)
    except (OSError, subprocess.CalledProcessError):
        pass
    raise SourceShapeError(f"{checkout}: no git directory found")


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class GitTree:
    """The files of one commit: listing, reading, and pathlib-style globbing (no `**`)."""

    def __init__(self, git_dir: Path, rev: str):
        self.git_dir = git_dir
        self.rev = rev
        self.commit = _git(git_dir, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
        self.date = _git(git_dir, "log", "-1", "--format=%cI", self.commit).strip()
        self.blobs: dict[str, str] = {}
        for rec in _git(git_dir, "ls-tree", "-r", "-z", self.commit).split("\0"):
            if not rec:
                continue
            meta, path = rec.split("\t", 1)
            _mode, kind, oid = meta.split()
            if kind == "blob":
                self.blobs[path] = oid
        self.dirs = {p.rsplit("/", i)[0] for p in self.blobs for i in range(1, p.count("/") + 1)}

    def read(self, path: str) -> bytes:
        if path not in self.blobs:
            raise SourceShapeError(f"missing source file {path} at {self.rev} ({self.commit[:8]})")
        return _git(self.git_dir, "cat-file", "blob", self.blobs[path], binary=True)

    def exists(self, path: str) -> bool:
        return path in self.blobs or path.rstrip("/") in self.dirs

    def glob(self, pattern: str) -> list[str]:
        pdir, _, pbase = pattern.rpartition("/")
        out = []
        for p in self.blobs:
            d, _, b = p.rpartition("/")
            if fnmatch.fnmatchcase(d, pdir) and fnmatch.fnmatchcase(b, pbase):
                out.append(p)
        return sorted(out)

    def count_lines(self, path: str) -> int:
        return self.read(path).count(b"\n") + 1

    def behind(self, other_commit: str) -> int | None:
        """How many commits `other_commit` lacks relative to this tree's commit (None if unrelated)."""
        try:
            return int(_git(self.git_dir, "rev-list", "--count", f"{other_commit}..{self.commit}").strip())
        except SourceShapeError:
            return None


def pushed_base(git_dir: Path, head: str, remote_ref: str = "refs/remotes/origin/main") -> str | None:
    """Newest ancestor of `head` that is already on the remote default branch."""
    try:
        return _git(git_dir, "merge-base", head, remote_ref).strip() or None
    except SourceShapeError:
        return None


class Linker:
    """Maps a cited path (zensim-relative, or `zenanalyze:`-prefixed) and line to a GitHub URL.

    zensim files whose bytes equal the pushed base get a `blob/<base>` permalink.
    Files the base lacks or that differ from it (this lane's own unpushed files,
    local edits) link to `blob/main`, which is where they will be once landed.
    """

    def __init__(self, zensim_base: GitTree | None, read_blobs: dict[str, str], zenanalyze: GitTree | None):
        self.base = zensim_base
        self.read_blobs = read_blobs
        self.za = zenanalyze

    def ref_for(self, path: str) -> tuple[str, str, str]:
        if path.startswith("zenanalyze:"):
            p = path.split(":", 1)[1]
            return "zenanalyze", (self.za.commit if self.za else "main"), p
        p = re.sub(r"^\./", "", path)
        if self.base is not None and p in self.base.blobs:
            local = self.read_blobs.get(p)
            if local is None or local == self.base.blobs[p]:
                return "zensim", self.base.commit, p
        elif self.base is not None and p.rstrip("/") in self.base.dirs:
            return "zensim", self.base.commit, p
        return "zensim", "main", p

    def __call__(self, path: str, line: int | None = None) -> str:
        repo, ref, p = self.ref_for(path)
        return f"{GITHUB[repo]}/blob/{ref}/{p}" + (f"#L{line}" if line else "")
