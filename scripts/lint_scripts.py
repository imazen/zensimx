#!/usr/bin/env python3
"""Fail the build on scripts that cannot possibly run.

WHY THIS EXISTS. On 2026-07-15 an audit of `scripts/v_next/` found 25 of 130
scripts pinned to binaries in sibling worktrees that had since been deleted
(`zensim--cross-codec-metric`, `--v10`, `--eval-accel`, `--picker-train`, ...).
They had been unrunnable for weeks. One more (`metric_compare_report.py`) had
not *parsed* since a bulk sed inserted an unescaped `<meta charset="utf-8">`
into a Python string literal — also unnoticed, because nobody ran it.

The mechanism is structural, not careless:

  1. an agent opens sibling worktree `zensim--foo`
  2. it writes a script hardcoding `<zen>/zensim--foo/target/release/bar`
  3. the worktree is cleaned up (correctly! the cleanup rule is mandatory)
  4. the script stays, permanently dead, and nothing notices

(That example is spelled `<zen>/` rather than the literal path because the
2026-07-15 mass-repoint sed — `zensim--*` -> `zensim` across 69 files — rewrote
this very docstring and silently deleted the illustration. Bulk seds do not
read context. That is the same mechanism that left `metric_compare_report.py`
unparseable for weeks.)

CLAUDE.md's worktree-cleanup rule covers the worktree. It does not cover the
scripts that pointed INTO it. This linter closes that gap: a dead reference
fails here instead of rotting for weeks and inflating the directory into
something nobody can read.

    python3 scripts/lint_scripts.py            # report + exit 1 on failure
    python3 scripts/lint_scripts.py --list     # report only, always exit 0

Checks:
  PARSE     — the file is valid Python
  DEAD-WT   — references a `zensim--*` sibling worktree that no longer exists
  DEAD-BIN  — hardcodes a `target/{release,debug}/...` path that does not exist
              AND whose source cannot be found
  DEAD-SCRIPT — shells to a `scripts/...` path that does not exist (this is the
              cross-language edge: a .sh calling a deleted .py)
  DEAD-LINK / DEAD-REF — a README under scripts/ indexes a file that is gone.
              The index is how people find tools; when it rots it sends the
              next session to rebuild something that already exists.
  WIN-PATH  — a tracked filename Windows cannot check out at all
  CONFLICT  — a leaked git/jj conflict-marker line (see check_conflict_markers
              below for the exact forms) in any tracked text file, not just
              scripts/ -- a conflict "resolved" by hand that missed a line
  HYGIENE   — hygiene: address/identifier check. Any tracked text file matching
              a class in scripts/lib/hygiene_patterns.txt, which is the SAME
              file scripts/safe_push.sh reads for its pre-push gate, so the two
              can never drift into checking different things. That gate scans
              the outgoing diff; this scans the whole tracked tree, which is how
              an inherited hit gets found rather than only a newly added one.

A hardcoded build artifact under THIS repo's own `target/` is only a warning:
it may simply not be built yet. A path under a *deleted worktree* is fatal —
that one can never come back.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ZEN = ROOT.parent

PATH_RE = re.compile(r'["\'](/(?:home|mnt)/[^"\'\s]{6,})["\']')
WT_RE = re.compile(r"(zensim--[a-z0-9][a-z0-9-]*)")
# A repo-relative script path, however it is reached: `python3 scripts/x.py`,
# a bare `scripts/x.py`, or one embedded in a longer command.
SCRIPT_RE = re.compile(r"(?<![\w/.-])(scripts/[\w/-]+\.(?:py|sh))")

# Exact (caller, literal) -> (repository, frozen revision, reason).
_V40_POSTFIT = "/".join(("scripts", "jobsys", "v40_postfit.sh"))
_V40_METRICS = "66c0a961876bc582fb73484ff0cca7f2ffed673b"
CROSS_REPO_SCRIPT_REFS = {
    ("scripts/tests/test_v40_postfit_artifacts.py", _V40_POSTFIT):
        ("zenmetrics", _V40_METRICS, "V40 artifact tests call the frozen postfit owner"),
    ("scripts/tests/v40_r4_prepare.py", _V40_POSTFIT):
        ("zenmetrics", _V40_METRICS, "V40 preparation copies the frozen postfit owner"),
}


def _cross_repo_ref(source: Path | None, literal: str) -> bool | None:
    if source is None:
        return None
    try:
        entry = CROSS_REPO_SCRIPT_REFS.get((source.relative_to(ROOT).as_posix(), literal))
    except ValueError:
        return None
    if entry is None:
        return None
    repo, revision, _reason = entry
    sibling = ZEN / repo
    if not sibling.is_dir():
        return True  # Only these registered references are unverifiable without the sibling.
    try:
        git = ["git", "-C", str(sibling)]
        if (sibling / ".jj").is_dir():
            git_dir = subprocess.check_output(
                ["jj", "git", "root", "--ignore-working-copy"],
                cwd=sibling, text=True, timeout=60,
            ).strip()
            git = ["git", "--git-dir=" + git_dir]
        result = subprocess.run(
            [*git, "cat-file", "-t", f"{revision}:{literal}"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "blob"


def dead_script_refs(text: str, source: Path | None = None) -> list[str]:
    """Repo-relative script paths that no longer exist.

    This check was missing at first, and the gap bit immediately: consolidating
    five `eval_v*_pjnd_check.py` into one broke six shell callers, and the
    linter still said "all runnable" because it only inspected
    `/target/release/` paths. It also surfaced a pre-existing break —
    `v0_20b/bake_v3.py` shelling to `affine_calibrate_znpr_v2.py`, deleted long
    ago. Cross-language edges are exactly where nothing else is looking.
    """
    cross = _cross_repo_ref(source, text)
    if cross is not None:
        return [] if cross else [f"DEAD-SCRIPT missing frozen sibling script: {text}"]
    return sorted(
        {
            f"DEAD-SCRIPT calls missing script: {ref}"
            for ref in SCRIPT_RE.findall(text)
            if not (ROOT / ref).exists()
        }
    )


MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#:]+?)\)")
MD_CODE_REF_RE = re.compile(r"`([\w./-]+\.(?:py|sh))`")


def check_markdown(p: Path) -> list[str]:
    """A README that indexes deleted scripts is worse than no README.

    scripts/v_next/README.md is how anyone finds these tools, and on 2026-07-15
    it still offered `affine_calibrate_znpr_v2.py` ("legacy v2... kept for
    reproducing pre-2026-05-13 bakes") and `verify_bake_srocc.py` as live
    options months after both were deleted. Nothing checks prose, so the index
    rots silently while the code it points at is long gone — and a stale index
    sends the next session to rebuild a tool that already exists, or to hunt for
    one that never will.

    Relative links are checked against the file's own directory; bare
    `code-quoted` script names are checked against that directory too, since
    that is how these tables cite tools.
    """
    fails: list[str] = []
    text = p.read_text(errors="replace")

    # A doc that DECLARES itself a record of the past legitimately names tools
    # that existed then; flagging those forever would make this linter noisy,
    # and a noisy linter gets muted. The exemption is deliberately opt-in and
    # visible: the doc must say so in its header. `HISTORY_v06_rebalance_
    # falsified.md` does ("**Status:** FALSIFIED 2026-05-18"). When this check
    # first ran, `CYCLE_7_DSSIM_COTRAIN_PLAN.md` did NOT — it still read as
    # "authorized" pending work for a hypothesis CLAUDE.md had recorded as
    # falsified two months earlier. That is exactly how a session retries a
    # dead idea, so it got a status header rather than an exemption.
    if re.search(r"\b(falsified|superseded|historical doc|concluded|abandoned)\b",
                 "\n".join(text.splitlines()[:12]), re.I):
        return []

    for target in MD_LINK_RE.findall(text):
        t = target.strip()
        if t.startswith(("http://", "https://", "mailto:", "/")):
            continue
        if not (p.parent / t).exists():
            fails.append(f"DEAD-LINK links to missing {t}")
    for name in MD_CODE_REF_RE.findall(text):
        # Only judge names that look like siblings in this directory; a path
        # into another repo or an illustrative name is not ours to resolve.
        if "/" in name:
            continue
        if (p.parent / name).exists():
            continue
        # Prose legitimately names a deleted file to say it IS deleted.
        for line in text.splitlines():
            if name in line and re.search(r"delet|remov|superseded|was\b|no longer|gone", line, re.I):
                break
        else:
            fails.append(f"DEAD-REF names missing script `{name}` without saying it is gone")
    return sorted(set(fails))


def code_strings(tree: ast.AST) -> list[str]:
    """Every string literal that is CODE, not a docstring.

    Docstrings are excluded deliberately: prose legitimately names dead paths
    (this file's own docstring cites `zensim--cross-codec-metric` as the
    cautionary example). Linting prose would make documenting the failure mode
    impossible — the linter would flag the explanation of the linter.
    """
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
    ]


def check(p: Path) -> list[str]:
    src = p.read_text(errors="replace")
    fails: list[str] = []

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"PARSE line {e.lineno}: {e.msg}"]

    literals = code_strings(tree)
    seen: set[str] = set()
    for lit in literals:
        for wt in WT_RE.findall(lit):
            if wt in seen:
                continue
            seen.add(wt)
            if not (ZEN / wt).is_dir():
                fails.append(f"DEAD-WT references deleted sibling worktree {wt}/")

    for lit in literals:
        for raw in PATH_RE.findall(f'"{lit}"'):
            if any(c in raw for c in "{}*?"):
                continue
            if "/target/release/" not in raw and "/target/debug/" not in raw:
                continue
            if Path(raw).exists() or WT_RE.search(raw):
                continue  # worktree case already reported above
            # A missing artifact whose SOURCE still exists is just unbuilt —
            # `cargo build` fixes it. Only an artifact with no source anywhere
            # is a fossil. Failing the buildable case would train people to
            # ignore this linter, which is worse than not having it.
            if unverifiable(raw) or find_source(Path(raw).name):
                continue
            fails.append(f"DEAD-BIN missing binary with no source: {raw}")
    for lit in literals:
        fails += dead_script_refs(lit, p)
    return sorted(set(fails))


def unverifiable(raw: str) -> bool:
    """True when `raw` lives in a sibling repo that is not checked out here.

    A path into `../zenmetrics` or `../zenanalyze` can only be judged when that
    repo is actually present. CI checks out zensim ALONE, so on the runner every
    sibling path resolves to nothing and the DEAD-BIN check fired on 13
    perfectly good `zenmetrics` invocations — the linter passed locally (where
    the siblings exist) and failed in CI. A check whose verdict depends on the
    developer's directory layout is not a check.

    So: verify what is here, and say nothing about what is not. The local run
    still catches sibling-repo rot (that is how the `zen-metrics` -> `zenmetrics`
    rename was found); CI just declines to guess.
    """
    for sib in ("zenanalyze", "zenmetrics"):
        if f"/{sib}/" in raw and not (ZEN / sib).is_dir():
            return True
    return False


def find_source(binary: str) -> Path | None:
    """Locate the source that would build `binary`, anywhere in the zen tree.

    Two ways a binary gets its name, and both must be checked:
      - convention: `src/bin/<name>.rs`
      - declaration: `[[bin]] name = "<name>"` in a Cargo.toml, where the file
        is usually `main.rs` and shares no name with the binary at all.

    Missing the second gave a false DEAD-BIN on `zenmetrics`, which is declared
    that way. A linter that cries wolf on a shipping binary gets muted, and
    then it protects nothing.

    Skips `target/` (build output) and `.claude/` (ephemeral agent copies —
    finding a source only there means it is NOT in the repo).
    """
    def usable(p: Path) -> bool:
        parts = p.parts
        return not ("target" in parts or ".claude" in parts or ".jj" in parts)

    for repo in (ROOT, ZEN / "zenanalyze", ZEN / "zenmetrics"):
        if not repo.is_dir():
            continue
        for cand in repo.rglob(f"{binary}.rs"):
            if usable(cand):
                return cand
        for toml in repo.rglob("Cargo.toml"):
            if not usable(toml):
                continue
            if re.search(
                rf'\[\[bin\]\][^\[]*?name\s*=\s*"{re.escape(binary)}"',
                toml.read_text(errors="replace"),
                re.S,
            ):
                return toml
    return None


def check_shell(p: Path) -> list[str]:
    """Same DEAD-WT / DEAD-BIN checks for shell, minus the AST.

    Shell was NOT covered when this linter first landed, and it reported "all
    runnable" while three `eval_cross_codec_v{6,7,8}.sh` invoked a python
    checker through `zensim--cross-codec-*` worktrees that no longer exist. A
    linter that green-lights dead scripts is worse than no linter: it converts
    "nobody checked" into "something checked and it's fine."

    There is no docstring concept in shell, so `#` comments are stripped
    instead — prose in a comment may legitimately name a dead path.
    """
    fails: list[str] = []
    body = "\n".join(
        line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
        for line in p.read_text(errors="replace").splitlines()
    )
    seen: set[str] = set()
    for wt in WT_RE.findall(body):
        if wt in seen:
            continue
        seen.add(wt)
        if not (ZEN / wt).is_dir():
            fails.append(f"DEAD-WT references deleted sibling worktree {wt}/")
    for raw in re.findall(r"(/(?:home|mnt)/[^\s\"';|)]{6,})", body):
        if any(c in raw for c in "{}*?$"):
            continue
        if "/target/release/" not in raw and "/target/debug/" not in raw:
            continue
        if Path(raw).exists() or WT_RE.search(raw):
            continue
        if unverifiable(raw) or find_source(Path(raw).name):
            continue
        fails.append(f"DEAD-BIN missing binary with no source: {raw}")
    fails += dead_script_refs(body, p)
    return sorted(set(fails))


# Characters Windows refuses in a filename. A tracked file carrying one makes
# `git checkout` fail outright on a Windows runner -- before any compile, so
# the job's log shows only `error: invalid path` and looks like a toolchain
# problem rather than a repo one.
WINDOWS_FORBIDDEN = set('"*:<>?|')


def check_windows_paths() -> list[str]:
    """Tracked paths that Windows cannot check out.

    MEASURED 2026-07-15: a file literally named `"$LOG"` (a shell redirect to
    an unset `$LOG`, committed by accident in 9a4e2272) broke BOTH mandatory
    Windows runners at the checkout step for weeks. CLAUDE.md requires
    windows-11-arm CI, so this is not cosmetic -- it silently removed a
    required platform from coverage.
    """
    files = _tracked_files()
    if files is None:
        return []
    bad = []
    for path in files:
        hits = sorted(set(path) & WINDOWS_FORBIDDEN)
        if hits:
            bad.append(
                f"WIN-PATH: tracked file {path!r} contains {''.join(hits)!r}, "
                "which Windows forbids in a filename -- `git checkout` fails on "
                "the windows runners before anything compiles. Delete it (git "
                "history keeps it) or rename it."
            )
    return bad


# --- Leaked conflict-marker lines ------------------------------------------
#
# Verified against real jj 0.44.0 output (`ui.conflict-marker-style = "diff"`,
# the default) by materializing an actual 2-sided conflict and reading the
# bytes jj wrote into the working-copy file. A jj block looks like this
# (leading "#   " added here only so this comment doesn't trip its own
# check -- the real lines start at column 0 with no indentation at all):
#
#   <<<<<<< conflict 1 of 1
#   %%%%%%% diff from: <rev> <sha> "<desc>"
#   \\\\\\\        to: <rev> <sha> "<desc>"
#   -removed line
#   +added line
#   +++++++ <rev> <sha> "<desc>"
#   kept-as-is content
#   >>>>>>> conflict 1 of 1 ends
#
# A plain git merge conflict is the <<<<<<< / ======= / >>>>>>> subset of
# this; git's diff3/zdiff3 conflict style additionally inserts a |||||||
# common-ancestor line between the open marker and =======. Every one of
# these is a run of ONE repeated character, 7 or more long -- both tools
# widen the run past 7 for a conflict nested inside an already-conflicted
# region, so the patterns below use {7,}, never a fixed {7}.
CONFLICT_OPEN_RE = re.compile(r"^<{7,}(?:\s.*)?$")
CONFLICT_CLOSE_RE = re.compile(r"^>{7,}(?:\s.*)?$")
# None of these six carry a label in real output beyond what's shown above;
# they mean something ONLY between an open and a close marker (see the
# `in_block` gate in find_conflict_markers) -- a bare ======= is a Markdown
# H1 underline and a bare ------- is an hr / table rule, both used all over
# this repo's own docs on purpose, and flagging either unconditionally would
# make this check useless within a day.
_CONFLICT_INNER_RES = (
    re.compile(r"^={7,}\s*$"),          # git `=======`
    re.compile(r"^\|{7,}(?:\s.*)?$"),   # git diff3/zdiff3 `||||||| <label>`
    re.compile(r"^%{7,}(?:\s.*)?$"),     # jj `%%%%%%% diff from: ...`
    re.compile(r"^\+{7,}(?:\s.*)?$"),   # jj `+++++++ <label>`
    re.compile(r"^-{7,}(?:\s.*)?$"),     # jj `-------` / a removed diff line
    re.compile(r"^\\{7,}(?:\s.*)?$"),  # jj's `to:` continuation line
)


def find_conflict_markers(text: str) -> list[tuple[int, str]]:
    """(line, message) for every leaked conflict-marker line in `text`.

    `<<<<<<<` and `>>>>>>>` are flagged UNCONDITIONALLY: there is no
    legitimate reason either shape opens a line of prose, Markdown, code, or
    config, and the 2026-09-05 incident this check exists for was exactly a
    lone trailing `>>>>>>> conflict 1 of 1 ends` with no opening marker left
    anywhere in the file -- a conflict resolved by hand that missed its own
    closing line. The other forms (`=======`, `|||||||`, and jj's
    `%%%%%%%`/`+++++++`/`-------`/backslash-run) are conflict markers only
    BETWEEN an open and a close, so they're gated on being inside that span.
    """
    hits: list[tuple[int, str]] = []
    in_block = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if CONFLICT_OPEN_RE.match(line):
            hits.append((lineno, f"conflict marker: {line.strip()!r}"))
            in_block = True
            continue
        if CONFLICT_CLOSE_RE.match(line):
            hits.append((lineno, f"conflict marker: {line.strip()!r}"))
            in_block = False
            continue
        if in_block and any(pat.match(line) for pat in _CONFLICT_INNER_RES):
            hits.append((lineno, f"conflict marker: {line.strip()!r}"))
    return hits


BINARY_SNIFF_BYTES = 8192


def _tracked_files() -> list[str] | None:
    """Every file tracked in the working copy -- however this checkout
    happens to reach its VCS.

    CI (`actions/checkout`) and the primary jj-colocated checkout both have a
    working `git`, so `git ls-files` is tried first, exactly like
    `check_windows_paths` above. A SECONDARY jj sibling workspace (`jj
    workspace add ../zensim--foo`) has no `.git` at all -- only the
    workspace jj was colocated from keeps the git sidecar -- so `git
    ls-files` there fails outright (nonzero exit, no exception) and this
    falls back to `jj file list`, which every jj workspace of the same repo
    answers identically since they share the underlying operation log.
    A plain source export can explicitly supply `LINT_TRACKED_FILES`, a
    newline-separated snapshot of `jj file list`. Missing, repeated or
    out-of-root paths fail rather than silently narrowing that snapshot.
    Returns None (checked nothing) only when neither VCS nor a manifest exists.
    """
    import subprocess

    manifest = os.environ.get("LINT_TRACKED_FILES")
    if manifest is not None:
        files = Path(manifest).read_text().splitlines()
        if not files or len(files) != len(set(files)):
            raise ValueError("tracked-file manifest must be nonempty and unique")
        for name in files:
            path = Path(name)
            target = ROOT / path
            if (path.is_absolute() or ".." in path.parts or not target.is_file()
                    or not target.resolve().is_relative_to(ROOT.resolve())):
                raise ValueError(f"invalid or missing tracked export path: {name!r}")
        return files

    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=off", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode == 0:
            return [f for f in out.stdout.split("\0") if f]
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        out = subprocess.run(
            ["jj", "file", "list"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode == 0:
            return [f for f in out.stdout.split("\n") if f]
    except (OSError, subprocess.SubprocessError):
        pass

    return None


def _scan_file(fp: Path) -> list[tuple[int, str]]:
    """Binary-sniff + decode + find_conflict_markers for one file on disk.

    Split out from check_conflict_markers so the per-file pipeline is
    testable directly against a real temp file, independent of the repo's
    own tracked-file list (which a test has no business depending on).
    Returns [] for a file that can't be read or looks binary -- same
    "nothing to report" shape as a clean text file, deliberately: neither
    case is a conflict-marker finding.
    """
    try:
        raw = fp.read_bytes()
    except OSError:
        return []
    if b"\0" in raw[:BINARY_SNIFF_BYTES]:
        return []
    text = raw.decode("utf-8", errors="replace")
    return find_conflict_markers(text)


def check_conflict_markers() -> list[str]:
    """Leaked git/jj conflict-marker lines in any tracked text file.

    WHY THIS EXISTS. On 2026-09-05 two lanes each committed Markdown
    (CHANGELOG.md, benchmarks/INDEX.md) carrying a stray trailing
    `>>>>>>> conflict 1 of 1 ends` -- a jj conflict resolved by hand that
    missed its own closing marker line. Both were caught only by a human
    re-reading the file later, and fixed by hand. Nothing had checked for
    this class of defect; this closes the gap the same way DEAD-WT/DEAD-BIN
    close theirs -- scan on every push instead of trusting a resolve to be
    complete.

    Binary files are skipped by a NUL sniff over their first
    BINARY_SNIFF_BYTES (a `.bin` bake, a `.parquet` sidecar, an image -- none
    of these are meant to be read as text, and a coincidental hit inside one
    is noise, not signal). `.orig`/`.rej` files are skipped outright: patch
    and merge tooling legitimately writes literal diff/conflict markers into
    those by design, and nobody intends to commit them clean.

    Unlike the scripts/-only checks above, this scans every tracked path in
    the whole repo -- a leaked marker is exactly as real in a doc or a Rust
    source file as it is in a script.
    """
    files = _tracked_files()
    if not files:
        return []

    fails: list[str] = []
    for rel in files:
        if rel.endswith((".orig", ".rej")):
            continue
        for lineno, msg in _scan_file(ROOT / rel):
            fails.append(f"{rel}:{lineno}: {msg}")
    return fails



HYGIENE_PATTERNS_FILE = SCRIPTS / "lib" / "hygiene_patterns.txt"
HOMEFLEET_NODES_DEFAULT = Path.home() / "work" / "zen" / "homefleet" / "zenmetrics" / "fleet" / "nodes.toml"


def _hygiene_patterns() -> list[tuple[str, "re.Pattern[str]"]]:
    """The shared class patterns, plus the private site-specific ones if the
    homefleet config is readable.

    The public list is deliberately generic -- a pattern file in a public repo
    that spelled out the values it protects would leak exactly what it exists
    to keep out -- so names and labels live in the private config's
    `hygiene_patterns` array instead. A missing config is not an error: the
    generic classes still run. Returns [] only if the shared file itself is
    unreadable, which the caller reports as a failure rather than a pass.
    """
    pats: list[tuple[str, re.Pattern[str]]] = []
    try:
        raw = HYGIENE_PATTERNS_FILE.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "\t" not in line:
            continue
        name, rx = line.split("\t", 1)
        try:
            pats.append((name.strip(), re.compile(rx)))
        except re.error:
            continue

    cfg_path = Path(os.environ.get("HOMEFLEET_NODES") or HOMEFLEET_NODES_DEFAULT).expanduser()
    try:
        import tomllib

        with open(cfg_path, "rb") as f:
            for i, extra in enumerate(tomllib.load(f).get("hygiene_patterns") or []):
                if isinstance(extra, str) and extra:
                    try:
                        pats.append((f"private-{i}", re.compile(extra)))
                    except re.error:
                        continue
    except (OSError, ImportError, ValueError):
        pass
    return pats


def check_address_identifiers() -> list[str]:
    """hygiene: address/identifier check over every tracked text file.

    WHY THIS EXISTS. Global CLAUDE.md forbids household and home-network
    identifiers in a public repo and prescribes grepping the diff before a
    push. Nothing enforced it, so on 2026-09-06 an audit found the class live
    in two public repos -- and the mechanism was not carelessness but COPYING:
    one documented example in a shared doc spread a LAN URL through a dozen
    files in two days. A class that spreads by being read has to be checked by
    a machine, not by remembering.

    Skips `.orig`/`.rej` (patch tooling writes what it likes there), binaries,
    and the pattern file itself -- which necessarily contains regexes
    describing the classes it matches, and a guard that refuses its own
    definition is a guard nobody can edit.
    """
    pats = _hygiene_patterns()
    if not pats:
        return [f"{HYGIENE_PATTERNS_FILE.relative_to(ROOT)}: unreadable or empty — "
                "cannot run the address/identifier check"]

    files = _tracked_files()
    if not files:
        return []

    rel_patterns = str(HYGIENE_PATTERNS_FILE.relative_to(ROOT))
    fails: list[str] = []
    for rel in files:
        if rel.endswith((".orig", ".rej")) or rel == rel_patterns:
            continue
        fp = ROOT / rel
        try:
            raw = fp.read_bytes()
        except OSError:
            continue
        if b"\0" in raw[:BINARY_SNIFF_BYTES]:
            continue
        for lineno, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
            for name, rx in pats:
                if rx.search(line):
                    fails.append(f"{rel}:{lineno}: [{name}] {line.strip()[:120]}")
                    break
    return fails


def main() -> int:
    list_only = "--list" in sys.argv
    failures: dict[str, list[str]] = {}
    n = 0
    for p in sorted(SCRIPTS.rglob("*.py")):
        if "__pycache__" in p.parts or ".venv" in p.parts:
            continue
        n += 1
        f = check(p)
        if f:
            failures[str(p.relative_to(ROOT))] = f
    for p in sorted(SCRIPTS.rglob("*.sh")):
        n += 1
        f = check_shell(p)
        if f:
            failures[str(p.relative_to(ROOT))] = f
    for p in sorted(SCRIPTS.rglob("*.md")):
        n += 1
        f = check_markdown(p)
        if f:
            failures[str(p.relative_to(ROOT))] = f

    win = check_windows_paths()
    if win:
        failures["<tracked paths>"] = win

    conflicts = check_conflict_markers()
    if conflicts:
        failures["<conflict markers>"] = conflicts

    hygiene = check_address_identifiers()
    if hygiene:
        failures["<hygiene: address/identifier check>"] = hygiene

    if not failures:
        print(f"lint_scripts: {n} scripts checked, all runnable")
        return 0

    for name, why in sorted(failures.items()):
        print(f"\n{name}")
        for w in why:
            print(f"    {w}")
    print(
        f"\nlint_scripts: {len(failures)} of {n} scripts cannot run.\n"
        "A script pinned to a deleted worktree is not a script, it is a fossil.\n"
        "Delete it (git history keeps it), or repoint it at a path that exists."
    )
    return 0 if list_only else 1


if __name__ == "__main__":
    sys.exit(main())
