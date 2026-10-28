"""Where did zensim's git history grow? Walk main's first-parent history oldest first, attribute each newly introduced
blob (by hash, first appearance only) to the commit that introduced it, and report cumulative bytes by commit and date,
the largest jumps, and the largest paths. Read-only: git log/cat-file only.

  python3 git_growth.py REPO OUT_DIR"""
import collections
import json
import subprocess
import sys
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
log = subprocess.run(["git", "-C", str(repo), "log", "--first-parent", "--reverse", "--no-abbrev", "--raw", "--no-renames",
                      "--format=C %H %ct %ad %s", "--date=short", "origin/main"], capture_output=True, text=True, check=True).stdout
commits, cur = [], None
for line in log.splitlines():
    if line.startswith("C "):
        _, sha, ct, date, *subj = line.split(" ")
        cur = dict(sha=sha, ct=int(ct), date=date, subject=" ".join(subj), blobs=[])
        commits.append(cur)
    elif line.startswith(":") and cur is not None:
        meta, path = line.split("\t", 1)
        parts = meta.split()
        new = parts[3]
        if set(new) != {"0"}:
            cur["blobs"].append((new, path))
uniq = {b for c in commits for b, _ in c["blobs"]}
proc = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-check=%(objectname) %(objectsize) %(objectsize:disk)"],
                      input="\n".join(uniq) + "\n", capture_output=True, text=True, check=True).stdout
size, disk = {}, {}
for line in proc.splitlines():
    sha, s, d = line.split()
    size[sha], disk[sha] = int(s), int(d)
seen, cum, cum_disk, rows, by_path = set(), 0, 0, [], collections.Counter()
for c in commits:
    added = added_disk = 0
    for b, path in c["blobs"]:
        if b in seen:
            continue
        seen.add(b)
        added += size.get(b, 0)
        added_disk += disk.get(b, 0)
        by_path[path] += size.get(b, 0)
    cum += added
    cum_disk += added_disk
    rows.append(dict(sha=c["sha"], date=c["date"], subject=c["subject"][:100], added=added, added_disk=added_disk,
                     cum=cum, cum_disk=cum_disk))
out.mkdir(parents=True, exist_ok=True)
(out / "growth.json").write_text(json.dumps(rows))
top_dirs = collections.Counter()
for p, s in by_path.items():
    top_dirs["/".join(p.split("/")[:2])] += s
summary = dict(commits=len(rows), total_bytes=cum, total_disk=cum_disk,
               largest_commits=sorted(rows, key=lambda r: -r["added"])[:25],
               largest_paths=by_path.most_common(40), largest_dirs=top_dirs.most_common(25))
(out / "growth_summary.json").write_text(json.dumps(summary, indent=1))
by_month = collections.OrderedDict()
for r in rows:
    by_month[r["date"][:7]] = (r["cum"], r["cum_disk"])
for m, (c_, d_) in by_month.items():
    print(f"{m}  cumulative new-blob bytes {c_/2**20:9.1f} MiB  (on disk {d_/2**20:8.1f} MiB)")
print("largest commits:")
for r in summary["largest_commits"][:12]:
    print(f"  {r['date']} {r['sha'][:10]} +{r['added']/2**20:7.2f} MiB  {r['subject']}")
print("largest dirs:")
for d, s in summary["largest_dirs"][:15]:
    print(f"  {s/2**20:8.2f} MiB  {d}")
