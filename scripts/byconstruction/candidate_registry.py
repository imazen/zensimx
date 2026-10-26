"""The by-construction candidate-bake registry: copy every candidate bake into one durable registry with sha256 and provenance (owner, 2026-10-25: keep candidate
bakes for a later AIC2026 evaluation). Re-runnable: existing entries are verified, never overwritten."""
import hashlib, json, os, shutil, sys
from pathlib import Path
R = Path("/mnt/v/output/zensim/candidates-2026-10-25")
B = "/mnt/v/output/zensim/byconstruction-2026-10-22"
H = os.path.expanduser("~")
# Sources are where each fit wrote its bake on dev; add a row for each new candidate and re-run (existing entries are
# verified, never overwritten). Mirror: tower /mnt/tower/output/zensim/candidates-2026-10-25/ (copy, then sha256sum -c).
C = [
 # name, bake, formula revision, data / recipe, gates summary, note
 ("B-rev5", f"{H}/work/zen/zensim--rev7main/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin", 5, "today's default B (diff+frag-820)", "", "shipped profile B"),
 ("B7", f"{H}/tmp/byc/perf/b_rev7.bin", 7, "B's weights stamped revision 7", "", "byte-identical weights to B-rev5"),
 ("M-rev5-s0", f"{H}/tmp/byc/full/mono-thr-mp32_s0/refit/production-f16.bin", 5, "M recipe mono:thr:mp32, v2e29 root (rev5)", f"{B}/gates-M", ""),
 ("M64-rev5-s0", f"{H}/tmp/byc/full/mono-thr-mp64_s0/refit/production-f16.bin", 5, "M64 recipe mono:thr:mp64, rev5", f"{B}/gates-m64full", ""),
 ("T-rev5-s0", f"{H}/tmp/byc/full/mono-thr_s0/refit/production-f16.bin", 5, "T recipe mono:thr, rev5", f"{B}/gates", ""),
 ("Tga-rev6", f"{H}/tmp/byc/parity/tga_full_rev6.bin", 6, "Tga (E21 + AIC2026 public_train leg), served form", "", "trained on AIC2026 public_train: in-sample there"),
] + [(f"M7-s{s}", f"{H}/tmp/byc/full/mono-thr-mp32-r7_s{s}/refit/production-f16.bin", 7, "M recipe, rev7raw root", f"{B}/gates-M7" if s == 0 else "", "") for s in (0, 1, 2)] \
  + [(f"Mdata-s{s}", f"/var/tmp/datalegs-20261025/rev7data/{'full' if s == 0 else f'full_s{s}'}/refit/production-f16.bin", 7, "M recipe + 11 new datasets' TRAIN legs (rev7data root)", f"{B}/gates-Mdata" if s == 0 else "", "trains on VCL@FER, PieAPP, FG-IQA (validate-only licences)") for s in (0, 1, 2)] \
  + [(f"Mdata64-s{s}", f"/var/tmp/datalegs-20261025/rev7data/full64_s{s}/refit/production-f16.bin", 7, "M64 shape on the Mdata data (rev7data root)", f"{B}/gates-Mdata64" if s == 0 else "", "trains on VCL@FER, PieAPP, FG-IQA (validate-only licences)") for s in (0, 1, 2)] \
  + [(f"Mlic-s{s}", f"/var/tmp/byc-finalists-20261026/Mlic/full_s{s}/refit/production-f16.bin", 7, "M recipe on the Mlic root (Mdata's legs without VCL@FER, PieAPP, FG-IQA; byc-mixes-20261025)", f"{B}/gates-Mlic" if s == 0 else "", "licence-clean under the 2026-10-25 policy") for s in (0, 1, 2)]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R.mkdir(parents=True, exist_ok=True)
rows = []
for name, src, rev, data, gates, note in C:
    d = R / name; d.mkdir(exist_ok=True)
    dst = d / "production-f16.bin"
    h = sha(src)
    if dst.exists():
        assert sha(dst) == h, f"{name}: registry copy differs from {src}"
    else:
        shutil.copy2(src, dst); assert sha(dst) == h
    for side in ("best.bin.spec.json",):
        sp = Path(src).parent / side
        if sp.exists() and not (d / side).exists(): shutil.copy2(sp, d / side)
    rows.append(dict(name=name, sha256=h, bytes=dst.stat().st_size, formula_revision=rev, data=data, source=src,
                     gates=gates, note=note))
(R / "CANDIDATES.json").write_text(json.dumps(rows, indent=1) + "\n")
with open(R / "CANDIDATES.tsv", "w") as f:
    f.write("name\tsha256\tbytes\tformula_revision\tdata\tgates\tnote\tsource\n")
    for r in rows: f.write("\t".join(str(r[k]) for k in ("name","sha256","bytes","formula_revision","data","gates","note","source")) + "\n")
with open(R / "SHA256SUMS", "w") as f:
    for r in rows: f.write(f"{r['sha256']}  {r['name']}/production-f16.bin\n")
print(len(rows), "candidates in", R)
