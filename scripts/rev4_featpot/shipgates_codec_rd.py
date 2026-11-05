"""Artifact-first adapter for codec-owned JPEG, WebP and AVIF research loops.

This dispatcher does no scoring or statistical inference. Each map provider
shares the same candidate scalar controller with its paired controls. Dense
global-only frontiers do not load a candidate model.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from shipgates import sha, write


def require_supported_gate_adapter(codec):
    if codec == "avif" and os.environ.get("AVIF_ZENSIM_BACKEND") not in ("svt", "zenav1-svt"):
        raise ValueError("AVIF gate cells require the zenav1-svt (svt-rs) backend: set AVIF_ZENSIM_BACKEND=svt so the CQ driver selects the Zenav1Svt lane (external sb-q-scale map seam implemented and validated 2026-10-17; the Zenravif lane remains release-gated diagnostics).")


def store(root, kind, path):
    digest = sha(path)
    directory = root / "artifacts" / kind
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / (digest + path.suffix)
    if not dest.exists():
        tmp = dest.parent / (digest + path.suffix + f".tmp-{os.getpid()}")
        with tmp.open("xb") as out, path.open("rb") as source:
            while block := source.read(1024 * 1024):
                out.write(block)
        try:
            os.link(tmp, dest)
        except FileExistsError:
            pass
        tmp.unlink(missing_ok=True)
    assert sha(dest) == digest, "CAS artifact changed"
    return dict(path=str(dest), sha256=digest, bytes=dest.stat().st_size,
                original=str(path))


def cell(args):
    require_supported_gate_adapter(args.codec)
    from shipgates_png import canonical_pair
    design = json.loads(args.design.read_text())
    source = design["sources"][args.source_id]
    models = design["models"]
    assert sha(source["path"]) == source["sha256"]
    assert sha(args.binary) == design["binary_sha256"]
    assert sha(args.oracle) == design["oracle_sha256"]
    free = os.statvfs(str(Path.home()))
    assert free.f_bavail * free.f_frsize >= 20 * 1024**3, "disk guard: below20GiB"
    args.output.mkdir(parents=True, exist_ok=False)
    probe = args.output / "probe"
    trace = args.output / "trace"
    prefix = {"jpeg": "ZENJPEG_ZQ_", "webp": "ZENWEBP_ZQ_", "avif": "AVIF_ZENSIM_"}[args.codec]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ZENSIM_", "ZENJPEG_ZQ_", "ZENWEBP_ZQ_", "AVIF_"))}
    env.update(ZENSIM_FORMULA_REV="5", ZENSIM_NEIGHBOUR_EXACT="1", RAYON_NUM_THREADS="1")
    if args.codec == "avif":
        command = [str(args.binary), "--out-dir", str(probe)]
        env.update(AVIF_RD_ARTIFACT_ROOT=str(args.output / "native-artifacts"),
                   AVIF_ZENSIM_BACKEND=os.environ.get("AVIF_ZENSIM_BACKEND", "svt"),
                   AVIF_ZENSIM_SPEED="6", AVIF_ZENSIM_SAVE_AVIF="1",
                   AVIF_ZENSIM_EMIT_BEST="1", AVIF_ZENSIM_H3_RULE="zerosum",
                   AVIF_ZENSIM_H3_GAIN="10" if args.mode == "active" else "0",
                   AVIF_ZENSIM_FACTOR_MAX="1.15", AVIF_ZENSIM_MAP_PROVIDER=args.provider,
                   AVIF_ZENSIM_SEED_CQ="125")
    elif args.codec == "jpeg":
        command = [str(args.binary), "--image", source["path"], "--out-dir", str(probe)]
    else:
        command = [str(args.binary), "--image", source["path"], "--candidate-out", str(probe),
                   "--method", "6"]
    if args.quality is not None:
        assert args.model is None and args.mode == "scalar"
        if args.codec == "avif":
            command += ["--image", source["path"], "--global-cq", str(args.quality)]
        else:
            command += ["--quality" if args.codec == "jpeg" else "--global-quality", str(args.quality)]
        model = None
    else:
        model = models[args.model]
        assert sha(model["path"]) == model["sha256"]
        if args.codec == "avif":
            corpus = args.output / "corpus.tsv"
            corpus.write_text(f"{source['path']}\t{source['name']}\t{source['content_class']}\n")
            command += ["--corpus-file", str(corpus), "--bake", model["path"], "--label", "shipgates",
                        "--zensim-targets", str(args.target), "--arms",
                        "baseline" if args.mode == "scalar" else "h3-mag", "--iters", "2"]
        else:
            command += ["--target", str(args.target)]
            command += ["--corrections", "2"] if args.codec == "jpeg" else ["--max-passes", "3", "--max-overshoot", "0"]
            env.update({prefix + "BAKE": model["path"], prefix + "SPATIAL": args.mode,
                        prefix + "TRACE_DIR": str(trace), prefix + "MAP_PROVIDER": args.provider,
                        prefix + ("SEED_Q" if args.codec == "jpeg" else "START_Q"): "50"})
    write(args.output / "COMMAND.json", dict(command=command,
          environment={k: v for k, v in env.items() if k.startswith((prefix, "ZENSIM_", "RAYON_"))}))
    with (args.output / "run.log").open("x") as out:
        subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    extension = {"jpeg": "jpg", "webp": "webp", "avif": "avif"}[args.codec]
    if args.codec == "avif" and model is not None:
        with (probe / "target_ab_shipgates.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == 1
        result = dict(bytes=int(rows[0]["bytes"]), full_encodes=int(rows[0]["iters_used"]),
                      native_manifest_row=rows[0])
        selected = list((probe / "decoded").glob("*.avif"))
        assert len(selected) == 1
        encoded = selected[0]
        decoded = encoded.with_suffix(".png")
        result["encoded_sha256"] = sha(encoded)
    else:
        result = json.loads((probe / "result.json").read_text())
        encoded = probe / ("selected." + extension)
        decoded = probe / "selected.png"
    assert sha(encoded) == result["encoded_sha256"]
    retained = [store(args.cas_root, args.codec, encoded)]
    fields = []
    for path in sorted(trace.glob("*")):
        if path.suffix == "." + extension:
            retained.append(store(args.cas_root, args.codec, path))
        else:
            fields.append(store(args.cas_root, args.codec + "-trace", path))
    if args.codec == "avif":
        native_items = [json.loads(line) for line in (args.output / "native-artifacts/index.jsonl").read_text().splitlines()]
        result["actual_completed_encoder_calls"] = sum(item["kind"] == "avif" for item in native_items)
        result["engagement_probe_encoder_calls"] = result["actual_completed_encoder_calls"] - result["full_encodes"]
        assert result["engagement_probe_encoder_calls"] >= 0
        for item in native_items:
            path = Path(item["path"])
            assert sha(path) == item["sha256"]
            saved = store(args.cas_root, item["kind"], path)
            saved["native_metadata"] = item["metadata"]
            (retained if item["kind"] == "avif" else fields).append(saved)
        measurements = []
        if model is not None:
            with (probe / "trace_shipgates.tsv").open() as stream:
                measurements = list(csv.DictReader(stream, delimiter="\t",fieldnames=[
                    "trace_id","iter","cq","realized_qindex","score","bytes","sb_min","sb_max","iter_ms"]))
            assert len(measurements) == result["full_encodes"]
            assert [int(r["iter"]) for r in measurements] == list(range(result["full_encodes"]))
            for row in measurements:
                name,content_class,target,arm=row["trace_id"].split("|")[1:]
                assert name==source["name"] and content_class==source["content_class"]
                assert float(target)==args.target and arm==("baseline" if args.mode=="scalar" else "h3-mag")
            for path in probe.glob("trace_*"):
                fields.append(store(args.cas_root, "avif-trace", path))
            if args.mode != "scalar":
                assert any("block-map" in item["original"] for item in fields), "missing native AVIF map"
    elif model is not None:
        assert trace.is_dir() and (trace / "measurements.tsv").is_file()
        with (trace / "measurements.tsv").open() as stream:
            measurements = list(csv.DictReader(stream, delimiter="\t"))
        assert measurements and len(measurements) == result["full_encodes"]
        if args.mode != "scalar":
            assert list(trace.glob("*.map.f32")), "requested maps were not retained"
            if args.provider == "butteraugli":
                assert list(trace.glob("*.butteraugli-pixel.f32")), "missing full peer maps"
    else:
        measurements = []
    # Oracle products remain diagnostics; independent judges use the codec's
    # rendered PNG, whose readback the native driver already verifies.
    oracle_dir = args.output / "oracle"
    oracle_dir.mkdir()
    seen = set()
    for item in retained:
        if item["sha256"] in seen:
            continue
        seen.add(item["sha256"])
        output = oracle_dir / (item["sha256"] + ".ppm")
        if args.codec == "jpeg":
            command = [str(args.oracle), "-rgb", "-outfile", str(output), item["path"]]
        elif args.codec == "webp":
            command = [str(args.oracle), item["path"], "-ppm", "-o", str(output)]
        else:
            command = [str(args.oracle), "-v", "error", "-xerror", "-err_detect", "explode",
                       "-threads", "1", "-c:v", "libdav1d", "-i", item["path"], "-f", "null", "-"]
        with (oracle_dir / (item["sha256"] + ".log")).open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.codec != "avif":
            assert output.is_file() and output.stat().st_size > 0
    png8_binary=Path(design["png8_binary"])
    assert sha(png8_binary)==design["png8_binary_sha256"]
    reference,decoded,png_metadata=canonical_pair(source["path"],source["path"],decoded,args.output,png8_binary)
    record = dict(codec=args.codec, source=source["name"], source_family=source["source_family"],
                  content_class=source["content_class"], source_sha256=source["sha256"],
                  reference=str(reference), decoded=str(decoded), **png_metadata,
                  decoded_png_sha256=sha(decoded),
                  encoded=retained[0]["path"], encoded_sha256=retained[0]["sha256"],
                  bytes=result["bytes"], quality=args.quality, target=args.target,
                  model=args.model, controller_model_sha256=model["sha256"] if model else None,
                  mode=args.mode, map_provider=args.provider if args.mode != "scalar" else None,
                  encoded_artifacts=retained, driver_fields=fields, measurements=measurements,
                  native_result=result, case=str(args.output), independent_decoder="success")
    write(args.output / "CELL.json", record)
    return record


def phase():
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-id", type=int, required=True)
    p.add_argument("--model")
    p.add_argument("--target", type=float)
    args = p.parse_args(sys.argv[2:])
    d = json.loads(args.design.read_text())
    require_supported_gate_adapter(d["codec"])
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    specs = [(f"q{i:02d}", q, None, "scalar", "zensim")
             for i, q in enumerate(d["qualities"])] if args.model is None else [
                 ("scalar", None, args.target, "scalar", "zensim"),
                 ("neutral", None, args.target, "neutral", "zensim"),
                 ("active", None, args.target, "active", "zensim"),
                 ("butteraugli", None, args.target, "active", "butteraugli")]
    for label, quality, target, mode, provider in specs:
        request = argparse.Namespace(codec=d["codec"], binary=Path(d["binary"]),
            oracle=Path(d["oracle"]), design=args.design, output=args.output / label,
            source_id=args.source_id, cas_root=Path(d["cas_root"]),
            model=args.model, mode=mode, provider=provider, quality=quality, target=target)
        row = cell(request)
        row["arm"] = "none" if args.model is None else (
            args.model if label == "active" else args.model + "-" + label)
        row["quality_index"] = label if args.model is None else "t" + str(args.target)
        write(request.output / "GRID_CELL.json", row)
        rows.append(row)
        with (args.output / "cells.jsonl").open("a") as out:
            out.write(json.dumps(row, allow_nan=False) + "\n")
        if len(rows) == 1:
            write(args.output / "FIRST_CELL_VERIFIED.json", row)
        print(f"{row['codec']} {row['source']} {row['arm']} quality={quality} target={target}: {row['bytes']}B; artifacts verified", flush=True)
    if args.model is not None:
        control = {r["arm"]: r["encoded_sha256"] for r in rows}
        assert control[args.model + "-neutral"] == control[args.model + "-scalar"], "neutral/scalar bitstreams differ"
    write(args.output / "COMPLETE.json", dict(cells=len(rows), design_sha256=sha(args.design)))


def matrix():
    p = argparse.ArgumentParser()
    p.add_argument("--codec", choices=["jpeg", "webp", "avif"], required=True)
    for name in ["binary", "oracle", "models", "sources", "output", "heavy-lock", "run-heavy", "png8-binary"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--targets", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--phase-workers", type=int, default=1,
        help="concurrent phases per source; >1 drops per-phase flock — wrap the "
             "whole matrix in ONE run-heavy scope instead (memory-governed)")
    args = p.parse_args(sys.argv[2:])
    admission=json.loads((args.sources.parent/"ADMISSION.json").read_text())
    assert admission["source_manifest_sha256"]==sha(args.sources) and admission["png8_binary_sha256"]==sha(args.png8_binary)
    assert sha(admission["receipt"])==admission["receipt_sha256"]
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sources = json.loads(args.sources.read_text())["sources"]
    models = json.loads(args.models.read_text())
    assert len(sources) == admission["sources"] == len({r["source_family"] for r in sources})
    assert len(sources) == json.loads(Path(admission["receipt"]).read_text())["sources"]
    for record in list(models.values()) + sources:
        assert sha(record["path"]) == record["sha256"]
    qualities = ([1. + 254. * i / 20 for i in range(21)] if args.codec == "avif"
                 else list(range(0, 101, 5)))
    if args.smoke_only:
        qualities = [128 if args.codec == "avif" else 50]
    targets = [70] if args.smoke_only else ([float(t) for t in args.targets.split(",")]
        if args.targets else [25, 40, 55, 70, 80, 90])
    source_ids = ([i for i,r in enumerate(sources) if r["name"]=="imazen-screen-8206"]
                  if args.smoke_only else list(range(len(sources))))
    assert not args.smoke_only or len(source_ids)==1
    design = dict(schema="shipgates-codec-native-rd-v1", codec=args.codec,
        binary=str(args.binary), binary_sha256=sha(args.binary), oracle=str(args.oracle),
        png8_binary=str(args.png8_binary),png8_binary_sha256=sha(args.png8_binary),
        oracle_sha256=sha(args.oracle), models=models, sources=sources, source_ids=source_ids,
        source_manifest_sha256=sha(args.sources), qualities=qualities, targets=targets,
        map_arms=["scalar", "neutral", "active", "butteraugli"],
        neutral_rule="candidate-neutral byte-identical to matching candidate-scalar",
        seed_quality=125 if args.codec == "avif" else 50,
        codec_settings=("speed6,420,8bit,zerosum,gain10,factor1.15,emit-best,threads1,backend=zenav1-svt,sb-q-scale"
            if args.codec == "avif" else "JPEG444,two corrections" if args.codec == "jpeg" else "method6,segments4,three passes"),
        scalar_controller="same candidate for active/neutral/scalar/butteraugli peer; no model on dense global frontier",
        cas_root=str(args.output), smoke_only=args.smoke_only, label_free=True)
    if args.resume and (args.output / "DESIGN.json").is_file():
        prior = json.loads((args.output / "DESIGN.json").read_text())
        assert prior == design, "resume design differs from recorded design"
    else:
        write(args.output / "DESIGN.json", design)
    runner = args.output / "runner.py"
    runner.write_bytes(Path(__file__).read_bytes())
    helper = args.output / "shipgates.py"
    helper.write_bytes(Path(__file__).with_name("shipgates.py").read_bytes())
    png_helper=args.output / "shipgates_png.py"
    png_helper.write_bytes(Path(__file__).with_name("shipgates_png.py").read_bytes())
    runner_pin = args.output / "RUNNER.json"
    runner_sums = {str(p): sha(p) for p in [runner, helper,png_helper]}
    if runner_pin.is_file():
        if json.loads(runner_pin.read_text()) != runner_sums:
            write(args.output / f"RUNNER.updated-{int(time.time())}.json",
                  dict(runner_sums, note="driver updated under resume; per-phase *-command.json records exact argv"))
    else:
        write(runner_pin, runner_sums)
    count = 0
    expected_per_source = len(qualities) + len(models) * len(targets) * 4
    for source_id in source_ids:
        source_dir = args.output / f"source-{source_id:02d}"
        phases = [("global", None, None)] + [
            (model + f"-t{target}", model, target) for model in models for target in targets]
        if args.resume and (source_dir / "COMPLETE.json").is_file():
            done = json.loads((source_dir / "COMPLETE.json").read_text())
            assert done["cells"] == expected_per_source, (
                f"source-{source_id:02d} COMPLETE.json cells {done['cells']} != expected {expected_per_source}")
            count += done["cells"]
            print(f"{args.codec} source{source_id:02d}: resume-complete ({done['cells']} cells)", flush=True)
            continue
        source_dir.mkdir(exist_ok=True)
        source_count = 0
        def run_phase(label, model, target):
            dest = source_dir / label
            if args.resume and (dest / "COMPLETE.json").is_file():
                done = json.loads((dest / "COMPLETE.json").read_text())
                lines = (dest / "cells.jsonl").read_text().splitlines()
                assert len(lines) == done["cells"]
                print(f"{args.codec} source{source_id:02d} {label}: resume-complete", flush=True)
                return len(lines)
            if dest.exists() and not (dest / "COMPLETE.json").is_file():
                dest.rename(dest.with_name(dest.name + f".partial-{int(time.time())}"))
            for stale in [source_dir / (label + "-command.json"), source_dir / (label + ".log")]:
                if stale.exists():
                    stale.rename(stale.with_name(stale.name + f".partial-{int(time.time())}"))
            heavy = [str(args.run_heavy), "--mem", "16G", "--jobs", "1", "--"]
            command = (heavy if args.phase_workers > 1 else
                       ["flock", str(args.heavy_lock)] + heavy) + [
                sys.executable, str(runner), "phase", "--design", str(args.output / "DESIGN.json"),
                "--source-id", str(source_id), "--output", str(dest)]
            if model is not None:
                command += ["--model", model, "--target", str(target)]
            write(source_dir / (label + "-command.json"), command)
            print(f"{args.codec} source{source_id:02d} {label}: queued", flush=True)
            log_path = source_dir / (label + ".log")
            with log_path.open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            done = json.loads((dest / "COMPLETE.json").read_text())
            lines = (dest / "cells.jsonl").read_text().splitlines()
            assert len(lines) == done["cells"]
            print(f"{args.codec} source{source_id:02d} {label}: {len(lines)}cells complete", flush=True)
            return len(lines)
        if args.phase_workers > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(args.phase_workers) as pool:
                for got in pool.map(lambda spec: run_phase(*spec), phases):
                    source_count += got
                    count += got
        else:
            for label, model, target in phases:
                got = run_phase(label, model, target)
                source_count += got
                count += got
        assert source_count == len(qualities) + len(models) * len(targets) * 4
        with (source_dir / "cells.jsonl").open("w") as out:
            for label, _, _ in phases:
                out.write((source_dir / label / "cells.jsonl").read_text())
        write(source_dir / "COMPLETE.json", dict(cells=source_count))
    if not (args.output / "COMPLETE.json").exists():
        write(args.output / "COMPLETE.json", dict(cells=count, sources=len(source_ids), smoke_only=args.smoke_only,
            qualification="encoder-only; independent judges and intervention controls required"))


def main():
    if sys.argv[1:2] == ["matrix"]:
        return matrix()
    if sys.argv[1:2] == ["phase"]:
        return phase()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--codec", required=True, choices=["jpeg", "webp", "avif"])
    for name in ["binary", "oracle", "design", "output", "cas-root"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-id", type=int, required=True)
    p.add_argument("--model")
    p.add_argument("--mode", choices=["scalar", "neutral", "active"], default="scalar")
    p.add_argument("--provider", choices=["zensim", "butteraugli"], default="zensim")
    select = p.add_mutually_exclusive_group(required=True)
    select.add_argument("--quality", type=float)
    select.add_argument("--target", type=float)
    args = p.parse_args()
    cell(args)


if __name__ == "__main__":
    main()
