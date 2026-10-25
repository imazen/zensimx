"""A steering-map server for steer_rd (zensim-workbench/loop-converge `MapServer` protocol) backed by Margarine,
zenmetrics' Butteraugli approximation (`crates/margarine`, CLI `margarine --diffmap`). Exploration record:
benchmarks/byconstruction_2026-10-22.md ("Margarine as the steering map").

Protocol on stdin/stdout, little-endian: one JSON header line, then per message a tag byte, u32 width, u32 height and
width x height RGBA8. Tag R sets the reference (no reply); tag D scores a distorted image and replies with the score
(f64) and width x height f32 steering values. Each value is Margarine's diffmap placed as `scale * max(v, 0) ** exponent`
(default 1 / 1: Margarine's own map, on Butteraugli's scale as the built-in loop uses Butteraugli's), and the score is
the largest placed value, as the workbench adapter scores a map. Images reach the CLI as 24-bit BMP files in a private
scratch directory; the map file is read back and deleted.

  python3 margarine_map_server.py --margarine BIN [--scale S] [--exponent E] [--scratch DIR] [--threads 1]
  python3 margarine_map_server.py --margarine BIN --check REF.png DIST.png   (same map bits and score as the CLI)
"""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def write_bmp(path: Path, rgba: np.ndarray) -> None:
    """24-bit bottom-up BMP of an opaque RGBA8 image."""
    h, w, _ = rgba.shape
    row = (w * 3 + 3) & ~3
    bgr = np.zeros((h, row), dtype=np.uint8)
    bgr[:, : w * 3] = rgba[::-1, :, 2::-1].reshape(h, w * 3)
    size = 54 + row * h
    header = struct.pack("<2sIHHI", b"BM", size, 0, 0, 54) + struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, row * h,
                                                                           2835, 2835, 0, 0)
    path.write_bytes(header + bgr.tobytes())


def read_exact(stream, n: int) -> bytes:
    buf = stream.read(n)
    if len(buf) != n:
        raise EOFError
    return buf


def check(a, ref_png: str, dist_png: str) -> int:
    """End to end against the CLI: this server (as a child speaking the protocol) and `margarine --diffmap` on the PNGs
    must give the same map bits and the same score, the score equal to the map's largest value."""
    from PIL import Image
    ref = np.asarray(Image.open(ref_png).convert("RGBA"))
    dist = np.asarray(Image.open(dist_png).convert("RGBA"))
    h, w, _ = ref.shape
    child = subprocess.Popen([sys.executable, __file__, "--margarine", a.margarine, "--scratch", a.scratch],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    header = json.loads(child.stdout.readline())
    for tag, img in ((b"R", ref), (b"D", dist)):
        child.stdin.write(tag + struct.pack("<II", w, h) + np.ascontiguousarray(img).tobytes())
    child.stdin.flush()
    score = struct.unpack("<d", read_exact(child.stdout, 8))[0]
    served = np.frombuffer(read_exact(child.stdout, w * h * 4), dtype="<f4")
    child.stdin.close()
    child.wait()
    with tempfile.TemporaryDirectory(prefix="margarine-check-", dir=a.scratch) as td:
        mapf = Path(td) / "map.f32le"
        out = json.loads(subprocess.run([a.margarine, "--diffmap", str(mapf), ref_png, dist_png], check=True,
                                        capture_output=True, text=True).stdout)
        direct = np.fromfile(mapf, dtype="<f4")
    same = direct.size == served.size and np.array_equal(direct.view("<u4"), served.view("<u4"))
    print(json.dumps({"header": header, "pixels": int(w * h), "map_bits_equal": bool(same), "score_server": score,
                      "score_cli": out["score"], "map_max": float(direct.max())}))
    return 0 if same and score == float(direct.max()) and abs(score - out["score"]) <= 1e-6 * max(1.0, score) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margarine", required=True, help="the margarine CLI (just margarine-build in zenmetrics)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--exponent", type=float, default=1.0)
    ap.add_argument("--scratch", default=str(Path.home() / "tmp"))
    ap.add_argument("--threads", type=int, default=1, help="RAYON_NUM_THREADS for each Margarine call (steer_rd runs one "
                    "server per worker)")
    ap.add_argument("--check", nargs=2, metavar=("REF_PNG", "DIST_PNG"), help="end-to-end check against the CLI")
    a = ap.parse_args()
    if a.check:
        return check(a, *a.check)
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))
    binary = Path(a.margarine).resolve()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    header = {"metric": "margarine", "map": "diffmap", "scale": a.scale, "exponent": a.exponent,
              "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    stdout.write((json.dumps(header) + "\n").encode())
    stdout.flush()
    with tempfile.TemporaryDirectory(prefix="margarine-server-", dir=a.scratch) as td:
        ref, dist, mapf = Path(td) / "ref.bmp", Path(td) / "dist.bmp", Path(td) / "map.f32le"
        have_ref = None
        while True:
            try:
                tag = read_exact(stdin, 1)
            except EOFError:
                return 0
            w, h = struct.unpack("<II", read_exact(stdin, 8))
            rgba = np.frombuffer(read_exact(stdin, w * h * 4), dtype=np.uint8).reshape(h, w, 4)
            if (rgba[:, :, 3] != 255).any():
                raise ValueError("Margarine needs opaque images")
            if tag == b"R":
                write_bmp(ref, rgba)
                have_ref = (w, h)
                continue
            if tag != b"D":
                raise ValueError(f"unknown tag {tag!r}")
            if have_ref != (w, h):
                raise ValueError(f"distorted {w}x{h} without a matching reference {have_ref}")
            write_bmp(dist, rgba)
            mapf.unlink(missing_ok=True)
            p = subprocess.run([str(binary), "--diffmap", str(mapf), str(ref), str(dist)], capture_output=True,
                               text=True, env=env)
            if p.returncode != 0:
                raise RuntimeError(f"margarine failed: {p.stderr.strip()}")
            out = json.loads(p.stdout)
            if (out["width"], out["height"]) != (w, h):
                raise ValueError(f"margarine map {out['width']}x{out['height']} for a {w}x{h} image")
            values = np.fromfile(mapf, dtype="<f4")
            if values.size != w * h or not np.isfinite(values).all():
                raise ValueError("margarine map size or values invalid")
            placed = (a.scale * np.maximum(values.astype(np.float64), 0.0) ** a.exponent).astype("<f4")
            score = float(placed.max())
            stdout.write(struct.pack("<d", score) + placed.tobytes())
            stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
