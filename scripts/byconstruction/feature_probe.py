"""Probe damage features with pure global edits vs clearly visible damage (label-free).

Exploration record: benchmarks/byconstruction_2026-10-22.md. For each reference (the mixture test's 48 TRAIN-eligible
references) it writes global edits (luma mean shift, contrast scale about the mean, chroma saturation scale) and visible
damage (Gaussian blur, luma noise, chroma noise, blur then luma noise), dumps B's computed Rev5 feature vector for each
(`score_pair_with_bake --dump-features-to`, ids 0..719), and reports per-slot medians for the families asked for.

  python3 feature_probe.py --mixture DIR --bin-dir BIN --bake B.bin --out DIR [--refs 24]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mixture_test import blur, chroma_noise, luma_noise, rgb, to8, ycc  # noqa: E402

V2 = ("ssim_mean ssim_dev2 ssim_dev4 art det mse hf_gain hf_loss hf_mag_loss ssim_soft_peak art_soft_peak "
      "det_soft_peak masked_ssim masked_art masked_det masked_mse iw_ssim iw_art iw_det iw_mse pjnd_transducer "
      "pjnd_fragility gms transducer_lowk transducer_highk blockiness ringing banding edge_width").split()


def edits(ref, seed):
    y, cb, cr = ycc(ref.astype(np.float64))
    my = y.mean()
    return {
        "mean_shift+8": rgb(y + 8, cb, cr),
        "contrast_x0.85": rgb(my + 0.85 * (y - my), cb, cr),
        "saturation_x1.25": rgb(y, 1.25 * cb, 1.25 * cr),
        "saturation_x0.8": rgb(y, 0.8 * cb, 0.8 * cr),
        "blur1.2": blur(ref, 1.2),
        "luma_noise4": luma_noise(ref, 4, seed),
        "chroma_noise8": chroma_noise(ref, 8, seed + 1),
        "blur1.2+luma_noise4": luma_noise(to8(blur(ref, 1.2)), 4, seed + 2),
    }


def slots(name):
    k = V2.index(name)
    return [372 + 29 * cell + k for cell in range(12)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--bake", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", type=int, default=24)
    ap.add_argument("--all-families", action="store_true", help="report every v2 family, not the six of interest")
    ap.add_argument("--reuse", action="store_true", help="reuse dumped feature files already under --out")
    a = ap.parse_args()
    mix, out = Path(a.mixture), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    man = json.loads((mix / "manifest.json").read_text())
    origins = sorted({r["origin"] for r in man["rows"]})[: a.refs]
    feats = {}
    for ri, o in enumerate(origins):
        ref = np.asarray(Image.open(mix / "img" / f"{o}_ref.png").convert("RGB"))
        for name, img in edits(ref, 7000 + 10 * ri).items():
            p = out / "img" / f"{o}_{name}.png"
            dump = out / "img" / f"{o}_{name}.f32"
            if not (a.reuse and dump.is_file()):
                Image.fromarray(to8(img)).save(p)
                subprocess.run([str(Path(a.bin_dir) / "score_pair_with_bake"), "--bake", a.bake, "--bake-post", "raw",
                                "--ref", str(mix / "img" / f"{o}_ref.png"), "--dist", str(p),
                                "--dump-features-to", str(dump)], check=True, capture_output=True)
            b = dump.read_bytes()
            n = int.from_bytes(b[:4], "little")
            feats.setdefault(name, []).append(np.frombuffer(b[8:8 + 4 * n], dtype="<f4").astype(np.float64))
    report = {}
    for fam in (V2 if a.all_families else ("ringing", "iw_ssim", "iw_mse", "hf_loss", "mse", "ssim_mean")):
        ids = slots(fam)
        report[fam] = {name: float(np.median([v[ids].sum() for v in vs])) for name, vs in feats.items()}
    (out / ("probe_all.json" if a.all_families else "probe.json")).write_text(json.dumps({"origins": origins, "bake": a.bake, "median_family_sum": report},
                                               indent=1) + "\n")
    names = list(feats)
    print(f"{'family (sum over 12 cells), median over refs':40s} " + " ".join(f"{n:>20s}" for n in names))
    for fam, row in report.items():
        print(f"{fam:40s} " + " ".join(f"{row[n]:20.4f}" for n in names))
    # within-reference direction: does adding luma noise to the blurred image lower hf_loss?
    hb, hbn = (np.array([v[slots("hf_loss")].sum() for v in feats[k]]) for k in ("blur1.2", "blur1.2+luma_noise4"))
    print(f"hf_loss falls when luma noise is added to the blurred image on {int((hbn < hb).sum())} of {len(hb)} refs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
