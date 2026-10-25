#!/usr/bin/env python3
"""A private zenmetrics workbench with the zensimx module (zensim-workbench) beside any modules the base already lists.

  assemble --base DIR --module PKG --manifest JSON --out DIR
      Copy a workbench site (a mirror of the public site with the extra-module hook, e.g. another private workbench),
      put the module beside its worker.js as pkg-zensimx/, and merge the module's manifest entry into the site's
      extra-modules.json (entries of other modules are kept; a previous zensimx entry is replaced). The workbench
      lists extra modules in manifest order; zensimx goes first.
  update --site DIR --module PKG --manifest JSON
      The same module swap in an assembled site, in place: the previous pkg-zensimx/ and extra-modules.json move
      beside the site as <site>.replaced-<UTC time>/.
  smoke-compare NATIVE.jsonl WASM.jsonl
      Compare the native and wasm smoke lines (`just byc-workbench-smoke`): every score within 1e-6 and every map sum
      within 1e-4 of the native value (the f32 map kernels may round differently per SIMD tier).
  serve DIR [--host 0.0.0.0] [--port 3348] [--cert PEM --key PEM]
      Serve DIR cross-origin isolated (COOP/COEP), as the workbench needs. Browsers isolate only secure contexts, so
      other machines need HTTPS with a certificate for the host name.

The module embeds its bakes, which may be private (trained on WG1 labels): serve on the LAN only, never publish.
Exploration record: benchmarks/byconstruction_2026-10-22.md ("Serving the top candidate").
"""
import argparse
import json
import shutil
import ssl
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def assemble(a):
    base, out = Path(a.base), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    source = json.loads((base / "site-source.json").read_text())
    shutil.copytree(base, out)
    assets = out / source.get("asset_prefix", "")
    if "extra-modules.json" not in (assets / "worker.js").read_text():
        raise SystemExit("this worker.js has no extra-module hook")
    shutil.copytree(a.module, assets / "pkg-zensimx")
    modules = merge(assets, a.manifest)
    print(f"{out}: modules {[m['name'] for m in modules]}")


def merge(assets, manifest):
    """Put the module's manifest entry first in the site's extra-modules.json, keeping other modules' entries."""
    mine = json.loads(Path(manifest).read_text())["modules"]
    existing = assets / "extra-modules.json"
    modules = json.loads(existing.read_text())["modules"] if existing.exists() else []
    modules = mine + [m for m in modules if m.get("name") != "zensimx"]
    existing.write_text(json.dumps({"modules": modules}, indent=2) + "\n")
    return modules


def update(a):
    site = Path(a.site)
    source = json.loads((site / "site-source.json").read_text())
    assets = site / source.get("asset_prefix", "")
    if "extra-modules.json" not in (assets / "worker.js").read_text():
        raise SystemExit("this worker.js has no extra-module hook")
    kept = site.parent / f"{site.name}.replaced-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    kept.mkdir()
    for name in ("pkg-zensimx", "extra-modules.json"):
        if (assets / name).exists():
            shutil.copytree(assets / name, kept / name) if (assets / name).is_dir() else shutil.copy2(assets / name, kept)
    shutil.rmtree(assets / "pkg-zensimx", ignore_errors=True)
    shutil.copytree(a.module, assets / "pkg-zensimx")
    modules = merge(assets, a.manifest)
    print(f"{site}: modules {[m['name'] for m in modules]}; previous files in {kept}")


class Isolated(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript",
                      ".mjs": "text/javascript", ".json": "application/json"}


def serve(a):
    httpd = ThreadingHTTPServer((a.host, a.port), partial(Isolated, directory=a.dir))
    scheme = "http"
    if a.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.cert, a.key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    print(f"serving {a.dir} on {scheme}://{a.host}:{a.port}/", flush=True)
    httpd.serve_forever()


def smoke_compare(a):
    native = [json.loads(line) for line in Path(a.native).read_text().splitlines()]
    wasm = {r["metric"]: r for r in (json.loads(line) for line in Path(a.wasm).read_text().splitlines())}
    bad = []
    for n in native:
        m = n["metric"]
        w = wasm.get(m)
        if w is None:
            bad.append(f"{m}: missing from wasm")
            continue
        for v in n["score"]["values"]:
            wv = next((x["value"] for x in w["score"]["values"] if x["column"] == v["column"]), None)
            if wv is None or abs(wv - v["value"]) > 1e-6 * max(1.0, abs(v["value"])):
                bad.append(f"{m} {v['column']}: native {v['value']} wasm {wv}")
        for name, s in n["maps"].items():
            ws = w["maps"].get(name)
            if ws is None or abs(ws - s) > 1e-4 * max(1.0, abs(s)):
                bad.append(f"{m} map {name}: native {s} wasm {ws}")
        print(m, "score", [v["value"] for v in n["score"]["values"]], "maps", n["maps"])
    if bad:
        raise SystemExit("\n".join(bad))
    print(f"{len(native)} metrics agree")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("assemble")
    p.add_argument("--base", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("update")
    p.add_argument("--site", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--manifest", required=True)
    p = sub.add_parser("smoke-compare")
    p.add_argument("native")
    p.add_argument("wasm")
    p = sub.add_parser("serve")
    p.add_argument("dir")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=3348)
    p.add_argument("--cert")
    p.add_argument("--key")
    a = ap.parse_args()
    {"assemble": assemble, "update": update, "serve": serve, "smoke-compare": smoke_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
