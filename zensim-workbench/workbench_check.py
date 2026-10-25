#!/usr/bin/env python3
"""The zensimx module inside the zenmetrics workbench, in headless Chromium.

Serves an assembled site (`scripts/byconstruction/workbench_site.py assemble`) on 127.0.0.1 and, through the
workbench's own worker.js: routes every zensimx metric to the module, scores the smoke pair
(`examples/smoke.rs`, `smoke.mjs`) and reads every map, then compares the score with the native one (within 1e-6)
and each map's cell sum with the native worker-map sum (within 1e-4; the f32 map kernels may round differently per
SIMD tier). Then the page: the metric menu lists the module's metrics in their own group, first among the extra
modules, and a `?metric=` link opens the page on a module metric.

  workbench_check.py SITE NATIVE_SMOKE.jsonl    (needs Playwright with Chromium; `just byc-workbench-check`)

`ZENSIM_CHECK_CHROMIUM` names a Chromium binary to use instead of Playwright's own build.
"""
import json
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

site = Path(sys.argv[1]).resolve()
native = {r["metric"]: r for r in (json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines())}
prefix = json.loads((site / "site-source.json").read_text()).get("asset_prefix", "")
manifest = next(m for m in json.loads((site / prefix / "extra-modules.json").read_text())["modules"]
                if m["name"] == "zensimx")


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *_):
        pass


WORKER = r"""
async ({prefix, metrics}) => {
  const worker = new Worker(new URL(`${prefix}/worker.js`, location.href), {type: 'module'});
  let pending = null, id = 0;
  worker.onmessage = ({data}) => {
    if (['progress', 'module', 'control', 'score'].includes(data.type)) return;
    const resolve = pending; pending = null; resolve(data);
  };
  const call = message => new Promise(resolve => { pending = resolve; worker.postMessage(message); });
  const ready = await call({type: 'init', threads: 2});
  if (ready.type !== 'ready') throw new Error(JSON.stringify(ready));
  const [w, h] = [96, 80];
  const px = (x, y) => { const v = ((x * 7 + y * 13) % 200) + 20; return [v, Math.floor(v / 2) + 40, 255 - v, 255]; };
  const r = new Uint8Array(w * h * 4), d = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 4, a = px(x, y), b = px(Math.min(x + 1, w - 1), y);
    r.set(a, o);
    d.set([Math.floor((a[0] + b[0]) / 2), Math.min(Math.floor((a[1] + b[1]) / 2) + 2, 255), Math.floor((a[2] + b[2]) / 2), 255], o);
  }
  const sum = m => Array.from(m).reduce((s, v) => s + v, 0);
  const out = {};
  for (const metric of metrics) {
    const result = await call({id: ++id, metric, mapMetric: metric, mapName: '', reference: r.slice(), distorted: d.slice(), width: w, height: h});
    if (result.type !== 'result') { out[metric] = {error: JSON.stringify(result)}; continue; }
    const maps = {[result.mapName]: sum(result.map)};
    for (const name of result.mapNames.filter(n => n !== result.mapName)) {
      const m = await call({type: 'map', id: ++id, mapMetric: metric, mapName: name});
      maps[name] = m.map ? sum(m.map) : JSON.stringify(m);
    }
    out[metric] = {module: result.score.module, values: result.score.values, maps};
  }
  worker.terminate();
  return {extras: ready.extras.map(e => e.name), out};
}
"""

server = ThreadingHTTPServer(("127.0.0.1", 3322), partial(Handler, directory=str(site)))
threading.Thread(target=server.serve_forever, daemon=True).start()
metrics = [m["name"] for m in manifest["metrics"]]
diffs = []
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get("ZENSIM_CHECK_CHROMIUM") or None)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("http://127.0.0.1:3322/")
        got = page.evaluate(WORKER, {"prefix": prefix, "metrics": metrics})
        assert "zensimx" in got["extras"], got["extras"]
        for metric in metrics:
            g, n = got["out"][metric], native[metric]
            if "error" in g:
                diffs.append(f"{metric}: {g['error'][:300]}")
                continue
            if g["module"] != "zensimx":
                diffs.append(f"{metric}: routed to {g['module']}")
            for v in n["score"]["values"]:
                gv = next((x["value"] for x in g["values"] if x["column"] == v["column"]), None)
                if gv is None or abs(gv - v["value"]) > 1e-6 * max(1.0, abs(v["value"])):
                    diffs.append(f"{metric} {v['column']}: native {v['value']} browser {gv}")
            for name, s in n["spatial"].items():
                gs = g["maps"].get(name)
                if not isinstance(gs, (int, float)) or abs(gs - s) > 1e-4 * max(1.0, abs(s)):
                    diffs.append(f"{metric} map {name}: native {s} browser {gs}")
            print(metric, "score", [x["value"] for x in g["values"]], "maps", g["maps"])
        page.reload()
        page.wait_for_function("document.querySelectorAll('#metric optgroup').length > 0", timeout=60000)
        groups = page.evaluate("[...document.querySelectorAll('#metric optgroup')].map(g => [g.label, [...g.children].map(o => o.value)])")
        mine = [values for label, values in groups if label == manifest["label"]]
        if mine != [metrics]:
            diffs.append(f"menu groups: {groups}")
        extra = [label for label, _ in groups[1:]]
        if not extra or extra[0] != manifest["label"]:
            diffs.append(f"the module's group is not the first after the page's own metrics: {[g[0] for g in groups]}")
        # A link that names a module metric opens the page on it.
        page.goto(f"http://127.0.0.1:3322/?metric={quote(metrics[0])}")
        page.wait_for_function("document.querySelectorAll('#metric optgroup').length > 0", timeout=60000)
        chosen = page.evaluate("document.getElementById('metric').value")
        if chosen != metrics[0]:
            diffs.append(f"?metric={metrics[0]} opened {chosen}")
        if errors:
            diffs.append(f"page errors: {errors}")
        browser.close()
finally:
    server.shutdown()
    server.server_close()
for d in diffs:
    print("DIFF", d[:400])
if diffs:
    raise SystemExit(f"{len(diffs)} differences")
print(f"browser: {len(metrics)} metrics scored and mapped through the workbench worker, equal to native")
