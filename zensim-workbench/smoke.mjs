// Wasm side of the smoke check (`just byc-workbench-smoke`): the same pair and
// lines as `examples/smoke.rs`, from the built module at PKG (pkg-zensimx).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const pkg = process.argv[2];
const mod = await import(pathToFileURL(join(pkg, 'zensim_workbench.js')).href);
mod.initSync({ module: readFileSync(join(pkg, 'zensim_workbench_bg.wasm')) });

const [w, h] = [96, 80];
const px = (x, y) => {
  const v = ((x * 7 + y * 13) % 200) + 20;
  return [v, Math.floor(v / 2) + 40, 255 - v, 255];
};
const r = new Uint8Array(w * h * 4);
const d = new Uint8Array(w * h * 4);
for (let y = 0; y < h; y++) {
  for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 4;
    const a = px(x, y), b = px(Math.min(x + 1, w - 1), y);
    r.set(a, o);
    d.set([Math.floor((a[0] + b[0]) / 2), Math.min(Math.floor((a[1] + b[1]) / 2) + 2, 255),
      Math.floor((a[2] + b[2]) / 2), 255], o);
  }
}
for (const metric of JSON.parse(mod.available_metrics())) {
  const score = JSON.parse(mod.score_rgba(metric, r, d, w, h, w * 4, ''));
  const maps = {};
  const names = JSON.parse(mod.spatial_map(metric, 'diffmap', r, d, w, h, w * 4, '').names);
  for (const name of names) {
    const lm = mod.loop_map(metric, name, r, d, w, h, w * 4, '', 1.0, 1.0);
    maps[name] = Array.from(lm.values).reduce((s, v) => s + v, 0);
  }
  console.log(JSON.stringify({ metric, score, maps }));
}
