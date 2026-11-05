import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url),
  { chromium } = require(process.env.VIZ_PLAYWRIGHT ?? "playwright");
const [url, path, out] = process.argv.slice(2),
  native = JSON.parse(readFileSync(path, "utf8"));
const browser = await chromium.launch({
  executablePath: process.env.VIZ_CHROMIUM,
  headless: true,
  args: ["--disable-dev-shm-usage"],
});
try {
  const page = await browser.newPage();
  if ((await page.goto(url + "/parity.html")).status() !== 200)
    throw Error("Missing parity probe page");
  const result = await page.evaluate(async (native) => {
    const m = await import("./pkg/zensim_workbench.js");
    await m.default();
    const failures = [],
      timings = [];
    let numbers = 0,
      maxScaledDelta = 0;
    function compare(n, g, path) {
      if (typeof n === "number") {
        numbers++;
        const delta = Math.abs(n - g) / Math.max(1, Math.abs(n));
        maxScaledDelta = Math.max(maxScaledDelta, delta);
        if (!Number.isFinite(g) || delta > 1e-6)
          failures.push({ path, native: n, wasm: g, delta });
      } else if (Array.isArray(n)) {
        if (!Array.isArray(g) || g.length !== n.length) {
          failures.push({
            path,
            nativeLength: n.length,
            wasmLength: g?.length,
          });
          return;
        }
        n.forEach((v, i) => compare(v, g[i], path + "/" + i));
      } else if (n && typeof n === "object") {
        for (const k of Object.keys(n)) compare(n[k], g?.[k], path + "/" + k);
      } else if (n !== g) failures.push({ path, native: n, wasm: g });
    }
    for (const [name, bank] of Object.entries(native)) {
      const floats = async (file) =>
        new Float32Array(
          await (await fetch("./banks/" + name + "/" + file)).arrayBuffer(),
        );
      const sample = await floats("sample.f32le"),
        grids = await floats("quantiles.f32le"),
        s = new m.LensSession("A", Uint32Array.from(bank.header.slots), sample);
      for (let i = 0; i < bank.curves.length; i++) {
        const n = bank.curves[i],
          index = bank.header.slots.indexOf(n.raw_feature),
          begin = performance.now(),
          g = JSON.parse(
            s.curve(n.raw_feature, grids.slice(index * 21, index * 21 + 21)),
          );
        timings.push({
          bank: name,
          feature: n.raw_feature,
          ms: performance.now() - begin,
        });
        compare(n, g, name + "/curve/" + n.raw_feature);
        compare(
          bank.diagnostics[i],
          JSON.parse(
            s.diagnostics(n.raw_feature, Uint32Array.of(n.raw_feature)),
          ),
          name + "/diagnostics/" + n.raw_feature,
        );
      }
      for (const f of bank.families) {
        compare(
          f.curve,
          JSON.parse(s.family(Uint32Array.from(f.features))),
          name + "/family/" + f.family,
        );
        compare(
          f.diagnostics,
          JSON.parse(
            s.diagnostics(f.features[0], Uint32Array.from(f.features)),
          ),
          name + "/family-diagnostics/" + f.family,
        );
      }
      s.free();
    }
    return {
      passed: failures.length === 0,
      numbers,
      maxScaledDelta,
      failures,
      timings,
    };
  }, native);
  writeFileSync(out, JSON.stringify(result, null, 2) + "\n");
  console.log(
    JSON.stringify({
      passed: result.passed,
      numbers: result.numbers,
      maxScaledDelta: result.maxScaledDelta,
      failures: result.failures.length,
    }),
  );
  if (!result.passed) throw Error("Native/Wasm lens mismatch");
} finally {
  await browser.close();
}
