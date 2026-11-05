// Native/Wasm scientific gates: fail on every unsupported, nonfinite or missing map.
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url),
  { chromium } = require(process.env.VIZ_PLAYWRIGHT ?? "playwright");
const [url, nativePath, out] = process.argv.slice(2),
  native = JSON.parse(readFileSync(nativePath, "utf8"));
const browser = await chromium.launch({
  executablePath: process.env.VIZ_CHROMIUM,
  headless: true,
  args: ["--disable-dev-shm-usage"],
});
const page = await browser.newPage();
let result;
try {
  if ((await page.goto(url + "/parity.html")).status() !== 200)
    throw Error("Missing parity probe page");
  result = await page.evaluate(async (native) => {
    const m = await import("./pkg/zensim_workbench.js");
    await m.default();
    if (m.thread_count() !== 1 || m.available_metrics() !== '["A"]')
      throw Error("Wrong module or thread count");
    const cases = [],
      failures = [];
    let maxScoreDelta = 0,
      maxCellScaledDelta = 0,
      maxSumScaledDelta = 0;
    for (const c of native) {
      const reference = new Uint8Array(
          await (await fetch("./fixtures/" + c.reference)).arrayBuffer(),
        ),
        distorted = new Uint8Array(
          await (await fetch("./fixtures/" + c.distorted)).arrayBuffer(),
        );
      const begin = performance.now(),
        session = new m.AtlasSession(
          "A",
          reference,
          distorted,
          c.width,
          c.height,
          c.stride,
        ),
        pairMs = performance.now() - begin;
      const summary = JSON.parse(session.summary()),
        scoreDelta = Math.abs(summary.score - c.summary.score),
        scoreLimit = 1e-6 * Math.max(1, Math.abs(c.summary.score));
      if (!Number.isFinite(c.legacy_score))
        throw Error("Missing independent native legacy score");
      const legacyDelta = Math.abs(summary.score - c.legacy_score);
      if (legacyDelta > scoreLimit) failures.push({ legacyDelta, scoreLimit });
      if (
        summary.bake_sha256 !== c.summary.bake_sha256 ||
        summary.bake_sha256 !== m.bake_sha256("A")
      )
        throw Error("Native/Wasm bake identity mismatch");
      maxScoreDelta = Math.max(maxScoreDelta, scoreDelta);
      if (scoreDelta > scoreLimit)
        failures.push({ case: [c.width, c.height], scoreDelta, scoreLimit });
      const maps = [];
      for (const n of c.maps) {
        const start = performance.now(),
          g = JSON.parse(session.detector(n.id)),
          mapMs = performance.now() - start;
        if (
          g.width !== n.width ||
          g.height !== n.height ||
          g.values.length !== n.values.length
        )
          throw Error(`Missing/different map ${n.id}`);
        let cellDelta = 0,
          cellScaled = 0,
          ns = 0,
          gs = 0;
        for (let k = 0; k < n.values.length; k++) {
          if (!Number.isFinite(g.values[k]))
            throw Error(`Nonfinite map ${n.id}/${k}`);
          const delta = Math.abs(g.values[k] - n.values[k]);
          cellDelta = Math.max(cellDelta, delta);
          cellScaled = Math.max(
            cellScaled,
            delta / Math.max(1, Math.abs(n.values[k])),
          );
          ns += n.values[k];
          gs += g.values[k];
        }
        const sumDelta = Math.abs(gs - ns),
          sumScaled = sumDelta / Math.max(1, Math.abs(ns));
        maxCellScaledDelta = Math.max(maxCellScaledDelta, cellScaled);
        maxSumScaledDelta = Math.max(maxSumScaledDelta, sumScaled);
        if (cellScaled > 1e-4 || sumScaled > 1e-4)
          failures.push({
            case: [c.width, c.height],
            id: n.id,
            cellDelta,
            cellScaled,
            sumDelta,
            sumScaled,
          });
        maps.push({
          id: n.id,
          cells: n.values.length,
          cellDelta,
          cellScaled,
          sumDelta,
          sumScaled,
          mapMs,
        });
      }
      if (!Array.isArray(c.basic) || c.basic.length !== 36)
        throw Error("Missing native basic formula witnesses");
      for (const b of c.basic) {
        const got = JSON.parse(session.basic(b.id, b.terms.x, b.terms.y));
        if (got.formula_replay_bit_identity !== true)
          throw Error("Basic formula replay failed");
        for (let i = 0; i < got.values.length; i++)
          if (
            Math.abs(got.values[i] - b.terms.values[i]) >
            1e-4 * Math.max(1, Math.abs(b.terms.values[i]))
          )
            failures.push({
              basic: b.id,
              field: i,
              native: b.terms.values[i],
              wasm: got.values[i],
            });
      }
      cases.push({
        width: c.width,
        height: c.height,
        stride: c.stride,
        pairMs,
        scoreDelta,
        legacyDelta,
        nativeBitChecks: c.native_pooled_bit_checks,
        maps,
      });
      session.free();
    }
    return {
      passed: failures.length === 0,
      cases,
      failures,
      maxScoreDelta,
      maxCellScaledDelta,
      maxSumScaledDelta,
      scoreTolerance: 1e-6,
      mapTolerance: 1e-4,
      threads: m.thread_count(),
    };
  }, native);
  writeFileSync(out, JSON.stringify(result, null, 2) + "\n");
  console.log(
    JSON.stringify({
      passed: result.passed,
      cases: result.cases.length,
      maps: result.cases.reduce((n, c) => n + c.maps.length, 0),
      maxScoreDelta: result.maxScoreDelta,
      maxCellScaledDelta: result.maxCellScaledDelta,
      maxSumScaledDelta: result.maxSumScaledDelta,
      failures: result.failures.length,
    }),
  );
  if (!result.passed)
    throw Error(
      `Parity failed; ${result.failures.length} failures saved in ${out}`,
    );
} finally {
  await browser.close();
}
