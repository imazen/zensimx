// Run under run-heavy + heavy.lock. Existing numeric and interaction gates run
// unchanged before the eight review decisions and the ninth (delta) capture.
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
const directory = process.env.VIZ_CAPTURE_DIR;
process.env.VIZ_REVIEW_NAME = "DESIGN-REVIEW-2";
process.env.VIZ_BROWSER_RECEIPT = "design2-gates.json";
await import("./design2.mjs");
delete process.env.VIZ_CAPTURE_DIR;
await import("./review1.mjs");
if (directory) process.env.VIZ_CAPTURE_DIR = directory;
const require = createRequire(import.meta.url),
  { chromium } = require(process.env.VIZ_PLAYWRIGHT || "playwright");
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.VIZ_CHROMIUM,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } }),
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const base = process.env.VIZ_URL || "http://dev.lan:3310/";
await page.goto(base);
await page.waitForFunction(() => !!window.zensimStudio);
const manifest = await (
  await page.request.get(new URL("data/manifest.json", base).href)
).json();
await page.evaluate(() => zensimStudio.preload("basic"));
await page.locator("#feature-search").fill("f20");
await page.locator("#matrix [data-feature=input20]").click();
await page.waitForFunction(
  () =>
    document.querySelector("#overlay").dataset.map === "input20" &&
    document.querySelector("#overlay").dataset.range === "robust",
);
assert.equal(
  await page
    .locator("#range-mode [aria-pressed=true]")
    .getAttribute("data-range"),
  "robust",
);
assert.equal(
  await page.locator("#pair-image").evaluate((n) => getComputedStyle(n).filter),
  "grayscale(1)",
);
const alphaStats = await page.locator("#overlay").evaluate((c) => {
  const p = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let positive = 0,
    zero = 0,
    min = 255,
    max = 0;
  for (let i = 3; i < p.length; i += 4) {
    if (p[i]) {
      positive++;
      min = Math.min(min, p[i]);
      max = Math.max(max, p[i]);
    } else zero++;
  }
  return {
    positive,
    zero,
    min,
    max,
    slider: Number(document.querySelector("#alpha").value),
    composite_opacity: Number(getComputedStyle(c).opacity),
  };
});
assert.ok(alphaStats.positive > 0 && alphaStats.zero > 0);
assert.ok(alphaStats.min >= 64);
assert.equal(alphaStats.max, 255);
assert.equal(alphaStats.composite_opacity, alphaStats.slider);
const atlas = await (
  await page.request.get(new URL(manifest.pairs[0].meta, base).href)
).json();
const feature20 = atlas.features.find((f) => f.key === "input20"),
  nativeF0 = manifest.bakes.find((b) => b.id === "F0").banks["cid22-fit"];
const curve20 = await (
  await page.request.get(new URL(nativeF0.lenses.f20, base).href)
).json();
assert.deepEqual(feature20.bank_plot_ranges[atlas.bank_sha256], [
  curve20.pdp.x[0],
  curve20.pdp.x.at(-1),
]);
const ranges = [];
for (const mode of ["own", "bank", "robust"]) {
  await page.locator(`#range-mode [data-range=${mode}]`).click();
  await page.waitForFunction(
    (mode) => document.querySelector("#overlay").dataset.range === mode,
    mode,
  );
  ranges.push({
    mode,
    legend: await page.locator("#map-legend").textContent(),
  });
}
assert.match(ranges[1].legend, /Bank p1–p99/);
assert.match(ranges[2].legend, /p2–p98 map pixels/);
await page.locator("#map-mode-button").click();
await page.waitForFunction(() => {
  const c = document.querySelector("#overlay"),
    p = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  for (let i = 3; i < p.length; i += 4) if (p[i] !== 255) return false;
  return true;
});
await page.locator("#map-mode-button").click();
const strip = await page.locator("#model-strip").boundingBox();
assert.ok(strip.y + strip.height < 400);
assert.ok(strip.height <= 40);
assert.match(
  await page.locator("#model-strip").textContent(),
  /\|w\|.*ablation.*read by.*flags/,
);
assert.equal(await page.locator(".feature-sidebar #status").count(), 0);
assert.ok(await page.locator(".app-name").getAttribute("title"));
assert.match(
  await page.locator("#tree-chip-legend").textContent(),
  /p0.*p100.*chip = bank percentile of this pair's value/,
);
assert.equal(await page.locator("#tree-chip-legend i").count(), 5);
await page.locator("[data-mode=difference]").click();
const deltaImage = await page.locator("#pair-image").evaluate(async (image) => {
  await image.decode();
  await new Promise(requestAnimationFrame);
  return image.currentSrc;
});
assert.equal(
  deltaImage,
  new URL(atlas.difference, new URL(manifest.pairs[0].meta, base)).href,
);
assert.equal(
  await page.locator("#pair-image").evaluate((n) => getComputedStyle(n).filter),
  "none",
);
if (directory) {
  const path = `${directory}/atlas-delta-dark-1600x1000.png`;
  assert.equal(existsSync(path), false);
  await page.screenshot({ path });
}
const deltaGeometry = await page.evaluate(() => ({
  viewport: [innerWidth, innerHeight],
  document: [
    document.documentElement.scrollWidth,
    document.documentElement.scrollHeight,
  ],
}));
assert.deepEqual(deltaGeometry.document, deltaGeometry.viewport);
await page.locator("[data-view=lens]").click();
assert.equal(await page.locator("#formula-demo").count(), 0);
await page.locator("[data-exhibit-tab=Dependence]").click();
const plot = await page.locator("#pdp-chart svg").evaluate((n) => ({
  plot:
    (Number(n.dataset.plotHeight) * n.getBoundingClientRect().height) /
    n.viewBox.baseVal.height,
  stage: document.querySelector("#center-stage").clientHeight,
}));
assert.ok(plot.plot * 100 >= plot.stage * 60, JSON.stringify(plot));
await page.locator(".gutter[data-side=left]").focus();
await page.keyboard.press("ArrowRight");
await page.waitForFunction(
  () =>
    document.querySelector("#pdp-chart svg").viewBox.baseVal.width ===
    Math.max(420, document.querySelector("#center-stage").clientWidth - 48),
);
const resizedPlot = await page.locator("#pdp-chart svg").evaluate((n) => ({
  plot:
    (Number(n.dataset.plotHeight) * n.getBoundingClientRect().height) /
    n.viewBox.baseVal.height,
  stage: document.querySelector("#center-stage").clientHeight,
}));
assert.ok(resizedPlot.plot * 100 >= resizedPlot.stage * 60);
await page.keyboard.press("ArrowLeft");
await page.locator("[data-exhibit-tab=Ranges]").click();
const rangePlot = await page
  .locator("#range-chart svg")
  .first()
  .evaluate((n) => ({
    plot:
      (Number(n.dataset.plotHeight) * n.getBoundingClientRect().height) /
      n.viewBox.baseVal.height,
    stage: document.querySelector("#center-stage").clientHeight,
  }));
assert.ok(rangePlot.plot * 100 >= rangePlot.stage * 60);
await page.locator("[data-exhibit-tab=Attribution]").click();
assert.equal(
  await page
    .locator("[data-exhibit=Attribution]")
    .evaluate((n) => n.clientHeight),
  856,
);
await page.locator("[data-exhibit-tab=Concerns]").click();
const concernsHeight = await page
  .locator("#concern-table")
  .evaluate((n) => n.clientHeight);
assert.ok(concernsHeight >= 540);
await page.locator("#kind").selectOption("duplicate_input");
assert.ok((await page.locator("#concern-table tbody tr").count()) > 0);
assert.equal(
  await page
    .locator("#concern-table tbody tr")
    .evaluateAll((rows) =>
      rows.every((r) => /^r [-+0-9.e]+$/.test(r.children[4].textContent)),
    ),
  true,
);
await page.locator("#kind").selectOption("");
await page.locator("[data-exhibit-tab=Weights]").click();
const weightsHeight = await page
  .locator("#weight-wrap")
  .evaluate((n) => n.clientHeight);
assert.ok(weightsHeight >= 540);
const weightsGeometry = await page.evaluate(() => ({
  viewport: [innerWidth, innerHeight],
  document: [
    document.documentElement.scrollWidth,
    document.documentElement.scrollHeight,
  ],
}));
assert.deepEqual(weightsGeometry.document, weightsGeometry.viewport);
assert.match(
  await page.locator("#global-weight-label").textContent(),
  /global/,
);
assert.ok(await page.locator("#global-weight-label").getAttribute("title"));
const capacities = {};
for (const bake of ["F0", "F1", "F2", "E"]) {
  await page.locator("#bake").selectOption(bake);
  await page.waitForFunction(
    (bake) =>
      document.querySelector("#capacity-strip").dataset.bake === bake &&
      document.querySelectorAll("#capacity-strip button").length === 128,
    bake,
  );
  const source = manifest.bakes.find((b) => b.id === bake).banks["cid22-fit"];
  const weights = await (
    await page.request.get(new URL(source.weights, base).href)
  ).json();
  await page.waitForFunction(
    (counts) =>
      document.querySelector(".capacity-counts").textContent ===
      `${counts.live} live · ${counts.residual} residual · ${counts.dead} dead`,
    weights.capacity.counts,
  );
  const actual = await page
    .locator("#capacity-strip button")
    .evaluateAll((nodes) =>
      nodes.map((n) => ({
        unit: Number(n.dataset.unit),
        status: n.dataset.status,
        title: n.title,
      })),
    );
  assert.deepEqual(
    actual.map((u) => ({ unit: u.unit, status: u.status })),
    weights.capacity.units.map((u) => ({ unit: u.unit, status: u.status })),
  );
  assert.ok(
    actual.every(
      (u) => u.title.includes("Output coefficient") && u.title.includes("bias"),
    ),
  );
  for (let j = 0; j < actual.length; j++) {
    assert.ok(
      actual[j].title.includes(
        `Output coefficient ${weights.capacity.units[j].output_coefficient}`,
      ),
    );
    assert.ok(
      actual[j].title.includes(`bias ${weights.capacity.units[j].bias}`),
    );
  }
  capacities[bake] = weights.capacity.counts;
  await page.locator('#capacity-strip [data-unit="126"]').click();
  await page.waitForFunction(() => zensimStudio.state().unit === 126);
  assert.match(await page.locator("#weight-detail").textContent(), /h126/);
  if (directory && bake === "F0") {
    for (const theme of ["dark", "light"]) {
      if ((await page.locator("html").getAttribute("data-theme")) !== theme)
        await page.locator("#theme").click();
      await page.locator(".app-name").hover();
      const path = `${directory}/lens-weights-${theme}-1600x1000.png`;
      assert.equal(existsSync(path), false);
      await page.screenshot({ path });
    }
    await page.locator("#theme").click();
  }
}
assert.deepEqual(capacities.F0, { dead: 95, live: 2, residual: 31 });
await page.setViewportSize({ width: 1280, height: 800 });
assert.equal(await page.locator(".bank-selected").isVisible(), true);
assert.equal(await page.locator(".bank-selected").textContent(), "CID22 TRAIN");
assert.match(await page.locator("#bank").getAttribute("title"), /12,163/);
assert.ok(
  await page
    .locator("#bank option")
    .evaluateAll((options) =>
      options.every((o) => o.label === o.title && o.title.length > 0),
    ),
);
assert.deepEqual(errors, []);
const result = {
  review: "DESIGN-REVIEW-2",
  source_commit: process.env.VIZ_REVIEW_COMMIT || null,
  items: Array.from({ length: 8 }, (_, i) => ({ item: i + 1, status: "done" })),
  alpha: alphaStats,
  ranges,
  plot,
  resized_plot: resizedPlot,
  range_plot: rangePlot,
  concerns_height: concernsHeight,
  weights_height: weightsHeight,
  weights_geometry: weightsGeometry,
  capacities,
  delta: deltaGeometry,
  delta_image: deltaImage,
  uncaught_errors: errors,
};
if (directory) {
  const baseline = JSON.parse(readFileSync(`${directory}/design2-gates.json`));
  result.acceptance = baseline;
  result.capture_directory = directory;
  writeFileSync(
    `${directory}/browser-review.json`,
    JSON.stringify(result, null, 2) + "\n",
    { flag: "wx" },
  );
  const captures = [
    ...baseline.screenshots.map((s) => s.name),
    "atlas-delta-dark-1600x1000.png",
    "lens-weights-dark-1600x1000.png",
    "lens-weights-light-1600x1000.png",
  ];
  writeFileSync(
    `${directory}/index.html`,
    `<!doctype html><meta charset="utf-8"><title>zensim-viz · Review 2</title>
<style>body{background:#0d0d0d;color:#f0efec;font:14px system-ui;margin:24px}main{display:grid;grid-template-columns:1fr 1fr;gap:24px}figure{margin:0}img{width:100%;height:auto}a{color:inherit}figcaption{padding:8px 0}</style>
<h1>Review 2 · both views, themes and viewport sizes</h1>
<p>App source ${process.env.VIZ_REVIEW_COMMIT}</p>
<p><a href="browser-review.json">Eight-item acceptance receipt</a> · <a href="design2-gates.json">Retained design gates and native cursor checks</a></p>
<main>${captures.map((name) => `<figure><a href="${name}"><img loading="lazy" src="${name}" alt="${name}"></a><figcaption>${name}</figcaption></figure>`).join("")}</main>\n`,
    { flag: "wx" },
  );
}
console.log(JSON.stringify(result, null, 2));
await browser.close();
