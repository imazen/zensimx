// Run only through viz-function-browser (heavy lock + run-heavy).
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
const directory = process.env.VIZ_CAPTURE_DIR;
await import("./review2.mjs");
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
const atlas = await (
  await page.request.get(new URL(manifest.pairs[0].meta, base).href)
).json();
await page.locator("[data-view=lens]").click();
await page.locator("[data-exhibit-tab=Weights]").click();
const selected = page.locator("#selected-feature-link a");
assert.ok((await selected.getAttribute("href")).includes("feature="));
await selected.click();
assert.equal(await page.locator("#reference-body .katex").count(), 1);
await page.locator("[data-exhibit-tab=Concerns]").click();
const concern = page.locator("#concern-table a[data-explain-input]").first();
assert.ok((await concern.textContent()).includes(" · "));
await concern.click();
assert.equal(await page.locator("#reference-body .katex").count(), 1);
await page.locator("[data-exhibit-tab=Attribution]").click();
const ablation = page
  .locator("#input-ranking-chart a[data-explain-input]")
  .first();
assert.ok(await ablation.getAttribute("href"));
await ablation.click();
assert.equal(await page.locator("#reference-body .katex").count(), 1);
const cases = {},
  captures = [];
let deepFeatureHref, deepUnitHref;
async function shot(name) {
  await page.locator("#function-exhibit").evaluate((n) => (n.scrollTop = 0));
  await page.locator(".app-name").hover();
  const geometry = await page.evaluate(() => ({
    viewport: [innerWidth, innerHeight],
    document: [
      document.documentElement.scrollWidth,
      document.documentElement.scrollHeight,
    ],
  }));
  assert.deepEqual(geometry.document, geometry.viewport);
  const plot = await page
    .locator(".function-chart")
    .evaluate((n) => ({
      bottom: n.getBoundingClientRect().bottom,
      paneBottom: document
        .querySelector("#function-exhibit")
        .getBoundingClientRect().bottom,
      plotHeight:
        (Number(n.dataset.plotHeight) * n.getBoundingClientRect().height) /
        n.viewBox.baseVal.height,
      stageHeight: document.querySelector("#center-stage").clientHeight,
    }));
  assert.ok(plot.bottom <= plot.paneBottom, JSON.stringify(plot));
  if (geometry.viewport[0] === 1600)
    assert.ok(
      plot.plotHeight * 100 >= plot.stageHeight * 60,
      JSON.stringify(plot),
    );
  if (directory) {
    const path = `${directory}/${name}`;
    assert.equal(existsSync(path), false);
    await page.screenshot({ path });
  }
  captures.push({ name, ...geometry, plot });
}
for (const bake of ["F0", "F1", "F2", "E"]) {
  await page.locator("#bake").selectOption(bake);
  await page.locator("[data-exhibit-tab=Function]").click();
  await page.waitForFunction(
    (bake) =>
      document.querySelector("#function-exhibit").dataset.bake === bake &&
      document.querySelector("#function-term"),
    bake,
  );
  const source = manifest.bakes.find((b) => b.id === bake).banks["cid22-fit"];
  assert.ok(source.function);
  const doc = await (
    await page.request.get(new URL(source.function, base).href)
  ).json();
  const weights = await (
    await page.request.get(new URL(source.weights, base).href)
  ).json();
  assert.equal(doc.certificate.certified, true);
  assert.equal(
    await page.locator("#function-term option").count(),
    doc.pairs.length + doc.ordinary_terms.length,
  );
  assert.match(
    await page.locator(".memorization-note summary").textContent(),
    /not established/,
  );
  cases[bake] = {
    pairs: doc.pairs.length,
    retained_other_terms: doc.ordinary_terms.length,
    zero_output_nondead: doc.zero_output_nondead_units.length,
    linear_background: doc.linear_background.units.length,
    other_hinges: doc.other_hinge_count,
    line_bound: doc.certified_line_segments_upper_bound,
    global_affine_bound: doc.global_affine_forms_upper_bound,
    units: doc.included_units,
  };
  for (const pair of doc.pairs) {
    await page.locator("#function-term").selectOption(`pair:${pair.index}`);
    await page.waitForFunction(
      (index) =>
        document.querySelector("#function-exhibit").dataset.term ===
        `pair:${index}`,
      pair.index,
    );
    if (bake === "E" && pair.index === 1)
      deepUnitHref = await page
        .locator(".term-formula .unit-link")
        .last()
        .getAttribute("href");
    assert.equal(
      await page.locator(".function-chart").getAttribute("data-native-replay"),
      "true",
    );
    assert.equal(
      await page.locator(".function-chart").getAttribute("data-points"),
      "65",
    );
    await page.locator("#function-coefficients > summary").click();
    await page.waitForSelector(".function-coefficient-table tbody tr");
    const actual = await page
      .locator(".function-coefficient-table tbody tr")
      .evaluateAll((rows) =>
        rows.map((r) => ({
          input: Number(r.querySelector("a").dataset.explainInput),
          values: [...r.cells].slice(1).map((c) => c.dataset.rawNumber),
        })),
      );
    assert.deepEqual(
      actual,
      pair.coefficients.map((r) => ({
        input: r.input,
        values: [r.a, r.b, r.delta, weights.scaler_scale[r.input]].map(String),
      })),
    );
    await page.locator("#function-coefficients > summary").click();
    await page.locator("#function-values > summary").click();
    const values = await page
      .locator("#function-values tbody tr")
      .evaluateAll((rows) =>
        rows.map((r) => [
          r.cells[0].dataset.rawNumber,
          r.cells[2].dataset.rawNumber,
          r.cells[3].dataset.rawNumber,
          r.cells[4].dataset.rawNumber,
        ]),
      );
    assert.deepEqual(
      values,
      pair.slice.points.map((p) =>
        [p.x, p.term_sum, p.network, p.served].map(String),
      ),
    );
    await page.locator("#function-values > summary").click();
  }
  const pair = doc.pairs[0];
  await page.locator("#function-term").selectOption("pair:0");
  const termPath = await page
    .locator(".function-chart path")
    .last()
    .getAttribute("d");
  await page.locator("[data-function-mode=network]").click();
  assert.notEqual(
    await page.locator(".function-chart path").last().getAttribute("d"),
    termPath,
  );
  await page.locator("[data-function-mode=served]").click();
  assert.equal(await page.locator(".function-chart path").count(), 1);
  await page.locator("[data-function-mode=term]").click();
  await page.locator("#function-coefficients > summary").click();
  const product = weights.inputs.find((i) => i.factor != null),
    productFeature = atlas.features.find(
      (f) => f.id === product.feature && f.fragility,
    );
  const productLink = page.locator(
    `.function-coefficient-table a[data-explain-input="${product.index}"]`,
  );
  const href = await productLink.getAttribute("href");
  assert.ok(href.includes(`feature=${productFeature.key}`));
  await productLink.click();
  await page.waitForFunction(
    (key) => document.querySelector("#reference-body").dataset.feature === key,
    productFeature.key,
  );
  assert.equal(await page.locator("#reference-body .katex").count(), 1);
  if (bake === "E") deepFeatureHref = href;
  await page.locator("#function-coefficients > summary").click();
  const unitsLink = page.locator(".term-formula .unit-link").last();
  if (await unitsLink.count()) {
    await unitsLink.click();
  }
  await page.locator("[data-exhibit-tab=Function]").click();
  await page.locator("#function-term").selectOption("pair:0");
  await page.locator(".function-axis a").click();
  for (const size of [
    { width: 1600, height: 1000 },
    ...(["F0", "E"].includes(bake) ? [{ width: 1280, height: 800 }] : []),
  ]) {
    await page.setViewportSize(size);
    await page.waitForFunction(
      () =>
        document.querySelector(".function-chart").viewBox.baseVal.width ===
        Math.max(420, document.querySelector("#center-stage").clientWidth - 48),
    );
    for (const theme of ["dark", "light"]) {
      if ((await page.locator("html").getAttribute("data-theme")) !== theme)
        await page.locator("#theme").click();
      await shot(`function-${bake}-${theme}-${size.width}x${size.height}.png`);
    }
  }
  if (doc.ordinary_terms.length) {
    const term = doc.ordinary_terms[0];
    await page.locator("#function-term").selectOption(`unit:${term.unit}`);
    assert.match(
      await page.locator(".term-formula").textContent(),
      /retained exactly/,
    );
    await page.locator("#function-coefficients > summary").click();
    if (doc.linear_background.units.length) {
      await page.locator("#function-background > summary").click();
      await page.waitForSelector("#function-background tbody tr");
      assert.deepEqual(
        await page
          .locator("#function-background tbody tr")
          .evaluateAll((rows) => rows.map((r) => r.cells[1].dataset.rawNumber)),
        doc.linear_background.coefficients.map(String),
      );
      await page.locator("#function-background > summary").click();
    }
    await page.waitForSelector(".function-coefficient-table tbody tr");
    assert.equal(
      await page.locator(".function-coefficient-table tbody tr").count(),
      weights.inputs.length,
    );
    await page.locator("#function-coefficients > summary").click();
    await page.setViewportSize({ width: 1600, height: 1000 });
    if ((await page.locator("html").getAttribute("data-theme")) !== "dark")
      await page.locator("#theme").click();
    await page.locator(".function-axis a").click();
    await page.waitForFunction(
      () =>
        document.querySelector(".function-chart").viewBox.baseVal.width ===
        Math.max(420, document.querySelector("#center-stage").clientWidth - 48),
    );
    await shot(`function-${bake}-residual-dark-1600x1000.png`);
  }
  await page.setViewportSize({ width: 1600, height: 1000 });
}
for (const hash of [deepFeatureHref, deepUnitHref]) {
  assert.ok(hash);
  const linked = await browser.newPage({
    viewport: { width: 1600, height: 1000 },
  });
  linked.on("pageerror", (e) => errors.push(e.message));
  await linked.goto(new URL(hash, base).href);
  await linked.waitForFunction(
    () => window.zensimStudio && document.querySelector("#bake").value === "E",
  );
  if (hash.includes("feature="))
    await linked.waitForFunction(
      (key) =>
        document.querySelector("#reference-body").dataset.feature === key,
      new URLSearchParams(hash.slice(1)).get("feature"),
    );
  else {
    await linked.waitForFunction(
      () =>
        document
          .querySelector("[data-exhibit-tab=Function]")
          .getAttribute("aria-pressed") === "true" &&
        document.querySelector(".function-chart"),
    );
    const unit = Number(new URLSearchParams(hash.slice(1)).get("unit"));
    await linked.waitForFunction(
      (unit) => zensimStudio.state().unit === unit,
      unit,
    );
    assert.equal(
      await linked.locator("#function-exhibit").getAttribute("data-term"),
      "pair:1",
    );
  }
  await linked.close();
}
assert.deepEqual(errors, []);
const result = {
  source_commit: process.env.VIZ_REVIEW_COMMIT || null,
  feature_links: [
    "selected weight",
    "concern",
    "ablation",
    "raw and fragility coefficient tables",
    "copied feature URL",
    "copied unit URL",
  ],
  cases,
  captures,
  uncaught_errors: errors,
};
if (directory) {
  writeFileSync(
    `${directory}/function-review.json`,
    JSON.stringify(result, null, 2) + "\n",
    { flag: "wx" },
  );
  const old = JSON.parse(readFileSync(`${directory}/design2-gates.json`));
  const names = [
    ...old.screenshots.map((s) => s.name),
    "atlas-delta-dark-1600x1000.png",
    "lens-weights-dark-1600x1000.png",
    "lens-weights-light-1600x1000.png",
    ...captures.map((s) => s.name),
  ];
  writeFileSync(
    `${directory}/function-index.html`,
    `<!doctype html><meta charset="utf-8"><title>zensim · explained MLP function</title><style>body{font:14px system-ui;background:#0d0d0d;color:#f0efec;margin:24px}a{color:inherit}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}img{width:100%}figcaption{padding:8px 0}</style><h1>Explained features, paired hinges and structural capacity</h1><p>Source ${result.source_commit}</p><p><a href="function-review.json">Function and link acceptance</a> · <a href="browser-review.json">Retained review-2 gates</a></p><main>${names.map((n) => `<figure><a href="${n}"><img loading="lazy" src="${n}" alt="${n}"></a><figcaption>${n}</figcaption></figure>`).join("")}</main>\n`,
    { flag: "wx" },
  );
}
console.log(JSON.stringify(result, null, 2));
await browser.close();
