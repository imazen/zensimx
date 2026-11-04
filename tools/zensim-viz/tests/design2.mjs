import { createRequire } from "node:module";
import assert from "node:assert/strict";
import { existsSync, writeFileSync } from "node:fs";
const require = createRequire(import.meta.url),
  { chromium } = require(process.env.VIZ_PLAYWRIGHT || "playwright");
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.VIZ_CHROMIUM,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } }),
  errors = [],
  base = process.env.VIZ_URL || "http://dev.lan:3310/";
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(base);
await page.waitForFunction(() => !!window.zensimStudio);
await page.waitForSelector(".katex");
const ids = await page.evaluate(() => zensimStudio.catalog()),
  refs = await page.evaluate(() => zensimStudio.references());
assert.equal(ids.length, 984);
assert.equal(new Set(ids.map((f) => f.key)).size, 984);
for (const f of ids) {
  assert.ok(Number.isInteger(f.id) && f.id >= 0 && f.id < 720);
  assert.ok(refs.entries[f.reference]);
}
for (const key of Object.keys(refs.entries))
  assert.ok(
    ids.some((f) => (key === "fragility" ? f.fragility : f.reference === key)),
    `unused reference ${key}`,
  );
const rendered = await page.locator("#matrix .cell").count();
assert.ok(rendered > 0 && rendered < 120);

const visited = await page.evaluate(async () => {
  const keys = new Set(),
    root = document.querySelector("#matrix"),
    toggle = document.querySelector("#fragility-twin");
  for (const twin of [false, true]) {
    toggle.checked = twin;
    toggle.dispatchEvent(new Event("change"));
    for (
      let top = 0;
      top <= root.scrollHeight + root.clientHeight;
      top += root.clientHeight - 64
    ) {
      root.scrollTop = top;
      await new Promise(requestAnimationFrame);
      root
        .querySelectorAll("[data-feature]")
        .forEach((b) => keys.add(b.dataset.feature));
    }
  }
  toggle.checked = false;
  toggle.dispatchEvent(new Event("change"));
  root.scrollTop = 0;
  await new Promise(requestAnimationFrame);
  return [...keys];
});
assert.deepEqual(
  visited.sort(),
  ids.map((f) => f.key).sort(),
  "every virtual row must resolve to the stored catalog",
);
for (const [key, r] of Object.entries(refs.entries)) {
  const row = ids.find((f) =>
    key === "fragility" ? f.fragility : f.reference === key,
  );
  await page.evaluate((key) => zensimStudio.select(key), row.key);
  assert.equal(
    await page.locator("#reference-body .katex").count(),
    1,
    `equation ${key}`,
  );
  assert.ok(r.formula);
  assert.ok(r.code.startsWith("https://"));
  for (const c of r.inspiration) {
    assert.ok(c.author_year && c.title && c.url);
  }
}
await page.evaluate(() => zensimStudio.select("input0"));
const gutter = page.locator(".gutter[data-side=left]");
await gutter.focus();
for (let i = 0; i < 30; i++) await page.keyboard.press("ArrowRight");
assert.ok(
  await page
    .locator("#center-stage")
    .evaluate((n) => n.clientWidth * 100 >= innerWidth * 55),
);
const resizedLeft = await page
  .locator(".feature-sidebar")
  .evaluate((n) => n.clientWidth);
assert.ok(resizedLeft >= 240 && resizedLeft <= 480);
for (let i = 0; i < 30; i++) await page.keyboard.press("ArrowLeft");
assert.equal(
  await page.locator(".feature-sidebar").evaluate((n) => n.clientWidth),
  240,
);
await page.setViewportSize({ width: 1601, height: 1000 });
await page.setViewportSize({ width: 1600, height: 1000 });
await page.keyboard.press("/");
assert.equal(
  await page
    .locator("#feature-search")
    .evaluate((n) => n === document.activeElement),
  true,
);
await page.locator("#feature-search").blur();
const family = page.locator("#matrix .tree-header").first();
await family.click();
assert.equal(
  await page
    .locator("#matrix .tree-header")
    .first()
    .getAttribute("aria-expanded"),
  "false",
);
await page.locator("#feature-search").fill("f20");
await page.locator("#feature-search").fill("");
assert.equal(
  await page
    .locator("#matrix .tree-header")
    .first()
    .getAttribute("aria-expanded"),
  "false",
);
await page.locator("#matrix .tree-header").first().click();
await page.evaluate(() => zensimStudio.preload("basic"));
const times = [];
for (const key of [
  "input13",
  "input26",
  "input39",
  "input20",
  "input1",
  "input0",
]) {
  await page.evaluate((key) => zensimStudio.select(key), key);
  await page.waitForFunction(
    (key) => document.querySelector("#overlay").dataset.map === key,
    key,
  );
  times.push(
    Number(await page.locator("#overlay").getAttribute("data-latency")),
  );
}
assert.ok(
  times.every((t) => t < 50),
  `preloaded latency ${times}`,
);
await page.locator("#feature-search").fill("edge_det");
await page.locator("#matrix .cell").first().click();
assert.match(await page.locator("#pins").textContent(), /f/);
assert.equal(await page.locator("#release-hold").isVisible(), true);
await page.locator("#release-hold").click();
await page.locator("#feature-search").fill("");

const manifest = await (
  await page.request.get(new URL("data/manifest.json", base).href)
).json();
const meta = await (
  await page.request.get(new URL(manifest.pairs[0].meta, base).href)
).json();
const native = meta.features.find((f) => f.key === "input20");
assert.ok(native.terms_file, "level 2 terms are required");
await page.evaluate(() => zensimStudio.select("input20"));
const headerUrl = new URL(
  native.terms_file,
  new URL(manifest.pairs[0].meta, base),
);
const terms = await (await page.request.get(headerUrl.href)).json();
assert.equal(terms.formula_replay_bit_identity, true);
const raw = await (
  await page.request.get(new URL(terms.data_file, headerUrl).href)
).body();
const cursorSamples = [];
const checkCursor = async (
  u,
  v,
  termHeader = terms,
  data = raw,
  result = "detail",
) => {
  const rect = await page.locator("#pair-image").boundingBox();
  await page.evaluate(({ x, y }) => zensimStudio.demo(x, y), {
    x: rect.x + u * rect.width,
    y: rect.y + v * rect.height,
  });
  const index =
    Math.floor((v * termHeader.native_grid[1]) / termHeader.step) *
      termHeader.width +
    Math.floor((u * termHeader.native_grid[0]) / termHeader.step);
  assert.equal(
    Number(await page.locator("#formula-demo").getAttribute("data-index")),
    index,
  );
  const plane = termHeader.planes.indexOf(result);
  const expected = data.readFloatLE(
    (plane * termHeader.width * termHeader.height + index) * 4,
  );
  assert.equal(
    Number(await page.locator("#formula-demo").getAttribute("data-result")),
    expected,
  );
  cursorSamples.push({
    feature: await page.locator("#formula-demo").getAttribute("data-feature"),
    index,
    result,
    value: expected,
    readout: await page.locator("#formula-demo").textContent(),
  });
};
await checkCursor(0.413, 0.297);
await page.locator("#zoom").focus();
await page.keyboard.press("Home");
for (let i = 0; i < 10; i++) await page.keyboard.press("ArrowRight");
assert.equal(await page.locator("#zoom").inputValue(), "2");
const stageBounds = await page.locator("#stage").boundingBox();
await page.mouse.move(stageBounds.x + 300, stageBounds.y + 300);
await page.mouse.down();
await page.mouse.move(stageBounds.x + 330, stageBounds.y + 320);
await page.mouse.up();
await checkCursor(0.49, 0.46);
await page.locator("#stage").dblclick();
assert.equal(await page.locator("#zoom").inputValue(), "1");
await page
  .locator(".reference-panel")
  .evaluate((n) => (n.scrollTop = n.scrollHeight));
assert.match(
  await page.locator("#reference-body").textContent(),
  /total \|weight\|/,
);
assert.doesNotMatch(
  await page.locator("#reference-body").textContent(),
  /total \|weight\| not exported/,
);
const selectedBake = manifest.bakes.find((b) => b.id === "F0").banks[
  "cid22-fit"
];
const nativeAudit = await (
  await page.request.get(new URL(selectedBake.audit, base).href)
).json();
const flags20 =
  nativeAudit.flags.filter((f) => f.layer === 0 && f.from === 20).length +
  selectedBake.monotonicity_flags.filter((f) => f.feature === 20).length;
assert.ok(flags20 > 0);
assert.equal(
  Number(await page.locator("#reference-body").getAttribute("data-flag-count")),
  flags20,
);
for (const id of [39, 117]) {
  await page.evaluate((id) => zensimStudio.select(`input${id}`), id);
  const f = meta.features.find((f) => f.key === `input${id}`),
    url = new URL(f.terms_file, new URL(manifest.pairs[0].meta, base));
  const header = await (await page.request.get(url.href)).json(),
    data = await (
      await page.request.get(new URL(header.data_file, url).href)
    ).body();
  await checkCursor(
    id === 39 ? 0.999 : 0.023,
    id === 39 ? 0.999 : 0.977,
    header,
    data,
    "d1",
  );
}
// Exercise each exhibit and the bank-scoped evidence before taking the overview captures.
await page.locator("[data-view=lens]").click();
for (const tab of [
  "Concerns",
  "Weights",
  "Dependence",
  "Ranges",
  "Attribution",
]) {
  await page.locator(`[data-exhibit-tab=${tab}]`).click();
  assert.equal(await page.locator("[data-exhibit]:visible").count(), 1);
}
await page.locator(".reference-options").evaluate((n) => (n.open = true));
await page.locator("#lens-family").selectOption("v2");
await page.waitForFunction(
  () => document.querySelector("#pdp-title").textContent === "v2 family",
);
await page.locator("#lens-family").selectOption("");
await page.waitForFunction(() =>
  document.querySelector("#pdp-title").textContent.startsWith("f"),
);
await page.locator("#bank").selectOption("hdr-teacher");
await page.waitForFunction(() =>
  document.querySelector("#audit-summary").textContent.includes("7,390"),
);
await page.locator("#bank").selectOption("cid22-fit");
await page.waitForFunction(() =>
  document.querySelector("#audit-summary").textContent.includes("12,163"),
);
await page.locator("[data-view=atlas]").click();
await page.locator("#pair").selectOption("pair-06");
await page.waitForFunction(() =>
  document.querySelector("#subject").textContent.includes("Flower"),
);
await page.locator("#pair").selectOption("pair-01");
await page.waitForFunction(() =>
  document.querySelector("#subject").textContent.includes("Architecture"),
);
await page.locator("#feature-search").fill("f20");
await page.locator("#matrix [data-feature=input20]").click();
await page.locator("#feature-search").fill("");
await page.locator(".reference-panel").evaluate((n) => (n.scrollTop = 0));
const screenshots = [];
for (const size of [
  { width: 1600, height: 1000 },
  { width: 1280, height: 800 },
]) {
  await page.setViewportSize(size);
  for (const view of ["atlas", "lens"]) {
    await page.locator(`[data-view=${view}]`).click();
    if (view === "lens")
      await page.locator("[data-exhibit-tab=Dependence]").click();
    for (const theme of ["dark", "light"]) {
      if ((await page.locator("html").getAttribute("data-theme")) !== theme)
        await page.locator("#theme").click();
      await page.waitForTimeout(100);
      const geometry = await page.evaluate(() => ({
        viewport: [innerWidth, innerHeight],
        document: [
          document.documentElement.scrollWidth,
          document.documentElement.scrollHeight,
        ],
        stage: document
          .querySelector("#center-stage")
          .getBoundingClientRect()
          .toJSON(),
        exhibits: [...document.querySelectorAll("[data-exhibit]")].filter(
          (e) => !e.hidden,
        ).length,
      }));
      assert.deepEqual(geometry.document, geometry.viewport);
      for (const id of ["bake", "pair", "bank"])
        assert.ok(
          await page
            .locator(`#${id}`)
            .evaluate((n) => parseFloat(getComputedStyle(n).fontSize) >= 12),
        );
      if (view === "atlas") {
        const fits = await page.evaluate(() => {
          const image = document
              .querySelector("#pair-image")
              .getBoundingClientRect(),
            stage = document.querySelector("#stage").getBoundingClientRect();
          return image.width <= stage.width && image.height <= stage.height;
        });
        assert.ok(fits, "image must fit again after a hidden-pane resize");
      } else
        assert.match(
          await page.locator("#map-legend").textContent(),
          /Dependence.*native curves/,
        );
      if (size.width === 1600) {
        assert.ok(geometry.stage.width * 100 >= 55 * size.width);
        assert.ok(geometry.stage.height * 100 >= 70 * size.height);
      }
      assert.equal(geometry.exhibits, 1);
      const name = `${view}-${theme}-${size.width}x${size.height}.png`;
      if (process.env.VIZ_CAPTURE_DIR) {
        const path = `${process.env.VIZ_CAPTURE_DIR}/${name}`;
        assert.equal(existsSync(path), false);
        await page.screenshot({ path });
      }
      screenshots.push({ name, ...geometry });
    }
  }
}
assert.deepEqual(errors, []);
const review = {
  review: process.env.VIZ_REVIEW_NAME || "DESIGN-2",
  coverage: refs.counts,
  sidebar_ids: ids.length,
  virtual_rows_visited: visited.length,
  equations_rendered: Object.keys(refs.entries).length,
  native_cursor_samples: cursorSamples,
  reference_flag_count: flags20,
  source_commit: process.env.VIZ_REVIEW_COMMIT || null,
  resized_left_px: resizedLeft,
  virtualized_cells: rendered,
  preloaded_hover_ms: times,
  screenshots,
  uncaught_errors: errors,
};
if (process.env.VIZ_CAPTURE_DIR)
  writeFileSync(
    `${process.env.VIZ_CAPTURE_DIR}/${process.env.VIZ_BROWSER_RECEIPT || "browser-review.json"}`,
    JSON.stringify(review, null, 2) + "\n",
    { flag: "wx" },
  );
console.log(JSON.stringify(review, null, 2));
await browser.close();
