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
      if (size.width === 1600) {
        assert.ok(geometry.stage.width >= 0.55 * size.width);
        assert.ok(geometry.stage.height >= 0.7 * size.height);
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
  review: "DESIGN-2",
  coverage: refs.counts,
  sidebar_ids: ids.length,
  virtualized_cells: rendered,
  preloaded_hover_ms: times,
  screenshots,
  uncaught_errors: errors,
};
if (process.env.VIZ_CAPTURE_DIR)
  writeFileSync(
    `${process.env.VIZ_CAPTURE_DIR}/browser-review.json`,
    JSON.stringify(review, null, 2) + "\n",
    { flag: "wx" },
  );
console.log(JSON.stringify(review, null, 2));
await browser.close();
