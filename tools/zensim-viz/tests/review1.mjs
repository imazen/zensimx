// Design-review acceptance checks against complete native bundles.
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import { existsSync, writeFileSync } from "node:fs";
const require = createRequire(import.meta.url),
  { chromium } = require(process.env.VIZ_PLAYWRIGHT || "playwright");
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
  executablePath: process.env.VIZ_CHROMIUM,
});
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } }),
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const base = process.env.VIZ_URL || "http://dev.lan:3310/";
const manifest = await (
  await page.request.get(new URL("data/manifest.json", base).href)
).json();
const f = manifest.bakes.find((b) => b.id === "F0").banks["cid22-fit"];
assert.ok(f.comparisons.E, "complete comparison evidence required");
const weights = await (
  await page.request.get(new URL(f.weights, base).href)
).json();
await page.goto(base);
await page.waitForSelector("#matrix .cell:not(.hollow)");
await page.waitForLoadState("networkidle");
assert.equal(await page.locator("#colour").inputValue(), "percentile");
assert.match(await page.locator("#pins").textContent(), /pin a feature: Enter/);
assert.equal(
  await page
    .locator("#matrix .cell.hollow")
    .evaluateAll((cells) =>
      cells.every(
        (c) =>
          c.closest(".channel") === c.closest(".matrix-row").lastElementChild,
      ),
    ),
  true,
);
assert.equal(
  await page
    .locator("#matrix .matrix-row:not(.matrix-header)")
    .evaluateAll((rows) =>
      rows.every((r) => r.querySelectorAll(".global-cell").length <= 1),
    ),
  true,
);
const cell = page.locator("#matrix .cell:not(.hollow)").first();
await cell.hover();
await page.waitForTimeout(50);
assert.match(await page.locator("#map-legend").textContent(), /pooling grid/);
const alpha = await page.locator("#overlay").evaluate((c) => {
  const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let zero = false,
    partial = false;
  for (let i = 3; i < d.length; i += 4) {
    if (d[i] === 0) zero = true;
    if (d[i] > 0 && d[i] < 255) partial = true;
  }
  return { zero, partial };
});
assert.deepEqual(alpha, { zero: true, partial: true });
await page.locator("#map-only").check();
await page.waitForFunction(() => {
  const c = document.querySelector("#overlay"),
    d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  for (let i = 3; i < d.length; i += 4) if (d[i] !== 255) return false;
  return true;
});
await page.locator("#map-only").uncheck();
await page.waitForTimeout(50);
await page.locator("#colour").selectOption("signed");
const meta = await (
  await page.request.get(new URL(manifest.pairs[0].meta, base).href)
).json();
const bankSha = manifest.banks.find((b) => b.id === "cid22-fit").sha256;
const encoded = await page.locator("#matrix .cell").evaluateAll((cells) =>
  cells.map((c) => ({
    key: c.dataset.feature,
    diverging: c.dataset.diverging,
  })),
);
for (const cell of encoded) {
  const feature = meta.features.find((f) => f.key === cell.key),
    range = feature.bank_ranges[bankSha];
  assert.equal(cell.diverging, String(range[0] < 0 && range[1] > 0));
}
await page.locator("#colour").selectOption("percentile");
await page.getByRole("button", { name: "MLP lens", exact: true }).click();
await page.waitForSelector("#pdp-chart .iqr-band");
assert.ok((await page.locator("#concern-table tbody tr").count()) <= 50);
assert.match(
  await page.locator("#concern-title").textContent(),
  /groups.*measured flags/,
);
await page.getByRole("button", { name: "Next 50", exact: true }).click();
assert.match(await page.locator("#concern-pager").textContent(), /Page 2/);
await page.getByRole("button", { name: "Previous 50", exact: true }).click();
await page.locator("#severity").selectOption("note");
await page.locator("#kind").selectOption("monotonicity");
assert.ok((await page.locator("#concern-table tbody tr").count()) > 0);
assert.ok((await page.locator("#concern-table tbody tr").count()) <= 50);
assert.match(await page.locator("#concern-table").textContent(), /note/);
await page.locator("#severity").selectOption("");
await page.locator("#kind").selectOption("");
await page.locator("#input").selectOption("371");
await page.waitForFunction(() =>
  document.querySelector("#weight-detail").textContent.includes("33 of 128"),
);
assert.doesNotMatch(
  await page.locator("#weight-detail").textContent(),
  /dead input/,
);
assert.equal(
  await page.locator("#weight-matrix").evaluate((c) => c.width),
  128,
);
assert.equal(
  await page
    .locator("#weight-matrix")
    .evaluate((c) => c.getBoundingClientRect().width),
  128,
);
assert.equal(await page.locator("#unit-input-strip g").count(), 32);
assert.equal(await page.locator("#input-unit-strip g").count(), 16);
assert.equal(
  await page
    .locator("#input-unit-strip .hit")
    .first()
    .evaluate((e) => getComputedStyle(e).fill),
  "rgba(0, 0, 0, 0)",
);
await page.locator("#lens-family").selectOption("basic");
const basicRow = weights.row_order.findIndex(
  (i) => weights.inputs[i].family === "basic",
);
const otherRow = weights.row_order.findIndex(
  (i) => weights.inputs[i].family !== "basic",
);
const familyAlpha = await page
  .locator("#weight-matrix")
  .evaluate(
    (c, { a, b }) => [
      c.getContext("2d").getImageData(0, a, 1, 1).data[3],
      c.getContext("2d").getImageData(0, b, 1, 1).data[3],
    ],
    { a: basicRow, b: otherRow },
  );
assert.deepEqual(familyAlpha, [255, 102]);
await page.locator("#lens-family").selectOption("");
const zoom = page.locator("#weight-zoom"),
  ids = await zoom.evaluate((c) => ({
    rows: c.dataset.rows.split(",").map(Number),
    columns: c.dataset.columns.split(",").map(Number),
    width: c.width,
    height: c.height,
  })),
  bounds = await zoom.boundingBox();
await page.mouse.click(
  bounds.x + (72 / ids.width) * bounds.width,
  bounds.y + (32 / ids.height) * bounds.height,
);
await page.waitForFunction(
  ({ input, unit }) =>
    document
      .querySelector("#weight-detail")
      .textContent.includes(`input ${input} ·`) &&
    document.querySelector("#weight-detail").textContent.includes(`h${unit}:`),
  { input: ids.rows[0], unit: ids.columns[0] },
);
const input20 = weights.inputs.find(
  (i) => i.feature === 20 && i.factor == null,
).index;
await page.locator("#input").selectOption(String(input20));
await page.waitForFunction(
  () => document.querySelector("#pdp-title").textContent === "f20",
);
assert.equal(await page.locator("#pdp-chart .ice").count(), 40);
assert.ok((await page.locator("#pdp-table th").count()) >= 205);
const original = await page.locator("#pdp-chart .mean").getAttribute("d");
await page.locator("#center-ice").check();
await page.waitForFunction(() =>
  document.querySelector("#pdp-chart").textContent.includes("Δ served score"),
);
assert.notEqual(
  await page.locator("#pdp-chart .mean").getAttribute("d"),
  original,
);
await page.locator("#compare").selectOption("E");
await page.waitForSelector("#pdp-chart .compare");
assert.match(
  await page.locator("#pdp-legend").textContent(),
  /E mean · dashed/,
);
assert.equal(
  await page
    .locator("#pdp-chart text")
    .filter({ hasText: /^E · dashed$/ })
    .count(),
  0,
);
assert.equal(await page.locator("#family-chart g").count(), 35);
assert.match(
  await page.locator("#input-ranking-chart").textContent(),
  /f20 · basic_/,
);
assert.ok(
  await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
);
const capture = async (view, theme) => {
  if (!process.env.VIZ_CAPTURE_DIR) return;
  if ((await page.locator("html").getAttribute("data-theme")) !== theme)
    await page.locator("#theme").click();
  await page.locator("#subject").hover();
  const path = `${process.env.VIZ_CAPTURE_DIR}/${view}-${theme}.png`;
  assert.equal(existsSync(path), false, "preserve earlier captures");
  await page.screenshot({ path, fullPage: true });
};
await page.locator("#hidden-units button").filter({ hasText: /^126$/ }).click();
await capture("mlp-lens", "dark");
await capture("mlp-lens", "light");
await page.getByRole("button", { name: "Atlas", exact: true }).click();
await page.locator("#theme").click();
await page.locator("#matrix .cell:not(.hollow)").first().hover();
await page.waitForTimeout(50);
await capture("atlas", "dark");
await capture("atlas", "light");
await page.getByRole("button", { name: "MLP lens", exact: true }).click();
await page.locator("#bake").selectOption("F1");
await page.waitForFunction(() =>
  document.querySelector("#audit-summary").textContent.includes("126,080"),
);
await page.locator("#kind").selectOption("dead-input");
await page.waitForFunction(
  () => document.querySelectorAll("#concern-table tbody tr").length === 2,
);
await page.locator("#concern-table tbody tr").first().click();
await page.waitForFunction(() =>
  document
    .querySelector("#weight-detail")
    .textContent.includes("all 128 weights are zero: dead input"),
);
assert.match(await page.locator("#concern-table").textContent(), /critical/);
await page.locator("#bank").selectOption("hdr-teacher");
await page.waitForFunction(() =>
  document
    .querySelector("#status")
    .textContent.includes("no exported evidence"),
);
assert.equal(await page.locator("#concern-table tbody tr").count(), 0);
assert.deepEqual(errors, []);
if (process.env.VIZ_CAPTURE_DIR)
  writeFileSync(
    `${process.env.VIZ_CAPTURE_DIR}/browser-review.json`,
    JSON.stringify(
      {
        review: "DESIGN-REVIEW-1",
        checks:
          "grouping/paging; magnitude notes; exact zeros; native band/centered ICE; minimap/zoom IDs; ranking names; overlay alpha; percentile/signed gating; globals; empty pins; both themes; missing bank",
        uncaught_errors: errors,
      },
      null,
      2,
    ),
  );
console.log(
  "Design review 1 browser checks passed; native populations/flags remain fully exported.",
);
await browser.close();
