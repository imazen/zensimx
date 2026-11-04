// Real-browser interaction checks. VIZ_PLAYWRIGHT points to an existing local
// Playwright installation; screenshots always describe the loaded evidence.
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import { existsSync, writeFileSync } from "node:fs";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.VIZ_PLAYWRIGHT || "playwright");
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
  executablePath: process.env.VIZ_CHROMIUM,
});
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(process.env.VIZ_URL || "http://dev.lan:3310/");
await page.waitForFunction(
  () =>
    document.querySelector("#status").textContent !==
    "Loading exported evidence…",
);
await page.getByRole("button", { name: "MLP lens", exact: true }).click();
await assert.equal(await page.locator("#lens").isVisible(), true);
await page.getByRole("button", { name: "Light theme", exact: true }).click();
assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
await page.getByRole("button", { name: "Dark theme", exact: true }).click();
const scope =
  process.env.VIZ_TEST_SCOPE ||
  (process.env.VIZ_EMPTY_STATE === "1" ? "frame" : "all");
assert.ok(["frame", "audit", "all"].includes(scope));
const emptyMode = scope === "frame";
if (!emptyMode) {
  assert.ok(
    (await page.locator("#concern-table tbody tr").count()) > 0,
    "real audit evidence must be loaded",
  );
  await page.locator("#concern-table tbody tr").first().click();
  assert.match(await page.locator("#weight-detail").textContent(), /input/);
  await page.locator("#weight-matrix").focus();
  const before = await page.locator("#weight-detail").textContent();
  await page.keyboard.press("ArrowRight");
  const after = await page.locator("#weight-detail").textContent();
  assert.notEqual(before, after);
  await page
    .getByRole("button", { name: "Table view", exact: true })
    .first()
    .click();
}
if (process.env.VIZ_SCREENSHOT) {
  assert.equal(
    existsSync(process.env.VIZ_SCREENSHOT),
    false,
    "preserve existing screenshots",
  );
  await page.screenshot({ path: process.env.VIZ_SCREENSHOT, fullPage: true });
}
await page.getByRole("button", { name: "Atlas", exact: true }).click();
const cells = page.locator("#matrix .cell");
if (scope === "all") {
  assert.ok((await cells.count()) > 0, "real atlas evidence must be loaded");
  await cells.first().focus();
  await page.keyboard.press("Enter");
  assert.equal(await page.locator("#pins figure").count(), 1);
  await page.keyboard.press("Backspace");
  assert.equal(await page.locator("#pins figure").count(), 0);
  await cells.first().hover();
  await page.waitForTimeout(100);
  assert.match(await page.locator("#feature-detail").textContent(), /value/);
}
if (scope === "all" && process.env.VIZ_TEST_COMPLETE === "1") {
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    "open data tables must scroll inside their panel",
  );
  await page.waitForLoadState("networkidle");
  const latency = await page.evaluate(async () => {
    const samples = [],
      overlay = document.querySelector("#overlay");
    const cells = [
      ...document.querySelectorAll("#matrix .cell:not(.hollow)"),
    ].slice(1, 9);
    for (const cell of cells) {
      const elapsed = await new Promise((resolve) => {
        const start = performance.now();
        const observer = new MutationObserver(() => {
          if (overlay.dataset.map === cell.dataset.feature) {
            observer.disconnect();
            resolve(performance.now() - start);
          }
        });
        observer.observe(overlay, {
          attributes: true,
          attributeFilter: ["data-map"],
        });
        cell.dispatchEvent(new PointerEvent("pointerenter"));
      });
      samples.push(elapsed);
    }
    return samples;
  });
  console.log("Preloaded hover milliseconds:", JSON.stringify(latency));
  assert.ok(
    latency.every((v) => v <= 50),
    "every measured preloaded hover must finish within 50 ms",
  );
  await page.locator("#family-lane").check();
  await page.locator("#matrix .cell:not(.hollow)").first().hover();
  await page.waitForFunction(
    () => document.querySelectorAll("#family-maps canvas").length === 4,
  );
  await page.locator("#family-lane").uncheck();
  await page.locator("#colour").selectOption("signed");
  assert.match(await page.locator("#matrix-range").textContent(), /0/);
  await page.locator("#map-only").check();
  assert.equal(
    await page
      .locator("#pair-image")
      .evaluate((e) => getComputedStyle(e).visibility),
    "hidden",
  );
  await page.locator("#map-only").uncheck();
  await page.locator("#pair").selectOption("pair-06");
  await page.waitForFunction(() =>
    document.querySelector("#subject").textContent.includes("Flower"),
  );
  await page.locator("#pair").selectOption("pair-01");
  await page.waitForLoadState("networkidle");
  const capture = async (name) => {
    if (!process.env.VIZ_CAPTURE_DIR) return;
    const path = `${process.env.VIZ_CAPTURE_DIR}/${name}.png`;
    assert.equal(existsSync(path), false, "preserve earlier captures");
    await page.screenshot({ path, fullPage: true });
  };
  await page.locator("#colour").selectOption("percentile");
  await capture("atlas-dark");
  await page.locator("#theme").click();
  await capture("atlas-light");
  await page.locator("#theme").click();
  await page.getByRole("button", { name: "MLP lens", exact: true }).click();
  await page.waitForSelector("#pdp-chart .mean");
  assert.equal(await page.locator("#pdp-chart .ice").count(), 40);
  await page.locator("#compare").selectOption("E");
  await page.waitForSelector("#pdp-chart .compare");
  assert.ok(await page.locator("#comparison-panel").isVisible());
  await page.locator("#lens-family").selectOption("v2");
  await page.waitForFunction(
    () => document.querySelector("#pdp-title").textContent === "v2 family",
  );
  await page.locator("#lens-family").selectOption("");
  await page.waitForFunction(() =>
    document.querySelector("#pdp-title").textContent.startsWith("f"),
  );
  await page.locator("#kind").selectOption("monotonicity");
  assert.ok(
    (await page.locator("#concern-table tbody tr").count()) > 0,
    "native monotonicity counterexamples must be exposed",
  );
  await page.locator("#kind").selectOption("");
  if (await page.locator("#pdp-table").isVisible())
    await page.locator("[data-table=pdp-table]").click();
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    "tables must stay within the viewport",
  );
  await capture("mlp-lens-dark");
  await page.locator("#theme").click();
  await capture("mlp-lens-light");
  await page.locator("#theme").click();
  await page.locator("#bank").selectOption("hdr-teacher");
  await page.waitForFunction(() =>
    document.querySelector("#audit-summary").textContent.includes("7,390"),
  );
  await page.waitForSelector("#pdp-chart .mean");
  assert.equal(await page.locator("#pdp-chart .ice").count(), 40);
  await page.locator("#bake").selectOption("F1");
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.includes("no exported evidence"),
  );
  assert.equal(await page.locator("#concern-table tbody tr").count(), 0);
  await page.locator("#bank").selectOption("cid22-fit");
  await page.waitForSelector("#pdp-chart .mean");
  if (process.env.VIZ_CAPTURE_DIR)
    writeFileSync(
      `${process.env.VIZ_CAPTURE_DIR}/browser-hover-latency.json`,
      JSON.stringify(
        {
          milliseconds: latency,
          scope:
            "decoded visible-family maps; pointerenter to committed overlay canvas",
        },
        null,
        2,
      ),
    );
}
assert.deepEqual(errors, []);
console.log(
  emptyMode
    ? "Browser frame checks passed: view switching, themes and no uncaught page errors."
    : scope === "audit"
      ? "Browser audit checks passed: real concerns, weight selection, keyboard movement, themes and no page errors."
      : "Browser evidence checks passed: audit selection, weight keyboard navigation, map preview and pin/unpin.",
);
await browser.close();
