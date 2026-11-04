import { createRequire } from "node:module";
import assert from "node:assert/strict";
const scope =
  process.env.VIZ_TEST_SCOPE ||
  (process.env.VIZ_EMPTY_STATE === "1" ? "frame" : "all");
assert.ok(["frame", "audit", "all"].includes(scope));
if (scope === "all") {
  await import("./design2.mjs");
  const capture = process.env.VIZ_CAPTURE_DIR;
  delete process.env.VIZ_CAPTURE_DIR;
  await import("./review1.mjs");
  if (capture) process.env.VIZ_CAPTURE_DIR = capture;
} else {
  const require = createRequire(import.meta.url),
    { chromium } = require(process.env.VIZ_PLAYWRIGHT || "playwright");
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
    executablePath: process.env.VIZ_CHROMIUM,
  });
  const page = await browser.newPage({
      viewport: { width: 1600, height: 1000 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(process.env.VIZ_URL || "http://dev.lan:3310/");
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent !==
      "Loading exported evidence…",
  );
  await page.locator("[data-view=lens]").click();
  assert.equal(await page.locator("#lens").isVisible(), true);
  await page.locator("#theme").click();
  assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
  await page.locator("#theme").click();
  if (scope === "audit") {
    await page.locator("[data-exhibit-tab=Concerns]").click();
    assert.ok((await page.locator("#concern-table tbody tr").count()) > 0);
    await page.locator("#concern-table tbody tr").first().click();
    await page.locator("[data-exhibit-tab=Weights]").click();
    assert.match(await page.locator("#weight-detail").textContent(), /input/);
    await page.locator("#weight-matrix").focus();
    const before = await page.locator("#weight-detail").textContent();
    await page.keyboard.press("ArrowRight");
    assert.notEqual(await page.locator("#weight-detail").textContent(), before);
  }
  await page.locator("[data-view=atlas]").click();
  assert.equal(await page.locator("#atlas").isVisible(), true);
  assert.deepEqual(errors, []);
  await browser.close();
  console.log(`Browser ${scope} checks passed.`);
}
