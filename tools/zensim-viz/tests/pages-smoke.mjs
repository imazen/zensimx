import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url),
  { chromium } = require(process.env.VIZ_PLAYWRIGHT ?? "playwright");
const [url, out] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.VIZ_CHROMIUM,
  headless: true,
  args: ["--disable-dev-shm-usage"],
});
const errors = [],
  ranges = [],
  captures = [];
let checks = 0;
try {
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1000 },
  });
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("response", (r) => {
    if (r.url().endsWith(".imijset"))
      ranges.push({
        status: r.status(),
        range: r.request().headers().range,
        contentRange: r.headers()["content-range"],
      });
  });
  await page.goto(url);
  await page.waitForFunction(() => window.vizReady || window.vizError, null, {
    timeout: 120000,
  });
  const startupError = await page.evaluate(() => window.vizError);
  if (startupError) throw Error(startupError);
  const initial = await page.evaluate(() => ({
    error: window.vizError,
    inputs: viz.weights.inputs.length,
    flags: viz.audit.flags.length,
    entries: viz.archive.catalog.entries.length,
    source: viz.archive.catalog.source_commit,
    formula: !!document.querySelector(".katex"),
    pairMs: viz.state.pairMs,
  }));
  if (
    initial.error ||
    initial.inputs !== 984 ||
    initial.flags !== 34927 ||
    !initial.formula
  )
    throw Error(
      "Initial model/formula/flags failed: " + JSON.stringify(initial),
    );
  checks++;
  await page.evaluate(async () => {
    for (const [op, args] of [
      ["pair", [new Uint8Array(4), new Uint8Array(4), 1, 1, 4]],
      ["bank", [Uint32Array.of(0), Float32Array.of(0)]],
    ]) {
      let rejected = false;
      try {
        await viz.call(op, ...args);
      } catch {
        rejected = true;
      }
      if (!rejected) throw Error("Invalid ingress was accepted");
    }
    if (!(await viz.call("map", 0)).values.length)
      throw Error("Invalid pair destroyed retained atlas");
    if ((await viz.call("curve", 0, viz.gridFor(0))).rows !== 200)
      throw Error("Invalid bank destroyed retained lens");
  });
  checks++;
  const joins = await page.evaluate(() =>
    viz.weights.inputs.map((i) => ({
      index: i.index,
      name: i.name,
      feature: i.feature,
      factor: i.factor,
    })),
  );
  if (joins.some((i) => !i.name || !Number.isInteger(i.feature)))
    throw Error("Incomplete explained feature join");
  checks++;
  await page.locator("#search").fill("ringing");
  if ((await page.locator("#rows [data-input]").count()) === 0)
    throw Error("Search has no linked results");
  await page.locator("#search").fill("");
  checks++;
  const next = await page
    .locator("#rows .selected")
    .evaluate((n) =>
      Number(n.nextElementSibling.querySelector("[data-input]").dataset.input),
    );
  await page.locator("#tree").focus();
  await page.keyboard.press("ArrowDown");
  await page.waitForFunction((next) => viz.state.input === next, next);
  checks++;
  for (const bank of ["sdr", "hdr"]) {
    await page.locator("#bank").selectOption(bank);
    await page.waitForFunction(
      (bank) =>
        viz.state.bank === bank &&
        document
          .querySelector("#standing")
          .textContent.includes(bank.toUpperCase()),
      bank,
    );
    await page.locator('[data-view="lens"]').click();
    await page.waitForFunction(
      () =>
        viz.state.view === "lens" &&
        viz.state.curve?.raw_feature ===
          viz.weights.inputs[viz.state.input].feature,
    );
    for (const mode of [
      "input",
      "signal",
      "family",
      "diagnostics",
      "quantiles",
      "table",
    ]) {
      await page.locator("#lens-mode").selectOption(mode);
      await page.waitForFunction(
        (mode) => {
          const s = document.querySelector("#status").textContent;
          return (
            viz.state.lensMode === mode &&
            !/Computing|Loading/.test(s) &&
            /forward|Native|native|Exact/.test(s)
          );
        },
        mode,
        { timeout: 120000 },
      );
      checks++;
    }
  }
  await page.locator("#lens-mode").selectOption("input");
  await page.locator('[data-view="function"]').click();
  await page.waitForFunction(() =>
    document
      .querySelector("#table")
      .textContent.includes("approximating function"),
  );
  if (
    (await page.locator("#table [data-input]").count()) < 1 ||
    (await page.locator("#capacity a").count()) !== 128
  )
    throw Error("Function has missing feature links/capacity");
  checks++;
  await page.locator("#unit").selectOption("31");
  await page.waitForFunction(() => viz.state.unit === 31);
  checks++;
  await page.evaluate(() => {
    location.hash = "input=0&view=weights&unit=31&bank=sdr";
  });
  await page.waitForFunction(
    () =>
      viz.state.view === "weights" &&
      viz.state.bank === "sdr" &&
      document.querySelector("#matrix-labels"),
  );
  await page.locator("[data-all-concerns]").click();
  const flag = await page.evaluate(() => {
    const f = viz.audit.flags.find((f) => f.layer === 1 && f.from != null);
    return f;
  });
  if (!flag) throw Error("Expected preserved output-layer concerns");
  await page.evaluate((unit) => {
    location.hash = `input=0&view=function&unit=${unit}&bank=sdr`;
  }, flag.from);
  await page.waitForFunction(
    (unit) =>
      viz.state.unit === unit &&
      document
        .querySelector("#table")
        .textContent.includes("Selected h" + unit),
    flag.from,
  );
  checks++;
  await page.evaluate(async () => {
    viz.state.input = viz.weights.inputs.find(
      (i) => i.signal === "var_loss",
    ).index;
    viz.state.view = "atlas";
    await viz.render();
  });
  if (
    !(await page.locator("#chart").evaluate((n) => n.hidden)) ||
    !(await page.locator("#table").evaluate((n) => n.hidden))
  )
    throw Error("Global statistic showed a spatial or unrelated exhibit");
  checks++;
  await page.evaluate(() => {
    viz.state.input = 0;
  });
  for (const [width, height] of [
    [1600, 1000],
    [1280, 800],
  ]) {
    await page.setViewportSize({ width, height });
    for (const theme of ["dark", "light"]) {
      await page.evaluate((theme) => {
        document.documentElement.dataset.theme = theme;
      }, theme);
      for (const view of ["atlas", "lens", "function", "weights"]) {
        await page.evaluate(async (view) => {
          viz.state.view = view;
          viz.state.unit = null;
          viz.state.allConcerns = false;
          viz.state.concernPage = 0;
          await viz.render();
        }, view);
        await page.evaluate(() => {
          for (const s of ["#reference", "#table", "#exhibit"])
            document.querySelector(s).scrollTop = 0;
        });
        const layout = await page.evaluate(() => ({
          height: document.documentElement.scrollHeight,
          viewport: innerHeight,
          width: document.documentElement.scrollWidth,
          viewportWidth: innerWidth,
          formula: !!document.querySelector(".katex"),
          stage: document
            .querySelector("#stage")
            .getBoundingClientRect()
            .toJSON(),
        }));
        if (
          layout.height !== height ||
          layout.width !== width ||
          !layout.formula
        )
          throw Error("Viewport/formula regression " + JSON.stringify(layout));
        if (
          view === "function" &&
          !(await page
            .locator(".function-plot")
            .first()
            .evaluate(
              (n) =>
                n.getBoundingClientRect().bottom <=
                document.querySelector("#exhibit").getBoundingClientRect()
                  .bottom,
            ))
        )
          throw Error("Function hinge clipped below the exhibit");
        const file = `${view}-${theme}-${width}x${height}.png`;
        await page.screenshot({ path: out + "/" + file });
        captures.push({ file, view, theme, width, height, layout });
        checks++;
      }
    }
  }
  const bytes = await page.evaluate(async () => {
    let bytes = 0;
    for (const e of viz.archive.catalog.entries)
      bytes += (await viz.archive.bytes(e.name)).length;
    return bytes;
  });
  checks++;
  if (
    !ranges.length ||
    ranges.some((r) => r.status !== 206 || !r.range || !r.contentRange)
  )
    throw Error("Range reader did not receive strict HTTP 206");
  if (errors.length) throw Error("Browser errors: " + errors.join("; "));
  writeFileSync(
    out + "/smoke.json",
    JSON.stringify(
      {
        passed: true,
        checks,
        initial,
        joins: joins.length,
        allEntryPayloadBytes: bytes,
        browser_range_verified: true,
        ranges,
        captures,
        errors,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(
    JSON.stringify({
      passed: true,
      checks,
      captures: captures.length,
      ranges: ranges.length,
    }),
  );
} catch (e) {
  writeFileSync(
    out + "/failure.json",
    JSON.stringify(
      { passed: false, error: String(e), errors, ranges, captures },
      null,
      2,
    ) + "\n",
  );
  throw e;
} finally {
  await browser.close();
}
