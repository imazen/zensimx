const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const url = process.argv[2], output = process.argv[3];
if (!url || !output) throw Error('Usage: node shaping_dashboard_browser.cjs URL OUTPUT_DIRECTORY');
(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto(url); await page.waitForFunction(() => document.querySelectorAll('#comparison tbody tr').length > 0);
    await page.click('#detailLink');
    await page.waitForFunction(() => Number(document.querySelector('#scatterRaw').dataset.points) > 0);
    await page.waitForFunction(() => Number(document.querySelector('#scatterNormalized').dataset.points) > 0);
    const populations = {};
    for (const pop of ['kadid', 'tid2013', 'konfig', 'cid22_a25', 'cid22_ssim2']) {
      await page.selectOption('#scatterPopulation', pop);
      populations[pop] = await page.locator('#scatterRaw').getAttribute('data-points');
      if (!(Number(populations[pop]) > 0)) throw Error('No scalar points for ' + pop);
    }
    await page.selectOption('#scatterPopulation', 'cid22_a25');
    const all = Number(await page.locator('#scatterRaw').getAttribute('data-points'));
    await page.selectOption('#scatterSource', '0');
    const filtered = Number(await page.locator('#scatterRaw').getAttribute('data-points'));
    if (!(filtered > 0 && filtered < all)) throw Error('Source filter did not select a strict subset');
    await page.selectOption('#scatterSource', '');
    await page.selectOption('#scatterRole', 'train');
    if (!(Number(await page.locator('#scatterRaw').getAttribute('data-points')) > 0)) throw Error('No training scatter');
    await page.selectOption('#scatterRole', 'validation');
    await page.locator('#detailedPanels').screenshot({ path: path.join(output, 'desktop.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#detailedPanels').screenshot({ path: path.join(output, 'mobile.png') });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    if (overflow) { const boxes = await page.evaluate(() => [...document.querySelectorAll('*')].map(e => ({tag:e.tagName,id:e.id,className:String(e.className),right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})).filter(e => e.right > innerWidth + 1)); fs.writeFileSync(path.join(output,'OVERFLOW.json'),JSON.stringify(boxes,null,2)); throw Error('Page overflows the mobile viewport'); }
    if (errors.length) throw Error(errors.join('\n'));
    const result = { passed: true, url, populations, source_filter: { all, filtered }, native_panel_visible: true, mobile_overflow: false, page_errors: errors };
    fs.writeFileSync(path.join(output, 'RESULT.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
