const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const url = process.argv[2], output = process.argv[3], family = process.argv[4];
const extensions = process.argv[5] === 'extensions';
const extensionCell = process.argv[6] || null;
if (!url || !output) throw Error('Usage: node shaping_dashboard_browser.cjs URL OUTPUT_DIRECTORY');
(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto(url); await page.waitForFunction(() => document.querySelectorAll('#comparison tbody tr').length > 0);
    const approvedProgress = await page.evaluate(() => {
      if (!data.scope_amendment) return null;
      if (!data.approved_progress) throw Error('Approved progress is missing from the reduced viewer');
      const expected=data.approved_progress.counts.rescue;
      const card=[...document.querySelectorAll('#cards .card')].find(e=>e.querySelector('small').textContent==='Approved rescues trained');
      if(!card||card.querySelector('b').textContent!==expected.successful_fit_cells+' / '+expected.required)
        throw Error('Rendered rescue progress differs from the approved subset');
      if(!document.querySelector('#coverage').textContent.includes(data.approved_progress.unresolved_fold_cells+' rescue/outer cell assignments'))
        throw Error('Rendered unresolved assignments differ from frozen shortlists');
      return {trained:expected.successful_fit_cells,required:expected.required,
        preserved:data.counts.rescue.successful_fit_cells,unresolved:data.approved_progress.unresolved_fold_cells};
    });
    let extensionCoverage = null;
    if (extensions) {
      await page.selectOption('#stage', 'exposure_extension');
      if (extensionCell) {
        const partition = await page.evaluate(id => {
          const row=data.extension_records.find(r=>r.cell.id===id && r.trained && r.details);
          if (!row) throw Error('Requested completed extension has no scatter: '+id);
          return row.cell.partition;
        },extensionCell);
        await page.selectOption('#partition',partition);
      }
      await page.waitForFunction(() => document.querySelectorAll('#comparison tbody tr').length > 0);
      extensionCoverage = await page.evaluate(extensionCell => {
        if (!data.extension_records.length) throw Error('Registered extension records are absent');
        if (data.records.some(r => r.cell.stage === 'exposure_extension')) throw Error('Extensions mixed into original-budget records');
        if (!document.querySelector('#setting').disabled) throw Error('Selected extension setting can be filtered away');
        if (rows().some(r => r.stage !== 'exposure_extension')) throw Error('Extension view includes another budget');
        const match=data.extension_comparisons.find(r => r.partition===document.querySelector('#partition').value &&
          (!extensionCell || r.cells.includes(extensionCell)) &&
          r.cells.some(id=>data.extension_records.find(v=>v.cell.id===id).curve.length));
        if (!match) throw Error('No saved extension curve in the selected partition');
        showDetail(match);
        if (!document.querySelector('#curves polyline')) throw Error('Saved extension curves did not render');
        if (document.querySelectorAll('#seeds .seed').length !== match.required) throw Error('Extension seed denominator differs');
        return {registered:data.extension_records.length,comparisons:data.extension_comparisons.length,shown:match.cells,curves:true};
      },extensionCell);
      if (extensionCell) await page.selectOption('#scatterFit',extensionCell);
      await page.waitForFunction(() => Number(document.querySelector('#scatterRaw').dataset.points) > 0 &&
        Number(document.querySelector('#scatterNormalized').dataset.points) > 0 &&
        document.querySelector('#scatterMessage').textContent.includes('Full native Zenstats is available.'));
      extensionCoverage.scatter_populations = {};
      for (const pop of ['kadid','tid2013','konfig','cid22_a25','cid22_ssim2']) {
        await page.selectOption('#scatterPopulation',pop);
        const count=Number(await page.locator('#scatterRaw').getAttribute('data-points'));
        if (!(count>0)) throw Error('No extension scalar points for '+pop);
        extensionCoverage.scatter_populations[pop]=count;
      }
      extensionCoverage.scatter_fit=await page.locator('#scatterFit').inputValue();
      if (!extensionCoverage.shown.includes(extensionCoverage.scatter_fit)) throw Error('Extension scatter belongs to another cohort');
      await page.selectOption('#scatterPopulation','cid22_a25');
      await page.locator('#detail').screenshot({path:path.join(output,'extensions.png')});
      await page.selectOption('#stage','inner');
      await page.waitForFunction(() => rows().length > 0 && rows().every(r=>r.stage==='inner'));
    }
    await page.click('#detailLink');
    if (family) {
      await page.evaluate(family => {
        const match=data.comparisons.find(r=>r.stage==='inner'&&r.partition===data.default_partition&&r.family===family&&r.setting===0);
        if(!match)throw Error('Requested family is absent: '+family);
        showDetail(match);
      },family);
    }
    await page.waitForFunction(() => Number(document.querySelector('#scatterRaw').dataset.points) > 0);
    await page.waitForFunction(() => Number(document.querySelector('#scatterNormalized').dataset.points) > 0);
    await page.waitForFunction(() => document.querySelector('#scatterMessage').textContent.includes('Full native Zenstats is available.'));
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
    const selected=await page.locator('#scatterFit').inputValue();
    const result = { passed: true, url, family:family||'relu', selected, populations, source_filter: { all, filtered }, native_panel_visible: true, mobile_overflow: false, page_errors: errors, approved_progress:approvedProgress, extension_coverage:extensionCoverage };
    fs.writeFileSync(path.join(output, 'RESULT.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
