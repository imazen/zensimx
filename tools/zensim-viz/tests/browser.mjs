// Real-browser interaction checks. VIZ_PLAYWRIGHT points to an existing local
// Playwright installation; screenshots always describe the loaded evidence.
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.VIZ_PLAYWRIGHT||'playwright');
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage'],executablePath:process.env.VIZ_CHROMIUM});
const page=await browser.newPage({viewport:{width:1600,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(process.env.VIZ_URL||'http://dev.lan:3310/');
await page.waitForFunction(()=>document.querySelector('#status').textContent!=='Loading exported evidence…');
await page.getByRole('button',{name:'MLP lens',exact:true}).click();
await assert.equal(await page.locator('#lens').isVisible(),true);
await page.getByRole('button',{name:'Light theme',exact:true}).click();
assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
await page.getByRole('button',{name:'Dark theme',exact:true}).click();
const emptyMode=process.env.VIZ_EMPTY_STATE==='1';
if(!emptyMode){
 assert.ok(await page.locator('#concern-table tbody tr').count()>0,'real audit evidence must be loaded');
 await page.locator('#concern-table tbody tr').first().click();
 assert.match(await page.locator('#weight-detail').textContent(),/input/);
 await page.locator('#weight-matrix').focus();
 const before=await page.locator('#weight-detail').textContent();
 await page.keyboard.press('ArrowRight');
 const after=await page.locator('#weight-detail').textContent();
 assert.notEqual(before,after);
 await page.getByRole('button',{name:'Table view',exact:true}).first().click();
}
if(process.env.VIZ_SCREENSHOT)await page.screenshot({path:process.env.VIZ_SCREENSHOT,fullPage:true});
await page.getByRole('button',{name:'Atlas',exact:true}).click();
const cells=page.locator('#matrix .cell');
if(!emptyMode){
 assert.ok(await cells.count()>0,'real atlas evidence must be loaded');
 await cells.first().focus();await page.keyboard.press('Enter');
 assert.equal(await page.locator('#pins figure').count(),1);
 await page.keyboard.press('Backspace');assert.equal(await page.locator('#pins figure').count(),0);
 await cells.first().hover();await page.waitForTimeout(100);
 assert.match(await page.locator('#feature-detail').textContent(),/value/);
}
assert.deepEqual(errors,[]);
console.log(emptyMode?'Browser empty-state checks passed: view switching, themes and no uncaught page errors.':'Browser evidence checks passed: audit selection, weight keyboard navigation, map preview and pin/unpin.');
await browser.close();
