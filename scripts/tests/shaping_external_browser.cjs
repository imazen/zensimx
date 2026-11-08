const {chromium}=require('playwright');
const fs=require('fs'),path=require('path');
(async()=>{
  const [url,out,expectedModels]=process.argv.slice(2);fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));await page.goto(url);
    await page.waitForFunction(()=>document.querySelectorAll('#rows tr').length>0);
    const points={};let checkedModelPlots=0;
    for(const [population,expected] of [['csiq/all',866],['live/clean10',267],['live/all',779]]){
      await page.selectOption('#population',population);
      const models=await page.locator('#model option').evaluateAll(rows=>rows.map(row=>row.value));
      if(expectedModels&&models.length!==Number(expectedModels))throw Error('Unexpected completed model count');
      for(const model of models){
        await page.selectOption('#model',model);
        await page.waitForFunction(({model,population,expected})=>{
          const svg=document.querySelector('#scatter');
          return svg.dataset.model===model&&svg.dataset.population===population&&svg.querySelectorAll('circle').length===expected;
        },{model,population,expected});
        checkedModelPlots++;
      }
      points[population]=await page.locator('#scatter circle').count();
    }
    // Hold an older response until the newer population is already displayed.
    let releaseHeld,markStarted;
    const held=new Promise(resolve=>{releaseHeld=resolve;}),started=new Promise(resolve=>{markStarted=resolve;});
    const heldUrl='**/csiq/all-scatter.json';
    await page.route(heldUrl,async route=>{markStarted();await held;await route.continue();});
    await page.selectOption('#population','csiq/all');await started;
    await page.selectOption('#population','live/clean10');
    await page.waitForFunction(()=>document.querySelector('#scatter').dataset.population==='live/clean10');
    releaseHeld();await page.waitForLoadState('networkidle');
    if(await page.locator('#scatter').getAttribute('data-population')!=='live/clean10'||await page.locator('#scatter circle').count()!==267)throw Error('Stale response replaced the selected population');
    await page.unroute(heldUrl);
    await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');
    if(errors.length)throw Error(errors.join('\n'));
    const result={passed:true,url,points,checked_model_plots:checkedModelPlots,stale_response_preserved_selection:true,page_errors:errors};fs.writeFileSync(path.join(out,'RESULT.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
