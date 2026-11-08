const {chromium}=require('playwright');
const fs=require('fs'),path=require('path');
(async()=>{
  const [url,out]=process.argv.slice(2);fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));await page.goto(url);
    await page.waitForFunction(()=>document.querySelectorAll('#rows tr').length>0);
    const points={};
    for(const [population,expected] of [['csiq/all',866],['live/clean10',267],['live/all',779]]){
      await page.selectOption('#population',population);
      await page.waitForFunction(n=>document.querySelectorAll('#scatter circle').length===n,expected);
      points[population]=await page.locator('#scatter circle').count();
    }
    await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');
    if(errors.length)throw Error(errors.join('\n'));
    const result={passed:true,url,points,page_errors:errors};fs.writeFileSync(path.join(out,'RESULT.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
