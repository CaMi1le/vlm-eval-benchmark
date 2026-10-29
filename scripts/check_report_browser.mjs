// Verify the raw campaign viewer under file://, with external DNS disabled.
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const target=path.resolve(process.argv[2]),resultFile=process.argv[3],cache=path.dirname(target);
const expectedModels=Number(process.argv[4]||4);
fs.mkdirSync(cache,{recursive:true});
// Snap Chromium has a private /tmp; use a shared workspace cache for DevTools.
const temporary=fs.mkdtempSync(path.join(cache,'campaign-browser-'));
const profile=path.join(temporary,'profile'),fd=fs.openSync(path.join(temporary,'chromium.log'),'w');
const browser=spawn(process.env.CHROMIUM_BIN||'/snap/bin/chromium',['--headless','--no-sandbox','--disable-gpu','--no-first-run',
 '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,
 '--host-resolver-rules=MAP * 0.0.0.0','--window-size=1440,1200',pathToFileURL(target).href],{stdio:['ignore',fd,fd]});
const pause=ms=>new Promise(r=>setTimeout(r,ms));let ws,nextId=0;const pending=new Map();
const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const deadline=setTimeout(()=>{browser.kill('SIGTERM');console.error('Browser verification timed out');process.exit(1)},55000);
try{
 let port;
 for(let i=0;i<100;i++){try{port=fs.readFileSync(profile+'/DevToolsActivePort','utf8').split('\n')[0];break}catch{}await pause(100)}
 if(!port)throw Error('Missing DevTools port');
 const pages=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json(),page=pages.find(p=>p.type==='page');
 ws=new WebSocket(page.webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
 ws.addEventListener('message',e=>{const m=JSON.parse(e.data),p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}});
 for(let i=0;i<100;i++){
  const r=await send('Runtime.evaluate',{expression:"document.querySelectorAll('article').length===12",returnByValue:true});
  if(r.result.value)break;if(i===99)throw Error('Viewer did not render');await pause(100);
 }
 const expression=`(async()=>{
 const check=(ok,message)=>{if(!ok)throw Error(message)},$=id=>document.getElementById(id),
 data=JSON.parse($('data').textContent),event=new Event('input');
 check(location.protocol==='file:','Not testing file protocol');check(data.cases.length===data.views.core.length,'Wrong question count');
 check(data.models.length===${expectedModels},'Wrong model count');let imagesChecked=0;
 for(const dataset of [...new Set(data.cases.map(c=>c.dataset))]){
  $('dataset').value=dataset;$('dataset').dispatchEvent(event);
  check(document.querySelectorAll('article').length===12,'Pagination failed');
  const imgs=[...document.querySelectorAll('article img')];
  await Promise.all(imgs.map(img=>new Promise((resolve,reject)=>{img.loading='eager';if(img.complete){img.naturalWidth?resolve():reject(Error('Broken image '+img.src))}else{img.onload=resolve;img.onerror=()=>reject(Error('Broken image '+img.src))}})));
  check(imgs.every(i=>i.naturalWidth>0),'Image decode failed');imagesChecked+=imgs.length;
 }
 $('dataset').value='';$('dataset').dispatchEvent(event);
 for(const [view,count] of ['regression','delta','core'].map(view=>[view,data.views[view].length])){
  $('view').value=view;$('view').dispatchEvent(event);check($('count').textContent.startsWith(count+' 题'),'Wrong view count');
 }
 $('next').click();check($('count').textContent.includes('2 /'),'Next page failed');$('prev').click();
 const chosen=data.cases.find(c=>data.cases.filter(o=>[o.id,o.prompt,o.category].join(' ').toLowerCase().includes(c.id.toLowerCase())).length===1);
 check(Boolean(chosen),'No unique search fixture');$('search').value=chosen.id;$('search').dispatchEvent(event);
 check(document.querySelectorAll('article').length===1,'Question search failed');
 check(document.querySelectorAll('.answer').length===${expectedModels},'Missing raw answers');
 document.querySelector('article .media a').click();
 check($('image').open,'Original image dialog failed');await $('original').decode();
 check($('original').naturalWidth===chosen.media[0].width,'Original image width changed');$('close').click();
 $('search').value='';$('search').dispatchEvent(event);
 if(data.scoring_protocol==='vlm-public-label-scoring-1'){
  const count=() => Number($('count').textContent.split(' ')[0]);
  for(const [filter,expected] of [
   ['different',data.cases.filter(c=>new Set(data.models.map(m=>c.models[m].image.correct)).size>1).length],
   ['allwrong',data.cases.filter(c=>data.models.every(m=>!c.models[m].image.correct)).length],
   ['unresolved',data.cases.filter(c=>data.models.some(m=>c.models[m].image.parsed===null)).length]]){
   $('filter').value=filter;$('filter').dispatchEvent(event);check(count()===expected,'Scored filter failed: '+filter);
  }
  const selectedModel=data.models.find(m=>data.performance[m].config?.backend==='openai_compatible_api')||data.models[0];
  $('model').value=selectedModel;$('filter').value='wrong';$('filter').dispatchEvent(event);
  check(count()===data.cases.filter(c=>!c.models[selectedModel].image.correct).length,'Per-model wrong-answer filter failed');
  $('filter').value='all';$('model').value='';$('filter').dispatchEvent(event);
  check(document.querySelector('.answer').textContent.includes('提取答案：'),'Missing extraction evidence');
  check(!document.querySelector('h1+p').textContent.includes('待本轮内容审查'),'Stale ungraded viewer text');
 }
 return {all_checks_passed:true,protocol:location.protocol,cases:data.cases.length,models:data.models.length,
 images_decoded:imagesChecked,views:['core','regression','delta'].map(view=>data.views[view].length),pagination_verified:true,original_dialog_verified:true,scored_filters_verified:Boolean(data.scoring_protocol)};
})()`;
 const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
 if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));
 if(resultFile)fs.writeFileSync(resultFile,JSON.stringify(r.result.value,null,2)+'\n');
 const screenshot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 if(resultFile)fs.writeFileSync(path.join(path.dirname(resultFile),'review-preview.png'),Buffer.from(screenshot.data,'base64'));
 console.log(JSON.stringify(r.result.value));
}finally{
 clearTimeout(deadline);
 if(ws?.readyState===WebSocket.OPEN){await send('Browser.close').catch(()=>{});ws.close()}
 browser.kill('SIGTERM');fs.closeSync(fd);await pause(200);fs.rmSync(temporary,{recursive:true,force:true});
}
