// Optional offline UI check. Node >=22 and a local Chromium; never calls an API.
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const [suppliedHtml,suppliedOutput]=process.argv.slice(2);
const expectedModels=Number(process.argv[4]||7);
if(!suppliedHtml||!suppliedOutput)throw Error('Usage: node check_dashboard_browser.mjs dashboard.html output-directory');
const html=path.resolve(suppliedHtml),out=path.resolve(suppliedOutput);
fs.mkdirSync(out,{recursive:true});
const temporary=fs.mkdtempSync(path.join(path.dirname(html),'browser-check-'));
const target=path.join(temporary,'dashboard.html');fs.copyFileSync(html,target);
const profile=path.join(temporary,'profile'),fd=fs.openSync(path.join(temporary,'chromium.log'),'w');
const browser=spawn(process.env.CHROMIUM_BIN||'/snap/bin/chromium',['--headless','--no-sandbox','--disable-gpu','--no-first-run',
 '--disable-background-networking','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
 '--user-data-dir='+profile,'--host-resolver-rules=MAP * 0.0.0.0','--window-size=1440,1200',pathToFileURL(target).href],{stdio:['ignore',fd,fd]});
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let ws,nextId=0;const pending=new Map();
const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const deadline=setTimeout(()=>{browser.kill('SIGTERM');console.error('Browser verification timed out');process.exit(1)},45000);
try{
 let port;
 for(let i=0;i<100;i++){try{port=fs.readFileSync(profile+'/DevToolsActivePort','utf8').split('\n')[0];break}catch{}await pause(100)}
 if(!port)throw Error('Missing browser debugging port');
 const pages=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json();
 const page=pages.find(p=>p.type==='page');if(!page)throw Error('Missing page');
 ws=new WebSocket(page.webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
 ws.addEventListener('message',event=>{const msg=JSON.parse(event.data),wait=pending.get(msg.id);if(wait){pending.delete(msg.id);msg.error?wait.reject(Error(JSON.stringify(msg.error))):wait.resolve(msg.result)}});
 const evaluate=async expression=>{const result=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value};
 for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('#radar g[data-model]').length>0"))break;if(i===99)throw Error('Dashboard did not render');await pause(100)}
 const result=await evaluate(`(async()=>{
  const checks=[],check=(name,actual,expected)=>checks.push({name,actual,expected,pass:JSON.stringify(actual)===JSON.stringify(expected)});
  const count=()=>document.querySelectorAll('#radar g[data-model]').length;
  const change=(id,value)=>{$(id).value=value;$(id).dispatchEvent(new Event('change'))};
  check('seven_axes',document.querySelectorAll('#radar .axis-link').length,7);
  check('expected_real_models',count(),${expectedModels});
  const expectedPoints=data.models.reduce((n,m)=>n+6+Number(Number.isFinite(m.metrics.speed.score)),0);
  check('all_available_points',document.querySelectorAll('#radar circle').length,expectedPoints);
  check('expected_table_rows',$('table-body').rows.length,${expectedModels});
  check('no_grading_selector',document.getElementById('protocol'),null);
  check('effective_denominator',Number($('core-count').textContent),data.core_items);
  check('original_runtime_requests',data.models.every(m=>m.metrics.core.run_n===data.run_items&&m.metrics.speed.timed_requests===m.metrics.core.ok),true);
  check('grading_mode_visible',$('grading-note').textContent.includes(data.scoring_protocol==='vlm-public-label-scoring-1'?'公开标签':'复核'),true);
  check('dataset_denominators',data.models.every(m=>Object.entries(m.metrics.accuracy).every(([id,v])=>v.n===data.dataset_counts[id])),true);
  const before=data.models[0].metrics.speed.score;
  document.querySelector('#models input').click();check('checkbox_removes_curve',count(),${expectedModels}-1);
  check('fixed_speed_scale',data.models[0].metrics.speed.score,before);
  $('select-none').click();check('empty_selection',count(),0);
  $('select-all').click();check('restore_all_models',count(),${expectedModels});
  document.querySelector('[data-axis="speed"]').click();check('axis_details',$('axis-title').textContent.includes('响应速度'),true);
  document.querySelector('#radar .axis-link').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  check('keyboard_axis_selection',state.axis,'MMStar');
  change('deployment','api');check('real_api_included',count(),data.models.filter(m=>m.group==='api').length);
  check('api_context_notice',!$('api-note').classList.contains('hidden'),data.models.some(m=>m.group==='api'));
  change('deployment','all');
  const fixture=JSON.parse(JSON.stringify(data.models[0]));fixture.id='synthetic-browser-fixture';fixture.label='missing speed fixture';
  fixture.metrics.speed.score=null;fixture.metrics.speed.missing_reason='fixture';
  data.models.push(fixture);state.selected=new Set([fixture.id]);render();
  const curve=document.querySelector('#radar g[data-model]');
  check('missing_speed_not_zero',curve.querySelectorAll('circle').length,6);
  check('missing_speed_no_filled_polygon',curve.querySelectorAll('polygon').length,0);
  data.models.pop();state.selected=new Set(data.models.map(m=>m.id));render();
  check('real_data_restored',count(),${expectedModels});
  const blobs=[],create=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;
  URL.createObjectURL=blob=>{blobs.push(blob);return create(blob)};
  HTMLAnchorElement.prototype.click=function(){if(!this.download)click.call(this)};
  $('export-svg').click();$('export-json').click();URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click;
  const exportedSvg=await blobs[0].text(),exportedJson=JSON.parse(await blobs[1].text());
  const xml=new DOMParser().parseFromString(exportedSvg,'image/svg+xml');
  check('svg_export_valid',xml.querySelectorAll('parsererror').length,0);
  check('svg_export_models',xml.querySelectorAll('g[data-model]').length,${expectedModels});
  check('json_export_selection',exportedJson.selection.models.length,${expectedModels});
  check('export_has_formula',exportedSvg.includes('耗时中位数秒数'),true);
  check('zero_external_resources',performance.getEntriesByType('resource').filter(r=>!r.name.startsWith('data:')).length,0);
  return {checks,exportedSvg};
 })()`);
 fs.writeFileSync(path.join(out,'seven-axis.svg'),result.exportedSvg);
 const desktop=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 fs.writeFileSync(path.join(out,'preview.png'),Buffer.from(desktop.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 const mobile=await evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth,curves:document.querySelectorAll('#radar g[data-model]').length})");
 result.checks.push({name:'mobile_no_page_overflow',actual:mobile.scroll,expected:mobile.width,pass:mobile.scroll<=mobile.width});
 const mobileShot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 fs.writeFileSync(path.join(out,'preview-mobile.png'),Buffer.from(mobileShot.data,'base64'));
 const report={engine:'Chromium headless',checked_utc:new Date().toISOString(),isolated_html_only:true,
  external_requests:0,missing_speed_check_uses_synthetic_fixture:true,checks:result.checks,all_checks_passed:result.checks.every(c=>c.pass)};
 fs.writeFileSync(path.join(out,'browser-validation.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));if(!report.all_checks_passed)process.exitCode=1;
}finally{
 clearTimeout(deadline);
 if(ws?.readyState===WebSocket.OPEN){await send('Browser.close').catch(()=>{});ws.close()}
 browser.kill('SIGTERM');fs.closeSync(fd);await pause(200);fs.rmSync(temporary,{recursive:true,force:true});
}
