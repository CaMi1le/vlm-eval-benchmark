"""Self-contained SVG dashboard; no network, frameworks, or external assets."""

TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM 七轴能力对照</title>
<style>
:root{color-scheme:light;--ink:#152d3d;--muted:#526877;--line:#dce5ea;--blue:#155da7;--paper:#fff;--page:#f3f6f7}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}main{max-width:1480px;margin:auto;padding:32px}h1,h2,h3,p{margin:0}h1{font-size:34px;letter-spacing:-1px;line-height:1.3}h2{font-size:21px}h3{font-size:16px}.eyebrow{color:var(--blue);font-size:12px;font-weight:750;letter-spacing:2px}.subtitle{color:var(--muted);margin-top:10px;max-width:950px}.topline{display:flex;justify-content:space-between;align-items:center;gap:14px}.badge{font-size:12px;color:#22654e;background:#e6f3ed;padding:6px 12px;border-radius:20px;white-space:nowrap}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.stat,.panel{background:var(--paper);border:1px solid var(--line);border-radius:14px}.stat{padding:15px 20px}.stat strong{font-size:27px;margin-right:10px}.muted,small{color:var(--muted)}.layout{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:20px}.panel{padding:24px}.plot-panel{padding:22px 22px 8px}.panel-title{display:flex;justify-content:space-between;align-items:baseline;gap:12px}.panel-title small{font-size:12px}.plot{width:100%;display:block}button,select{font:inherit;border:1px solid #bbcbd5;border-radius:7px;padding:7px 11px;color:var(--ink);background:white;cursor:pointer}button:hover{background:#eef4fa}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #aacced;outline-offset:3px}.tools{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.primary{background:#155da7;border-color:#155da7;color:white}.primary:hover{background:#104d8d}label.field{display:block;font-size:13px;color:var(--muted);margin:16px 0 6px}select{width:100%}.models{margin-top:10px}.model{display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid #eef1f3}.model input{margin-top:6px;accent-color:var(--blue)}.model-name{display:block;font-size:13px;font-weight:650}.model-id{display:block;font-size:11px;overflow-wrap:anywhere}.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.kind{font-size:11px;padding:1px 6px;border-radius:4px;background:#eef3f7;margin-left:6px;white-space:nowrap}.callout{font-size:13px;border-left:3px solid #6e9fc6;padding:10px 12px;background:#f0f6fb;margin-top:16px}.axis-card{margin-top:16px;padding-top:16px;border-top:1px solid var(--line);font-size:13px}.axis-card p{margin-top:5px}.axis-buttons{display:flex;gap:7px;flex-wrap:wrap;margin-top:16px}.axis-buttons button{font-size:12px}.axis-buttons button.active{background:#e8f1fc;border-color:#94b5d9;color:#155da7}.table-panel{margin-top:20px}.scroll{overflow-x:auto;margin-top:14px}table{border-collapse:collapse;min-width:1110px;width:100%;font-size:13px}th,td{padding:13px 11px;border-bottom:1px solid #e5ebef;text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:550;background:#f8fafb}th:first-child,td:first-child{text-align:left}td b{font-weight:650}td small{display:block;font-size:11px}.selected-axis{background:#eff5fc}.source-panel{margin-top:20px;font-size:13px}.source-panel p{margin:8px 0}.source-panel code{overflow-wrap:anywhere}.source-panel summary{cursor:pointer;font-weight:650}.empty{padding:30px;text-align:center;color:var(--muted)}footer{padding:22px 2px;color:var(--muted);font-size:12px}.review{color:#155da7;font-weight:650}.export-note{font-size:12px;color:var(--muted);margin-top:9px}.api-note{background:#fff5e4;border-left-color:#d19b37}.hidden{display:none}.axis-link{cursor:pointer}.axis-link:hover text{fill:#155da7}
@media(max-width:1050px){.layout{grid-template-columns:minmax(0,1fr) 300px}main{padding:20px}.panel{padding:18px}}
@media(max-width:800px){.layout{grid-template-columns:1fr}h1{font-size:27px}.stats{gap:8px}.stat{padding:12px}.stat strong{font-size:23px;display:block}.stat span{font-size:12px}.topline{align-items:flex-start}.model{padding:9px 0}main{padding:14px}.plot-panel{padding:14px 2px}.panel-title{padding:0 12px}.badge{font-size:11px}}
@media print{body{background:white}main{padding:0}.tools,.export-note{display:none}.layout{grid-template-columns:minmax(0,1fr) 300px}.panel{break-inside:avoid}footer{display:none}.stats{margin:12px 0}}
</style></head><body><main>
<div class="topline"><div><div class="eyebrow">VLM EVAL BENCHMARK</div><h1>七轴能力对照</h1></div><span class="badge">离线 · 可交互</span></div>
<p class="subtitle">六个公开测试子集展示能力侧面，第七轴展示本次部署的响应速度。所有轴越向外越好；保留原始正确题数与耗时，便于核查。</p>
<div class="stats"><div class="stat"><strong id="core-count">—</strong><span id="item-label">计分题目</span></div><div class="stat"><strong>6 + 1</strong><span>能力与速度维度</span></div><div class="stat"><strong id="model-count">—</strong><span>已归档模型</span></div></div>
<div class="layout">
<section class="panel plot-panel"><div class="panel-title"><h2>能力轮廓</h2><small>0–100 · 各轴统一向外增加</small></div>
<svg id="radar" class="plot" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 810" role="img" aria-label="七轴模型能力雷达图"></svg>
<div class="axis-buttons" id="axis-buttons" aria-label="查看维度说明"></div>
</section>
<aside class="panel"><h2>选择对照</h2>
<p class="callout" id="grading-note"></p>
<label class="field" for="deployment">部署来源</label><select id="deployment"><option value="all">全部来源</option><option value="local">本地 GPU 部署</option><option value="api">API 服务</option><option value="unknown">未标注环境</option></select>
<div class="tools"><button id="select-all" type="button">全选</button><button id="select-none" type="button">清空</button></div><div id="models" class="models"></div>
<div class="callout" id="speed-note"></div><div class="callout api-note hidden" id="api-note">API 完整耗时含客户端编码、代理、网络和远端处理，使用所选别名的默认思考设置；已有本地基线关闭思考。速度轴反映这些部署配置的响应，不等同于同硬件模型吞吐。</div>
<div class="axis-card"><h3 id="axis-title"></h3><p id="axis-description" class="muted"></p><p id="axis-values"></p></div>
<div class="tools"><button id="export-svg" class="primary" type="button">导出 SVG 图</button><button id="export-json" type="button">导出数据</button></div><p class="export-note">图与数据按当前选择导出。页面本身可单独复制后离线打开。</p>
</aside></div>
<section class="panel table-panel"><div class="panel-title"><h2>原始指标</h2><small id="table-caption"></small></div><div class="scroll"><table><thead id="table-head"></thead><tbody id="table-body"></tbody></table></div><p class="export-note">速度统计覆盖原始全部正式有图请求（包括不计分条目），排除预热和 6 题无图诊断。失败保留在准确率分母；有效计时未覆盖全部正式有图题时不绘制速度分值。</p></section>
<details class="panel source-panel"><summary>数据口径与追溯</summary><p>六项正确率均来自本轮各自固定子集，题数随图显示；不同数据集的评分规则不同。这是内部回归轮廓，不是完整公开榜单，也不使用多边形面积生成总排名。</p><p id="review-note"></p><p id="formula-note"></p><p>速度尺度固定，不随新增模型或勾选项变化。未记录的速度留空，以相邻有效点的线段表示，不补零。API 与本地耗时口径、思考策略和视觉预算可能不同。</p><p>冻结输入 SHA256：<code id="suite-hash"></code></p><div id="contexts"></div><p id="source-info"></p></details>
<footer id="footer"></footer>
</main><script id="dashboard-data" type="application/json">DASHBOARD_PAYLOAD</script><script>
'use strict';
const data=JSON.parse(document.getElementById('dashboard-data').textContent),$=id=>document.getElementById(id);
const colors=['#1768b1','#c56b16','#238467','#9b55ad','#c34364','#517b8e','#787119','#5949b9','#b04e30','#2c7a9a'];
const names={local:'本地 GPU',api:'API 服务',unknown:'未标注环境'};
const state={selected:new Set(data.models.map(m=>m.id)),axis:'MMStar'};
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const svgNode=(tag,attributes={},text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attributes))n.setAttribute(key,String(value));if(text!==undefined)n.textContent=text;return n};
const color=m=>colors[data.models.findIndex(row=>row.id===m.id)%colors.length];
const seconds=v=>Number.isFinite(v)?v.toFixed(3)+' s':'—';
const value=(m,index)=>index===6?m.metrics.speed.score:m.metrics.accuracy[data.axes[index].id].percent;
const visible=()=>data.models.filter(m=>($('deployment').value==='all'||m.group===$('deployment').value));
const selected=()=>visible().filter(m=>state.selected.has(m.id));
const polar=(index,amount,radius=211)=>{const angle=-Math.PI/2+index*2*Math.PI/7;return [460+Math.cos(angle)*radius*amount/100,326+Math.sin(angle)*radius*amount/100]};
const points=(amount,radius)=>data.axes.map((_,i)=>polar(i,amount,radius).join(',')).join(' ');
$('model-count').textContent=data.models.length;$('core-count').textContent=data.core_items;
const publicLabels=data.scoring_protocol==='vlm-public-label-scoring-1';
$('item-label').textContent=publicLabels?'公开标签计分题目':'复核后有效题目';
$('grading-note').textContent=publicLabels?data.grading_note:'统一内容评分 · 已逐题复核';
const datasetCount=id=>data.models[0].metrics.accuracy[id].n;
$('suite-hash').textContent=data.suite_sha256;
$('source-info').textContent='从 '+data.sources.length+' 份已完成评分摘要生成；原始摘要及 SHA256 随归档保留。生成时间：'+data.generated_utc;
$('formula-note').textContent='速度分值 = '+data.speed_formula+'；本页 reference_seconds = '+data.speed_reference_seconds+' s。耗时与准确率互不加权，不计算混合总分。';
$('speed-note').textContent='速度轴：100 × '+data.speed_reference_seconds+' / ('+data.speed_reference_seconds+' + 完整耗时中位数秒数)。例如 '+data.speed_reference_seconds+' 秒为 50 分；越快越外。下表保留中位数和 P90 秒数。';
$('footer').textContent='固定子集 / '+data.schema+' · 结果只读展示，不发送评测请求。';
for(const model of data.models){const p=node('p');p.append(node('strong',model.label+' · '+names[model.group]),node('br'),node('code',JSON.stringify(model.context)));$('contexts').append(p)}
for(const axis of data.axes){const button=node('button',axis.id==='speed'?'响应速度':axis.id);button.type='button';button.dataset.axis=axis.id;button.addEventListener('click',()=>{state.axis=axis.id;render()});$('axis-buttons').append(button)}
function draw(models){
 const svg=$('radar');svg.replaceChildren();svg.setAttribute('style','font-family:system-ui,sans-serif;background:white');
 svg.append(svgNode('title',{},'VLM 七轴能力对照'),svgNode('desc',{},'六个固定子集的正确率和响应速度分值，范围为零到一百。每条曲线代表一个选中模型。'),svgNode('text',{x:460,y:20,'text-anchor':'middle',fill:'#193649','font-size':16,'font-weight':650},'VLM · 六项能力 + 响应速度'));
 for(const tick of [20,40,60,80,100]){svg.append(svgNode('polygon',{points:points(tick),fill:'none',stroke:tick===100?'#b9cbd5':'#e3eaef','stroke-width':1}));svg.append(svgNode('text',{x:469,y:326-211*tick/100+4,fill:'#718591','font-size':11},String(tick)))}
 data.axes.forEach((axis,i)=>{const end=polar(i,100),label=polar(i,100,274),anchor=Math.abs(label[0]-460)<10?'middle':label[0]>460?'start':'end';svg.append(svgNode('line',{x1:460,y1:326,x2:end[0],y2:end[1],stroke:'#d3dfe6','stroke-width':1}));const group=svgNode('g',{'class':'axis-link',role:'button',tabindex:0,'aria-label':'查看'+axis.label});const title=svgNode('text',{x:label[0],y:label[1]-5,'text-anchor':anchor,fill:state.axis===axis.id?'#155da7':'#193649','font-size':17,'font-weight':650},axis.label);group.append(title,svgNode('text',{x:label[0],y:label[1]+16,'text-anchor':anchor,fill:'#667c8c','font-size':12},axis.id==='speed'?'完整响应 · 越快越外':axis.id+' · '+datasetCount(axis.id)+' 题'));const activate=()=>{state.axis=axis.id;render()};group.addEventListener('click',activate);group.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();activate()}});svg.append(group)});
 for(const model of models){const vals=data.axes.map((_,i)=>value(model,i)),stroke=color(model),dash=model.group==='api'?'7 4':'';const curve=svgNode('g',{'data-model':model.id});
  if(vals.every(v=>v!==null)){curve.append(svgNode('polygon',{points:vals.map((v,i)=>polar(i,v).join(',')).join(' '),fill:stroke,'fill-opacity':0.025,stroke,'stroke-width':2.3,'stroke-linejoin':'round','stroke-dasharray':dash}))}
  else{for(let i=0;i<7;i++){const j=(i+1)%7;if(vals[i]!==null&&vals[j]!==null){const a=polar(i,vals[i]),b=polar(j,vals[j]);curve.append(svgNode('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke,'stroke-width':2.3,'stroke-dasharray':dash}))}}}
  vals.forEach((v,i)=>{if(v===null)return;const [cx,cy]=polar(i,v),point=svgNode('circle',{cx,cy,r:3.8,fill:stroke,stroke:'white','stroke-width':1});const metrics=model.metrics,detail=i===6?'速度 '+v.toFixed(1)+' / 100；中位数 '+seconds(metrics.speed.median_seconds)+'；P90 '+seconds(metrics.speed.p90_seconds):metrics.accuracy[data.axes[i].id].correct+'/'+metrics.accuracy[data.axes[i].id].n+'，'+v.toFixed(1)+'%';point.append(svgNode('title',{},model.label+' · '+data.axes[i].label+'：'+detail));curve.append(point)});svg.append(curve)}
 if(!models.length)svg.append(svgNode('text',{x:460,y:330,'text-anchor':'middle',fill:'#667c8c','font-size':18},'请勾选模型开始对照'));
 const legendY=641;models.forEach((model,i)=>{const x=85+(i%2)*390,y=legendY+Math.floor(i/2)*28;svg.append(svgNode('line',{x1:x,y1:y,x2:x+25,y2:y,stroke:color(model),'stroke-width':3,'stroke-dasharray':model.group==='api'?'7 4':''}),svgNode('text',{x:x+35,y:y+4,fill:'#284454','font-size':13},model.label))});
 const foot=legendY+Math.ceil(models.length/2)*28+20;
 svg.append(svgNode('text',{x:460,y:foot,'text-anchor':'middle',fill:'#667c8c','font-size':11},(publicLabels?'公开标签计分 ':'复核有效 ')+data.core_items+' 题 · 各轴题数见标签 · 速度参考 '+data.speed_reference_seconds+' s；不按面积排名'));
 svg.append(svgNode('text',{x:460,y:foot+19,'text-anchor':'middle',fill:'#667c8c','font-size':11},'速度 = 100 × '+data.speed_reference_seconds+' / ('+data.speed_reference_seconds+' + 耗时中位数秒数) · 当前部署响应，跨来源口径不同'));
 svg.setAttribute('viewBox','0 0 920 '+Math.max(780,foot+49));
}
function render(){
 const candidates=visible(),models=selected();
 $('models').replaceChildren();for(const model of candidates){const label=node('label',undefined,'model'),input=node('input');input.type='checkbox';input.checked=state.selected.has(model.id);input.dataset.model=model.id;input.addEventListener('change',()=>{input.checked?state.selected.add(model.id):state.selected.delete(model.id);render()});const text=node('span'),name=node('span',undefined,'model-name'),swatch=node('span',undefined,'swatch');swatch.style.background=color(model);name.append(swatch,document.createTextNode(model.label),node('span',names[model.group],'kind'));text.append(name,node('span',model.id,'model-id'));label.append(input,text);$('models').append(label)}
 if(!candidates.length)$('models').append(node('p','该来源暂无已归档的完整主测结果。','empty'));
 $('api-note').classList.toggle('hidden',!models.some(m=>m.group==='api'));
 $('review-note').textContent=(publicLabels?'全部模型使用同一冻结公开标签与最终答案解析规则，格式差异单列；未修改标签或剔题。补测题速度含首轮超时与补测的累计请求耗时，阶段间等待不计；原始记录均保留。':'全部模型使用同一复核题集与修正标签，格式差异只作标注。逐题依据和原始哈希见评分报告。')+'计分题数 '+data.core_items+'；耗时统计 '+data.run_items+' 道有图请求。';
 for(const button of $('axis-buttons').children)button.classList.toggle('active',button.dataset.axis===state.axis);
 const axis=data.axes.find(a=>a.id===state.axis);$('axis-title').textContent=axis.label+' · '+axis.id;$('axis-description').textContent=axis.description;
 $('axis-values').replaceChildren();for(const model of models){const record=model.metrics,metric=axis.id==='speed'?record.speed:record.accuracy[axis.id];$('axis-values').append(node('div',model.label+'：'+(axis.id==='speed'?(metric.score===null?'未绘制；'+metric.missing_reason:metric.score.toFixed(1)+' 分 · '+seconds(metric.median_seconds)):metric.correct+'/'+metric.n+' · '+metric.percent.toFixed(1)+'%')))}
 draw(models);
 const header=node('tr');header.append(node('th','模型'));for(const a of data.axes.slice(0,6))header.append(node('th',a.id,a.id===state.axis?'selected-axis':''));for(const label of ['速度分值','完整耗时中位数','P90','成功 / 计划'])header.append(node('th',label));$('table-head').replaceChildren(header);$('table-body').replaceChildren();
 for(const model of models){const record=model.metrics,row=node('tr'),name=node('td');name.append(node('b',model.label),node('small',names[model.group]+' · '+model.id));row.append(name);for(const a of data.axes.slice(0,6)){const v=record.accuracy[a.id],td=node('td',undefined,a.id===state.axis?'selected-axis':'');td.append(node('b',v.percent.toFixed(1)+'%'),node('small',v.correct+'/'+v.n));row.append(td)}row.append(node('td',record.speed.score===null?'—':record.speed.score.toFixed(1)),node('td',seconds(record.speed.median_seconds)),node('td',seconds(record.speed.p90_seconds)),node('td',record.core.ok+'/'+record.core.run_n));$('table-body').append(row)}
 if(!models.length){const row=node('tr'),td=node('td','暂无选中模型','empty');td.colSpan=11;row.append(td);$('table-body').append(row)}
 $('table-caption').textContent=models.length+' 个选中模型 · 统一内容评分';
}
function download(filename,type,content){const url=URL.createObjectURL(new Blob([content],{type})),link=node('a');link.href=url;link.download=filename;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),10000)}
$('deployment').addEventListener('change',render);
$('select-all').addEventListener('click',()=>{for(const model of visible())state.selected.add(model.id);render()});$('select-none').addEventListener('click',()=>{for(const model of visible())state.selected.delete(model.id);render()});
$('export-svg').addEventListener('click',()=>download('vlm-seven-axis.svg','image/svg+xml;charset=utf-8',new XMLSerializer().serializeToString($('radar'))));
$('export-json').addEventListener('click',()=>download('vlm-seven-axis-data.json','application/json;charset=utf-8',JSON.stringify({...data,selection:{models:selected().map(m=>m.id),axis:state.axis}},null,2)));
render();
</script></body></html>'''
